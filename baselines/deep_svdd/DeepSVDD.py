import tensorflow as tf

from libs.architecture.target import Autoencoder
from libs.network.network import add_dense


class DeepSVDDAE(Autoencoder):

    def __init__(self, **kwargs):
        # As described in their paper, use LeakyReLU and don't use biases
        super(DeepSVDDAE, self).__init__(hidden_activation="leakyrelu", use_bias=False, **kwargs)

    def compile(self, loss="mean_squared_error", **kwargs):
        super(DeepSVDDAE, self).compile(loss=loss)

    # Based on https://github.com/nuclearboy95/Anomaly-Detection-Deep-SVDD-Tensorflow/blob/master/dsvdd/networks.py
    def _conv_encoder(self, input_shape) -> tf.keras.Model:
        model = tf.keras.models.Sequential(name="encoder")

        model.add(tf.keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=input_shape[1:]))
        model.add(tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
        model.add(tf.keras.layers.LeakyReLU(1e-2))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
        model.add(tf.keras.layers.LeakyReLU(1e-2))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, use_bias=False))

        return model

    def _conv_decoder(self, input_shape, output_shape):
        model = tf.keras.Sequential(name="decoder")

        # The original code expects 2 channels
        # https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/networks/mnist_LeNet.py
        model.add(tf.keras.layers.Reshape((4, 4, 2), input_shape=(input_shape[-1],)))

        # Keras does not have PyTorch's interpolation layers - UpSampling2D seems to be the closest
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2DTranspose(4, (5, 5), use_bias=False))
        model.add(tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
        model.add(tf.keras.layers.LeakyReLU(1e-2))

        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.Conv2DTranspose(8, (5, 5), use_bias=False))
        model.add(tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
        model.add(tf.keras.layers.LeakyReLU(1e-2))

        model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), padding='same', use_bias=False, activation="sigmoid"))

        return model

    def _dense_encoder(self, input_shape):
        model = tf.keras.Sequential(name="encoder")

        add_dense(
            model, layer_dims=self.layer_dims[:-1], p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:], use_bias=self.use_bias
        )
        # Code layer has no activation function
        model.add(tf.keras.layers.Dense(self.layer_dims[-1], name="code", use_bias=self.use_bias))

        return model


class DeepSVDD(tf.keras.Model):
    def __init__(
            self, pretrained_ae: Autoencoder, const_lambda: float = 0.0, name="DeepSVDD", **kwargs
    ):
        """
        Implementation of the paper "Deep one-class classification" originally by Ruff et al.
        :param pretrained_ae: pretrained autoencoder to avoid trivial mappings as proposed in their paper
        :param const_lambda: weighting factor for the weight regulariser (note: they seem not to use this in their code, so it is set to 0)
        :param name: name of the AD method
        :param kwargs:
        """
        super(DeepSVDD, self).__init__(name=name, **kwargs)

        # Models & variables
        self.m_ae: Autoencoder = pretrained_ae
        self.m_map: tf.keras.Model = None
        self.var_c: tf.Tensor = None

        # Constants
        self.const_lambda = const_lambda

    def build(self, input_shape):
        # The encoder will also be the mapping model - plus a flatting operation for 2D data sets
        m_map = tf.keras.models.clone_model(self.m_ae.m_enc)
        self.m_map = tf.keras.Model(
            m_map.inputs, tf.keras.layers.Flatten()(m_map(m_map.inputs))
        )
        # Copy the weights to the model
        self.m_map.set_weights(self.m_ae.m_enc.get_weights())

    @tf.function
    def _dist_to_c(self, x_in):
        # Use the squared distance
        loss_c = tf.square(x_in - self.var_c)
        loss_c = tf.reduce_sum(loss_c, axis=1)

        return loss_c

    @tf.function
    def frobenius_norm(self):
        # Frobenius norm on the layer kernels
        frobenius_loss = 0
        for cur_layer in self.m_map.get_layer("encoder").trainable_weights:
            frobenius_loss += tf.reduce_sum(tf.math.pow(cur_layer, 2))

        return frobenius_loss / 2

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape_map:
            # See where the models maps to
            t_mapped = self(data, training=True)
            # Check the distance to the centre
            loss_map = self._dist_to_c(t_mapped)
            # We'll use the one-class objective as it seems to perform better according to the paper
            loss_map = tf.reduce_mean(loss_map, axis=0)

            # Add the Frobenius regulariser
            # TODO: they don't seem to use it in their code
            # https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
            loss_reg = self.const_lambda * self.frobenius_norm()

            # Total loss
            loss_tot = loss_map + loss_reg

        # Backpropagate
        grad_map = tape_map.gradient(loss_tot, self.m_map.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_map, self.m_map.trainable_weights))

        return {"C-Loss": loss_map, "Frobenius Loss": loss_reg}

    def call(self, inputs, training=None, mask=None):
        return self.m_map(inputs, training=training, mask=mask)

    def compile(self, learning_rate, **kwargs):
        # Use adam
        opt_map = tf.keras.optimizers.Adam(0.001, decay=10e-6)
        super(DeepSVDD, self).compile(optimizer=opt_map)

    def calculate_c(self, x_in, c_eps=.1):
        assert self.m_map is not None, "Please build the model first, e.g. by calling it on the validation data"

        # Calculate c from the pretrained AE
        var_c = tf.reduce_mean(self.m_map(x_in), axis=0)

        # In their code, they recommend shifting the centre to something away from 0
        var_c = tf.where(
            (tf.math.abs(var_c) >= c_eps) | (var_c > 0), var_c, -c_eps
        )
        var_c = tf.where(
            (tf.math.abs(var_c) >= c_eps) | (var_c <= 0), var_c, c_eps
        )

        self.var_c = var_c

    def score(self, x_in):
        # Calculate the distance to c
        x_map = self(x_in)
        dist_c = self._dist_to_c(x_map)

        return dist_c.numpy()


class DeepSAD(DeepSVDD):
    def __init__(self, name="DeepSAD", **kwargs):
        super(DeepSAD, self).__init__(name=name, **kwargs)

    @tf.function
    def _semi_dist_to_c(self, x_in, y_in, eps=1e-6):
        # In their code, they use DeepSVDD's loss first and then add the exponent
        # https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/optim/DeepSAD_trainer.py
        loss_c = self._dist_to_c(x_in)
        # Add DeepSAD's exponent
        loss_c = tf.math.pow(loss_c + eps, y_in)

        return loss_c

    @tf.function
    def train_step(self, data):
        # Extract data
        x_train = data[0]
        y_train = data[1]

        with tf.GradientTape() as tape_map:
            # See where the models maps to
            t_mapped = self(x_train, training=True)
            # Check the distance to the centre
            loss_map = self._semi_dist_to_c(t_mapped, y_train)
            # We'll use the one-class objective as it seems to perform better according to the paper
            loss_map = tf.reduce_mean(loss_map, axis=0)

            # Add the Frobenius regulariser
            # TODO: they don't seem to use it in their code
            # https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/optim/DeepSAD_trainer.py
            loss_reg = self.const_lambda * self.frobenius_norm()

            # Total loss
            loss_tot = loss_map + loss_reg

        # Backpropagate
        grad_map = tape_map.gradient(loss_tot, self.m_map.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_map, self.m_map.trainable_weights))

        return {
            "C-Loss": loss_map,
            "Frobenius Loss": loss_reg
        }
