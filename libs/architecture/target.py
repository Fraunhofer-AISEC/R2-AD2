import tensorflow as tf
import tensorflow.keras as keras

from typing import List, Tuple

from libs.network.network import add_dense


class Autoencoder(keras.Model):
    def __init__(
            self, p_dropout: float = 0.4,
            hidden_activation: str = "relu", out_activation: str = "sigmoid", layer_dims: List[int] = None,
            use_bias: bool = True, code_dim_override: int = None, name="AE"
    ):
        """
        Create an autoencoder
        :param p_dropout: dropout percentage
        :param hidden_activation: activation function of the hidden layers
        :param out_activation: activation function of the output layer
        :param layer_dims: hidden layer dimensions from the input to the code, if None use a convolutional AE
        :param use_bias: include the bias vector in the layers (e.g. DeepSVDD does not use it)
        """
        super(Autoencoder, self).__init__(name=name)

        # Model config
        self.p_dropout = p_dropout
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.layer_dims = layer_dims
        self.use_bias = use_bias
        self.code_dim_override = code_dim_override

        if (layer_dims is not None) and (code_dim_override is not None):
            # Ok, it was a bad idea to enforce tuples
            new_layer_dims = list(layer_dims)
            new_layer_dims[-1] = code_dim_override
            self.layer_dims = new_layer_dims

        # Layers
        self.m_enc = None
        self.m_dec = None

        # Activation extractors
        self.m_enc_act = None
        self.m_dec_act = None
        self.m_dec_act_on_code = None
        self.m_all_act = None

    # -- Autoencoder Architectures --
    def _conv_encoder(self, input_shape):
        inputs = keras.layers.Input(shape=input_shape[1:])

        x = keras.layers.Conv2D(8, (3, 3), padding='same', name="enc_in")(inputs)
        if self.hidden_activation == "leakyrelu":
            x = keras.layers.LeakyReLU(.01)(x)
        else:
            x = keras.layers.Activation(self.hidden_activation)(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.SpatialDropout2D(self.p_dropout)(x)

        x = keras.layers.Conv2D(
            8, (3, 3), padding='same', name="code" if not isinstance(self.layer_dims, int) else "conv_code"
        )(x)
        if self.hidden_activation == "leakyrelu":
            x = keras.layers.LeakyReLU(.01)(x)
        else:
            x = keras.layers.Activation(self.hidden_activation)(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        z = keras.layers.SpatialDropout2D(self.p_dropout)(x)

        if isinstance(self.layer_dims, int):
            # Dense code space
            x = keras.layers.Flatten()(z)
            x = keras.layers.Dense(self.layer_dims, name="code")(x)
            if self.hidden_activation == "leakyrelu":
                x = keras.layers.LeakyReLU(.01)(x)
            else:
                x = keras.layers.Activation(self.hidden_activation)(x)
            z = keras.layers.Dropout(self.p_dropout)(x)

        model = keras.Model(inputs=inputs, outputs=z, name="encoder")
        return model

    def _conv_decoder(self, input_shape, output_dim):
        encoded = x = keras.Input(shape=input_shape[1:])

        if isinstance(self.layer_dims, int):
            # Dense code space
            x = keras.layers.Dense(512)(x)
            if self.hidden_activation == "leakyrelu":
                x = keras.layers.LeakyReLU(.01)(x)
            else:
                x = keras.layers.Activation(self.hidden_activation)(x)
            x = keras.layers.Dropout(self.p_dropout)(x)
            x = keras.layers.Reshape((8, 8, 8))(x)

        x = keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same')(x)
        if self.hidden_activation == "leakyrelu":
            x = keras.layers.LeakyReLU(.01)(x)
        else:
            x = keras.layers.Activation(self.hidden_activation)(x)
        x = keras.layers.SpatialDropout2D(self.p_dropout)(x)

        # 16x16: for the x-MNIST data sets, we need to scale down slightly to 14x14
        if output_dim[-2] == 28:
            x = keras.layers.Conv2D(8, (3, 3))(x)
            if self.hidden_activation == "leakyrelu":
                x = keras.layers.LeakyReLU(.01)(x)
            else:
                x = keras.layers.Activation(self.hidden_activation)(x)
            x = keras.layers.SpatialDropout2D(self.p_dropout)(x)

        decoded = keras.layers.Conv2DTranspose(filters=output_dim[-1], kernel_size=3, strides=2, padding='same', activation='sigmoid', name="decoded")(x)
        decoder = keras.Model(inputs=encoded, outputs=decoded, name="decoder")
        return decoder

    def _dense_encoder(self, input_shape):
        model = keras.Sequential(name="encoder")

        add_dense(
            model, layer_dims=self.layer_dims[:-1], p_dropout=self.p_dropout, first_name="enc_in",
            activation=self.hidden_activation, input_shape=input_shape[1:], use_bias=self.use_bias
        )
        # We add the last layer manually to name it accordingly
        model.add(keras.layers.Dense(
            self.layer_dims[-1], use_bias=self.use_bias, name="code"
        ))
        if self.hidden_activation == "leakyrelu":
            model.add(keras.layers.LeakyReLU(.01))
        else:
            model.add(keras.layers.Activation(self.hidden_activation))

        return model

    def _dense_decoder(self, input_shape, output_dim):
        model = keras.Sequential(name="decoder")

        add_dense(
            model, layer_dims=list(reversed(self.layer_dims[:-1])), p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:], use_bias=self.use_bias
        )
        # The last layer reconstructs the input
        model.add(keras.layers.Dense(
            self._get_out_size(out_shape=output_dim[1:]), activation=self.out_activation, use_bias=self.use_bias, name="decoded"
        ))
        model.add(keras.layers.Reshape(output_dim[1:]))

        return model

    def _get_out_size(self, out_shape: tuple) -> int:
        """
        Convert the output shape to its dimension
        """
        out_dim = 1
        for cur_dim in out_shape:
            out_dim *= cur_dim

        return out_dim

    # == Keras functions ==
    def build(self, input_shape):
        # Based on the given layers, we use a dense or convolutional AE
        self.m_enc = self._conv_encoder(input_shape) if not isinstance(self.layer_dims, tuple) \
            else self._dense_encoder(input_shape)
        self.m_dec = self._conv_decoder(self.m_enc.output_shape, input_shape) if not isinstance(self.layer_dims, tuple) \
            else self._dense_decoder(self.m_enc.output_shape, input_shape)

        self.build_extractors()

    def compile(self, learning_rate=0.0001, loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=None, **kwargs):
        new_optimizer = tf.keras.optimizers.Adam(learning_rate)
        return super(Autoencoder, self).compile(optimizer=new_optimizer, loss=loss, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=60, verbose=2, **kwargs):
        return super(Autoencoder, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def build_extractors(self):
        # On top, we build the activation extractors
        t_enc_act = keras.layers.Concatenate()([
            keras.layers.Flatten()(cur_layer.output) for cur_layer in self.m_enc.layers
            if isinstance(cur_layer, keras.layers.Activation) or isinstance(cur_layer, keras.layers.LeakyReLU)
        ])
        t_dec_act = keras.layers.Concatenate()([
            keras.layers.Flatten()(cur_layer.output) for cur_layer in self.m_dec.layers
            if isinstance(cur_layer, keras.layers.Activation) or isinstance(cur_layer, keras.layers.LeakyReLU)
        ])
        # Activation extractors
        self.m_enc_act = keras.Model(self.m_enc.inputs, t_enc_act, name="act_enc")
        self.m_dec_act_on_code = keras.Model(self.m_dec.inputs, t_dec_act, name="act_dec_on_code")
        self.m_dec_act = keras.Model(
            self.m_enc.inputs, self.m_dec_act_on_code(self.m_enc(self.m_enc.inputs)), name="act_dec"
        )

        # Concatenating both models gives us all activations
        t_all_act = keras.layers.Concatenate()([
            self.m_enc_act(self.m_enc_act.inputs), self.m_dec_act(self.m_enc_act.inputs)
        ])
        self.m_all_act = keras.Model(
            self.m_enc.inputs, t_all_act, name="act_all"
        )

    def call(self, inputs, training=False, mask=None):
        # Connect the encoder and decoder
        t_encoded = self.m_enc(inputs, training=training, mask=mask)
        t_decoded = self.m_dec(t_encoded, training=training, mask=mask)

        return t_decoded

    def get_config(self):
        config = {
            'p_dropout': self.p_dropout,
            'hidden_activation': self.hidden_activation,
            'out_activation': self.out_activation,
            'layer_dims': self.layer_dims,
        }

        base_config = super(Autoencoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

