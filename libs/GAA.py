import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np

from copy import deepcopy

from libs.network.network import add_dense


class GAA(keras.Model):
    def __init__(
            self, layer_dims: tuple,
            hidden_activation: str = "relu", p_dropout: float = .4,
            target_model: keras.Model = None, n_target: int = 3, n_target_epochs: int = 5,
            lstm_middle: bool = False, second_lstm: bool = True, distributed_norm: bool = True,
            grad_layers: tuple = "all", grad_types: tuple = ("kernel", "bias"),
            normalise_normal: bool = False, trivial_stddev: float = 1.0,
            train_on_gradients: bool = False
    ):
        """
        R2-AD2: Raw gRADient Anomaly Detection
        :param layer_dims: dimensions of the alarm network (some of them will be LSTM elements based on the other options)
        :param hidden_activation: activation for the dense layers
        :param p_dropout: dropout between the alarm layers
        :param target_model: pretrained target model
        :param n_target: number of targets, i.e. retraining steps (1 = just use the given target)
        :param n_target_epochs: retraining epochs for each additional target
        :param lstm_middle: if True let the first alarm layer be a dense layer, otherwise an LSTM
        :param second_lstm: if True let the first LSTM layer be followed by a second one
        :param distributed_norm: distribute the batch norm layer in time, i.e. use the very same weights for each time step
        :param grad_layers: restrict the target layers on which the gradient is calculated
        :param grad_types: restrict the weights on which the gradient is calculated, can be "bias", "kernel" or both
        :param normalise_normal: if True only use the normal samples to train the batch norm layer, else all training samples (without the synthetic anomalies)
        :param trivial_stddev: standard deviation of the Gaussian prior (mean=0.5), if None: no trivial anomalies, if 0: Uniform [0,1]
        :param train_on_gradients: if True the training data is expected to be gradients (saves computing time)
        """
        super(GAA, self).__init__()

        # Save the config
        self.layer_dims = layer_dims
        self.hidden_activation = hidden_activation
        self.p_dropout = p_dropout
        self.normalise_normal = normalise_normal
        self.trivial_stddev = trivial_stddev
        self.lstm_middle = lstm_middle
        self.second_lstm = second_lstm
        self.distributed_norm = distributed_norm and (n_target > 1)
        self.n_target = n_target
        self.n_target_epochs = n_target_epochs
        self.train_on_gradients = train_on_gradients

        # Select which gradients to use
        self.grad_types = grad_types
        self.grad_layers = grad_layers if isinstance(grad_layers, tuple) else (grad_layers, )

        # Save the models
        self.m_target = None
        self.m_targets = []
        self.f_gradients = []
        self.m_alarm = None
        self.m_preprocess = None
        if target_model: self.add_target(target_model)

        # Optimisers
        self.alarm_opt = None
        self.preprocess_opt = None

    def add_target(self, target_model, x_train: np.ndarray, batch_size: int = 64):
        """
        Add target network to GAA
        :param target_model: trained target model
        :param x_train: target's training data
        :param batch_size: target's batch size
        :return:
        """

        self.m_target = target_model

        # Add retrained versions of the target
        input_shape = np.shape(x_train[0])[1:]
        # Train the very same target, but freeze the weights after certain time
        for i_turn in range(self.n_target):
            m_target_copy = deepcopy(self.m_target)
            # Freeze
            m_target_copy = self._freeze_layers(m_target_copy)
            self.m_targets.append(m_target_copy)
            # Build the gradient extractor functions
            self.f_gradients.append(self.get_gradient_on(m_target_copy, input_shape))
            # Train the next target
            if i_turn + 1 < self.n_target:
                print(f"Continue training for target {i_turn+2}")
                self.m_target.fit(
                    x=x_train[0], y=x_train[1], epochs=self.n_target_epochs, batch_size=batch_size, verbose=2
                )

    def _freeze_layers(self, m_in: keras.Model):

        # If given, reduce the gradient to the desired layers
        if "all" not in self.grad_layers:
            # The AE is made of encoder & decoder
            for cur_submodel in m_in.layers:
                if cur_submodel.name not in ["encoder", "decoder", "discriminator"]:
                    continue

                # Freeze everything but the code layer to analyse only its gradient
                for cur_layer in cur_submodel.layers:
                    # For high dimensional data it might be favourable to only look at the intermediate gradients
                    if "middle" in self.grad_layers:
                        if cur_layer.name in ["enc_in", "decoded"]:
                            cur_layer.trainable = False
                    elif "all_but_input" in self.grad_layers:
                        if cur_layer.name == "enc_in":
                            cur_layer.trainable = False
                    elif "all_but_output" in self.grad_layers:
                        if cur_layer.name == "decoded":
                            cur_layer.trainable = False
                    # Only encoder
                    elif "encoder" in self.grad_layers:
                        if cur_submodel.name != "encoder":
                            cur_layer.trainable = False
                    # Only decoder
                    elif "decoder" in self.grad_layers:
                        if cur_submodel.name != "decoder":
                            cur_layer.trainable = False
                    # Specified layers
                    else:
                        if cur_layer.name not in self.grad_layers:
                            cur_layer.trainable = False

        return m_in

    def build(self, input_shape):
        # Introduce a separate model for preprocessing
        m_preprocess = keras.Sequential(name="preprocess")
        # Distribute the batch norm over time
        batch_norm = keras.layers.BatchNormalization() if not self.distributed_norm \
            else keras.layers.TimeDistributed(keras.layers.BatchNormalization())
        m_preprocess.add(batch_norm)
        self.m_preprocess = m_preprocess

        if (self.n_target > 1) and (not self.lstm_middle):
            # LSTM at the beginning
            m_alarm = keras.models.Sequential(name="alarm")
            m_alarm.add(keras.layers.LSTM(
                self.layer_dims[0], return_sequences=self.second_lstm, dropout=self.p_dropout
            ))
            if self.second_lstm:
                m_alarm.add(keras.layers.LSTM(
                    self.layer_dims[1], return_sequences=False, dropout=self.p_dropout
                ))
                add_dense(m_alarm, layer_dims=self.layer_dims[2:], activation=self.hidden_activation, p_dropout=self.p_dropout)
            else:
                add_dense(m_alarm, layer_dims=self.layer_dims[1:], activation=self.hidden_activation, p_dropout=self.p_dropout)
            m_alarm.add(keras.layers.Dense(1, activation="sigmoid"))
        elif (self.n_target > 1) and self.lstm_middle:
            # LSTM after dense layer
            m_alarm = keras.models.Sequential(name="alarm")
            if self.distributed_norm:
                m_alarm.add(keras.layers.TimeDistributed(
                    keras.layers.Dense(self.layer_dims[0])
                ))
                m_alarm.add(keras.layers.TimeDistributed(
                    keras.layers.Dropout(self.p_dropout)
                ))
            else:
                m_alarm.add(keras.layers.Dense(self.layer_dims[0]))
                m_alarm.add(keras.layers.Dropout(self.p_dropout))
            m_alarm.add(keras.layers.LSTM(
                self.layer_dims[1], return_sequences=self.second_lstm, dropout=self.p_dropout
            ))
            if self.second_lstm:
                m_alarm.add(keras.layers.LSTM(
                    self.layer_dims[2], return_sequences=False, dropout=self.p_dropout
                ))
                add_dense(m_alarm, layer_dims=self.layer_dims[3:], activation=self.hidden_activation, p_dropout=self.p_dropout)
            else:
                add_dense(m_alarm, layer_dims=self.layer_dims[2:], activation=self.hidden_activation, p_dropout=self.p_dropout)
            m_alarm.add(keras.layers.Dense(1, activation="sigmoid"))
        else:
            # If there is just one target
            m_alarm = keras.models.Sequential(name="alarm")
            add_dense(m_alarm, layer_dims=self.layer_dims, activation=self.hidden_activation, p_dropout=self.p_dropout)
            m_alarm.add(keras.layers.Dense(1, activation="sigmoid"))
        self.m_alarm = m_alarm

    def compile(
        self, learning_rate: float = .0001, loss: keras.losses.Loss = keras.losses.BinaryCrossentropy(from_logits=False), **kwargs
    ):
        # Let's use Adam
        self.alarm_opt = keras.optimizers.Adam(learning_rate)
        self.preprocess_opt = keras.optimizers.Adam(learning_rate)

        super(GAA, self).compile(loss=loss, **kwargs)

    def get_gradient_on(self, m_target, input_shape):

        # Return a tf.function with the respective target, so that the function is not dynamically built
        @tf.function(input_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float32),))
        def get_gradient(input_sample):
            # We assume that the batch dimension is missing
            cur_el = tf.expand_dims(input_sample, axis=0)

            # Get the loss on the target model
            with tf.GradientTape() as target_tape:
                z_pred = m_target.m_enc(cur_el, training=False)
                target_pred = m_target.m_dec(z_pred, training=False)
                # We try to reconstruct them
                target_loss = m_target.loss(
                    cur_el, target_pred
                )

            # Get the gradient as input for the alarm network
            in_grads = target_tape.gradient(target_loss, m_target.trainable_weights)

            # Filter for the grad type
            use_bias = True if "bias" in self.grad_types else False
            use_kernel = True if "kernel" in self.grad_types else False
            assert use_bias or use_kernel, "Please use the gradients for at least the weights or the biases"
            if not (use_bias and use_kernel):
                in_grads = [cur_grad for i_grad, cur_grad in enumerate(in_grads) if i_grad % 2 == int(use_bias)]

            # Flatten & concatenate
            in_grads = [keras.backend.flatten(cur_grad) for cur_grad in in_grads]
            in_grads = keras.backend.concatenate(in_grads, axis=0)
            # Expand at the time axis
            in_grads = tf.expand_dims(in_grads, axis=0)

            return in_grads

        return get_gradient

    def train_step(
            self, train_data,
    ):
        """
        Train the alarm model
        :param train_data: training data (x, y)
        :return: binary cross entropy loss on the alarm network
        """

        # Extract data
        x_train = train_data[0]
        y_train = train_data[1]
        # Deduce shapes
        batch_size = tf.shape(y_train)[0]
        input_shape = (batch_size, ) + self.m_target.m_enc.input_shape[1:]
        # Counterexamples
        if (self.trivial_stddev is not None) and (self.trivial_stddev > 0):
            # Gaussian
            x_noise = tf.random.normal(input_shape, mean=.5, stddev=self.trivial_stddev)
        else:
            # Uniform
            x_noise = tf.random.uniform(input_shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        y_noise = tf.ones_like(y_train)

        # Distinguish between the training data and noise
        with tf.GradientTape() as alarm_tape:
            # Noise
            y_noise_pred = self(x_noise, train_alarm=True)
            alarm_loss_noise = self.compiled_loss(
                y_pred=y_noise_pred, y_true=y_noise
            )

            # Training data
            if self.train_on_gradients:
                y_train_pred = self.call_predict(x_train, train_alarm=True)
            else:
                y_train_pred = self(x_train, train_alarm=True)
            alarm_loss_train = self.compiled_loss(
                y_pred=y_train_pred, y_true=y_train
            )

            # Total loss is the sum of it
            alarm_loss_tot = alarm_loss_train
            if self.trivial_stddev is not None:
                alarm_loss_tot += alarm_loss_noise

        alarm_grads = alarm_tape.gradient(alarm_loss_tot, self.m_alarm.trainable_weights)
        self.alarm_opt.apply_gradients(
            zip(alarm_grads, self.m_alarm.trainable_weights)
        )

        # Filter anomalies from x_train
        normal_idx = tf.where(tf.equal(y_train, 0))[:, 0]
        x_normal = tf.gather(x_train, normal_idx) if self.normalise_normal else x_train
        y_normal = tf.gather(y_train, normal_idx) if self.normalise_normal else y_train

        # Adapt the preprocessing model
        with tf.GradientTape() as preprocess_tape:
            # Training data
            if self.train_on_gradients:
                y_train_pred = self.call_predict(x_normal, train_preprocess=True)
            else:
                y_train_pred = self(x_normal, train_preprocess=True)
            preprocess_loss_train = self.compiled_loss(
                y_pred=y_train_pred, y_true=y_normal
            )

        preprocess_grads = preprocess_tape.gradient(preprocess_loss_train, self.m_preprocess.trainable_weights)
        self.preprocess_opt.apply_gradients(
            zip(preprocess_grads, self.m_preprocess.trainable_weights)
        )

        return {
            "Alarm Training Loss": alarm_loss_train,
            "Alarm Noise Loss": alarm_loss_noise,
            "Alarm Total Loss": alarm_loss_tot,
            "Preprocess Training Loss": preprocess_loss_train,
        }

    @staticmethod
    def parse_losses(loss_dict: dict) -> str:
        # Turn all losses to numpy
        all_losses = {cur_key: cur_val.numpy() for cur_key, cur_val in loss_dict.items()}
        # Show them as tabbed grid
        out_str = ""
        for cur_key, cur_val in all_losses.items():
            out_str += f"{cur_key}: {cur_val:.4f}\t"

        return out_str

    def call(self, inputs, training=None, mask=None, train_alarm=False, train_preprocess=False):
        assert self.m_target is not None and self.m_alarm is not None, "Please add a target & alarm net first."

        # In case training is set, we likely mean that all parts are trained
        if training:
            train_alarm = train_preprocess = True

        # 1) Get the gradients
        grads_target = self.call_extract_gradients(inputs)

        # 2) Perform batch norm and predict
        y_pred = self.call_predict(grads_target, train_alarm=train_alarm, train_preprocess=train_preprocess)

        return y_pred

    @tf.function()
    def call_extract_gradients(self, inputs):
        # Extract the gradients
        # Note: this is very costly on CNNs in Tensorflow ("Conv2DBackpropFilter uses a while_loop. Fix that!")
        grads_target = [
            tf.vectorized_map(
                cur_func, inputs
            ) for cur_func in self.f_gradients
        ]
        # Direct input for debugging
        # grads_target = self.f_gradients[0](inputs[0, :])
        # Concatenate on the time dimensions
        if self.n_target > 1:
            grads_target_time = keras.backend.concatenate(grads_target, axis=1)
        else:
            # Just one target? Omit the time dimension.
            grads_target_time = grads_target[0][:, 0, :]

        return grads_target_time

    @tf.function()
    def call_predict(self, grads_in, train_alarm=False, train_preprocess=False):
        normalised_grad = self.m_preprocess(grads_in, training=train_preprocess)
        y_pred = self.m_alarm(normalised_grad, training=train_alarm)

        return y_pred

    # -- Helpers --
    @staticmethod
    def describe_gradients(inputs, labels, target, batchnorm=None):
        """
        Choose a random anomalous and a random normal sample and inspect the gradients of trainable layers from target
        :return: final dataframe is we can compute it, else None
        """
        # First, sort out normal and anomalous samples
        normal_indices = tf.where(tf.equal(tf.squeeze(labels), 0))
        anom_indices = tf.where(tf.equal(tf.squeeze(labels), 1))

        # If we don't have any anomalous (or maybe any normal) samples, return None
        if tf.equal(tf.size(normal_indices), 0) or tf.equal(tf.size(anom_indices), 0):
            return None

        normal_idx = tf.random.uniform(shape=[], minval=0, maxval=normal_indices.shape[0], dtype=tf.int32)
        anom_idx = tf.random.uniform(shape=[], minval=0, maxval=anom_indices.shape[0], dtype=tf.int32)
        normal_sample, anom_sample = inputs[tf.squeeze(normal_indices[normal_idx]), :], inputs[tf.squeeze(anom_indices[anom_idx]), :]

        summaries = []

        for cnt, sample in enumerate([normal_sample, anom_sample]):
            type = "normal" if cnt == 0 else "anom"
            cur_el = tf.expand_dims(sample, axis=0)

            # Get the normal loss on the target model
            with tf.GradientTape() as target_tape:
                target_pred = target(cur_el)
                # We try to reconstruct them
                target_loss = keras.losses.BinaryCrossentropy(from_logits=False)(
                    cur_el, target_pred
                )

            # Get the gradient as training data for the alarm network
            grads_normal = target_tape.gradient(target_loss, target.trainable_weights)

            # If we only want to keep biases/weights
            grads_normal = [cur_grad for i_grad, cur_grad in enumerate(grads_normal) if i_grad % 2 == 0]

            # flatten
            grads_target = [keras.backend.flatten(cur_grad) for cur_grad in grads_normal]

            if batchnorm:
                grads_target = tf.concat(grads_target, axis=0)
                grads_target = [tf.squeeze(batchnorm(tf.expand_dims(grads_target, axis=0)))]

            for layer_num, cur_grad in enumerate(grads_target):
                # For each of the three layers (input, code, decoded) apply describe and save them in list
                # cur_grad = tf.math.abs(cur_grad)
                summary = pd.DataFrame({"Data: {}, layer: {}".format(type, layer_num): cur_grad.numpy()})
                summaries.append(summary.describe())

        return pd.concat(summaries, axis=1)

    @staticmethod
    def describe_beta_gamma(beta, gamma):
        """
        Method describing the batchnorm trainable parameters
        :return: Dataframe describing them
        """
        description = []
        beta_description = pd.DataFrame({"Beta": beta.numpy()})
        description.append(beta_description.describe())

        gamma_description = pd.DataFrame({"Gamma": gamma.numpy()})
        description.append(gamma_description.describe())

        return pd.concat(description, axis=1)

