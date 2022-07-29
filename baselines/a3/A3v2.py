import tensorflow as tf

from libs.architecture.target import Autoencoder
from libs.network.network import add_dense


class A3(tf.keras.Model):
    def __init__(
            self, layer_dims: tuple, m_target: Autoencoder = None, name="A3",
            hidden_activation: str = "relu", p_dropout: float = .2, use_trivial_anomalies: bool = True
    ):
        """
        Implementation of the paper "Activation Anomaly Analysis" originally by Sperl & Schulze et al.
        :param layer_dims: dimensions of the (fully-connected) alarm network
        :param m_target: pretrained autoencoder used as target network
        :param name: name of the method
        :param hidden_activation: activation function of the alarm network
        :param p_dropout: dropout percentage between the alarm network layers
        :param use_trivial_anomalies: add Gaussian noise as counterexample as done in their paper
        """
        super(A3, self).__init__(name=name)

        # Config
        self.layer_dims = layer_dims
        self.hidden_activation = hidden_activation
        self.p_dropout = p_dropout
        self.use_trivial_anomalies = use_trivial_anomalies

        # Network components
        self.m_ae: Autoencoder = None
        self.m_alarm: tf.keras.Model = None
        if m_target is not None: self.add_target(m_target)

        # Losses
        self.loss_alarm = None

        # Optimiser
        self.opt_alarm: tf.keras.optimizers.Optimizer = None

    # == Helper functions ==
    def add_target(self, m_target: Autoencoder):
        m_target.trainable = False
        self.m_ae = m_target

    # == Keras functions ==
    def compile(
            self,
            learning_rate=0.0001,
            loss_alarm=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            **kwargs
    ):
        super(A3, self).compile(**kwargs)

        self.loss_alarm = loss_alarm

        # We'll use adam as default optimiser
        self.opt_alarm = tf.keras.optimizers.Adam(learning_rate)

    def build(self, input_shape):

        # Alarm & gating use the same hidden dimensions, but other output sizes
        m_alarm = tf.keras.models.Sequential(name="Alarm")
        add_dense(m_alarm, layer_dims=self.layer_dims, activation=self.hidden_activation, p_dropout=self.p_dropout)
        m_alarm.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.m_alarm = m_alarm

    @tf.function
    def train_step(self, data):

        # Extract the data
        x_train = data[0]
        y_train_alarm = data[1]

        # Useful constants
        batch_size = tf.shape(x_train)[0]

        # Trivial anomalies
        x_noise = tf.random.normal(shape=tf.shape(x_train), mean=.5, stddev=1.0)
        y_noise_alarm = tf.keras.backend.ones_like(y_train_alarm)

        # Use the given labels
        with tf.GradientTape() as alarm_tape:
            # Combine them using the gating decision
            y_pred_alarm_train = self(x_train, training=True)
            y_pred_alarm_noise = self(x_noise, training=True)

            # Match the training labels
            loss_alarm = self.loss_alarm(y_true=y_train_alarm, y_pred=y_pred_alarm_train)
            if self.use_trivial_anomalies:
                loss_alarm += self.loss_alarm(y_true=y_noise_alarm, y_pred=y_pred_alarm_noise)

        # Backpropagate
        grad_alarm = alarm_tape.gradient(loss_alarm, self.m_alarm.trainable_weights)
        self.opt_alarm.apply_gradients(zip(grad_alarm, self.m_alarm.trainable_weights))

        return {
            "Alarm": loss_alarm
        }

    def call(self, inputs, training=None, mask=None):

        # The alarm analyses the target's hidden activations
        t_all_acts = self.m_ae.m_all_act(inputs, training=False)
        t_alarm_pred = self.m_alarm(t_all_acts, training=training, mask=mask)

        return t_alarm_pred

    def get_config(self):
        config = {
            'layer_dims': self.layer_dims,
        }

        base_config = super(A3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
