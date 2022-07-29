import tensorflow as tf
from libs.architecture.target import Autoencoder
from pathlib import Path


class GradConCAE(Autoencoder):
    def __init__(self, pretrained_ae: Autoencoder = None, grad_loss_weight: float = 0.03):
        """
        Implementation of the paper "Backpropagated Gradient Representations for Anomaly Detection" by Kwon et al.
        Paper: https://arxiv.org/abs/2007.09507
        Their implementation: https://github.com/olivesgatech/gradcon-anomaly
        :param pretrained_ae: autoencoder model, which will be used to build GradCon's components
        :param grad_loss_weight: weight on the gradient loss, note that it changes between training and inference
        """
        super(GradConCAE, self).__init__()
        # Saving the gradient history
        self.grad_counters = None
        self.averages = None
        self.gradient_buffer_checkpoint = None

        self.m_ae = pretrained_ae

        self.loss = None
        # Note that TF's cosine similarity is defined as loss, thus has the opposite sign
        self.cosine_similarity = tf.keras.losses.CosineSimilarity()
        self.grad_loss_weight = grad_loss_weight

    def build(self, input_shape, loss=tf.keras.losses.MeanSquaredError()):
        self.m_enc = tf.keras.models.clone_model(self.m_ae.m_enc)
        self.m_dec = tf.keras.models.clone_model(self.m_ae.m_dec)

        # Now that we know the architecture, we also know the shape of the gradients wrt each layer
        self.grad_counters = self.grad_counter_init()
        self.loss = loss

    def call(self, inputs, training=None, mask=None):
        encoded = self.m_enc(inputs, training=training, mask=mask)
        decoded = self.m_dec(encoded, training=training, mask=mask)

        return decoded

    def fit(self, x=None, y=None, batch_size=None, epochs=60, verbose=2, **fit_params):
        # GradCon has some conditional statements that require eager mode
        assert tf.executing_eagerly(), "GradCon can only be used in eager mode"

        data_set = tf.data.Dataset.from_tensor_slices((x, y))
        data_set = data_set.shuffle(10 * batch_size)
        batched_data = data_set.batch(batch_size)

        # Build manually to get gradient shapes in self.grad_counter
        self.build(input_shape=x[0:batch_size].shape)
        gradients = self.grad_counters

        # Print the losses
        all_losses = {}

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            for step, samples in enumerate(batched_data):
                all_losses = self.train_step(samples, gradients)
                recon_grad_dec = all_losses["recon_grad_dec"]

                # After each epoch, update the gradient average
                decoder_layers = int(len(gradients))
                for layer in range(decoder_layers):
                    gradients[layer].update(recon_grad_dec[layer * 2])

            GradConCAE.print_gradcon_losses(all_losses)

        # Save averages as tf.Variable such that tensorflow tracks them
        self.grad_counters = gradients
        self.averages = [tf.Variable(gradient.avg) for gradient in gradients]

    def train_step(self, train_data, gradients):
        x_train = train_data[0]

        # Persistent = true lets you call the gradient multiple times (and inside the current gradient tape)
        with tf.GradientTape(persistent=True) as gradient_tape:
            x_recon = self(x_train, training=True)
            recon_loss = self.loss(y_pred=x_recon, y_true=x_train)

            # First call of the gradient, use it for cosine similarity
            recon_grad_dec = gradient_tape.gradient(recon_loss, self.m_dec.trainable_weights)

            grad_loss = tf.zeros(shape=())

            if gradients[0].count > 0:
                for num_grad, grad in enumerate(recon_grad_dec):
                    if num_grad % 2 == 0:
                        # Compute the cosine similarity
                        grad_loss += self.cosine_similarity(
                            tf.reshape(grad, shape=(-1, 1)),
                            tf.reshape(gradients[num_grad // 2].avg, shape=(-1, 1)))

            grad_loss = grad_loss / len(gradients)

            loss = recon_loss + self.grad_loss_weight * grad_loss

        # Second call of the gradient, for gradient update (backpropagation)
        recon_grad = gradient_tape.gradient(loss, self.m_enc.trainable_weights + self.m_dec.trainable_weights)
        recon_grad_dec = recon_grad[(len(self.m_enc.trainable_weights)):]

        self.optimizer.apply_gradients(zip(recon_grad, self.m_enc.trainable_weights + self.m_dec.trainable_weights))

        return {
            "GradCon reconstruction loss: ": recon_loss,
            "GradCon grad_loss: ": grad_loss,
            "GradCon total_loss: ": loss,
            "recon_grad_dec": recon_grad_dec
        }

    @tf.function
    def compute_anomaly_score(self, x_test, grad_loss_weight: float = .12):
        """
        Compute anomaly scores for all data sample
        :param x_test: Test data
        :param grad_loss_weight: weight factor for the gradient loss ("beta" in the original paper)
        :return: anomaly score for each test sample
        """
        # compute score for each sample
        self.compiled_loss = tf.keras.losses.MeanSquaredError()
        # This causes too much memory consumption for large data sets
        # scores = tf.vectorized_map(
        #     self.get_gradients, x_test, fallback_to_while_loop=True
        # )
        # This is more gentle, but slow... pick your poison!
        scores = tf.map_fn(
            lambda x: self.get_gradients(x, grad_loss_weight=grad_loss_weight), x_test
        )

        return scores

    def get_gradients(self, input_sample, grad_loss_weight: float = .12):
        """
        Helper function for computing the scores
        Takes in a data_sample and computes its anomaly score as dictated by GradCon algorithm
        :param input_sample: one data sample
        :param grad_loss_weight: weight factor for the gradient loss ("alpha" in the original paper)
        :return:
        """
        input_sample = tf.expand_dims(input_sample, axis=0)

        with tf.GradientTape() as prediction_tape:
            prediction = self(input_sample, training=False)
            prediction_loss = self.loss(
                y_pred=prediction, y_true=input_sample
            )

        # Get the gradient for each sample
        prediction_grads = prediction_tape.gradient(prediction_loss, self.m_dec.trainable_weights)

        # History of gradients, averages saved at training time
        ref_grads = self.averages

        grad_loss = tf.zeros(shape=())
        for num_grad in range(len(ref_grads)):
            grad_loss += self.cosine_similarity(
                tf.reshape(prediction_grads[num_grad * 2], shape=(-1, 1)),
                tf.reshape(ref_grads[num_grad], shape=(-1, 1)))
        grad_loss = grad_loss / len(ref_grads)

        score = prediction_loss + grad_loss_weight * grad_loss

        return score

    def grad_counter_init(self):
        """
        Initialize data structure for storing gradient averages
        Used for computing gradient history (gradient averages used for grad_loss)
        return: list with gradient_counter for each decoder layer
        """
        grad_counters = []

        # iterate through layers and initialize everything to 0
        for (i_layer, cur_layer) in enumerate(self.m_dec.trainable_weights):
            if i_layer % 2 == 0:
                grad_counter = GradientCounter()
                grad_counter.avg = tf.zeros_like(cur_layer)
                grad_counters.append(grad_counter)

        return grad_counters

    @staticmethod
    def print_gradcon_losses(losses_dict):
        losses_dict = {loss_key: loss_val
                       for (loss_key, loss_val) in losses_dict.items() if "GradCon" in loss_key}

        losses = ""
        for loss_key, loss_val in losses_dict.items():
            losses += "{}{}  ".format(loss_key.replace(" GradCon", ""), float(loss_val))
        print(losses)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        super(GradConCAE, self).load_weights(filepath)

        # This will load the averages as tf.Variable tracked object (unclear why it works)
        gradients = self.grad_counter_init()
        self.averages = [tf.Variable(gradient.avg) for gradient in gradients]


class GradientCounter:
    def __init__(self):
        """
        Class used for saving gradient history. Use one GradientCounter for each layer gradient
        """
        self.values = 0
        self.avg = 0
        self.count = 0
        self.sum = 0

    def update(self, values, samples=1):
        """
        Update the gradient counter
        :param val: update values
        :param samples: number of samples that get updates, i.e. batch size
        """
        self.values = values
        self.sum += values
        self.count += 1
        self.avg = self.sum / self.count

