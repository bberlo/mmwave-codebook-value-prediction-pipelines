from experiment_environment.custom_items.lowRankDecomposition import LowRankDecomposition
from experiment_environment.custom_items.targetMapRecovery import TargetMapRecovery
from experiment_environment.custom_items.logRegression import LogisticRegression
import tensorflow as tf


class OrthMatchPursuit:

    def __init__(self, n_rx, n_adc, rmin, rmax, theta_min, theta_max, rsamp, theta_samp,
                 slow_time_seq_len, backbone_name, num_classes=8, b_size=30, random_state=None):

        # Static defined variables
        self.decomposition_rank = 20
        self.modes = [1, 2, 3]  # dimension 0 assumed to be batch size

        # Dynamic defined variables
        self.rand_state = random_state
        tf.random.set_seed(self.rand_state)

        self.n_rx = n_rx
        self.n_adc = n_adc
        self.seq_len = slow_time_seq_len

        self.rmin = rmin
        self.rmax = rmax
        self.rsamp = rsamp
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_samp = theta_samp

        self.num_classes = num_classes
        self.b_size = b_size
        self.name = backbone_name

    def get_model(self):

        a_input = tf.keras.layers.Input(batch_size=self.b_size, shape=(self.seq_len, self.n_adc, self.n_rx), dtype=tf.complex128)
        input_split = tf.keras.layers.Lambda(function=lambda x: tf.split(tf.transpose(x, perm=[0, 1, 3, 2]), num_or_size_splits=self.b_size, axis=0))(a_input)

        batch_samples_parsed = []
        for elem in input_split:

            # Custom TensorFlow implementation of baseline OMP algorithm as proposed in Mateos-Ramos et al., Model-Based
            # End-to-End Learning Multi-Target Integrated Sensing Communication, URL: https://arxiv.org/abs/2307.04111
            outp_elem = TargetMapRecovery(self.rmin, self.rmax, self.theta_min, self.theta_max,
                                          self.rsamp, self.theta_samp, self.n_adc, self.n_rx)(elem)
            batch_samples_parsed.append(outp_elem)

        outp = tf.keras.layers.Concatenate(axis=0)(batch_samples_parsed)

        # Custom TensorFlow implementation of N-mode SVD to make output tensor tractable for logistic regression
        # as proposed in Vasilescu et al., "Multilinear Analysis of Image Ensembles: TensorFaces",
        # URL: https://link.springer.com/chapter/10.1007/3-540-47969-4_30
        for mode in self.modes:

            outp = LowRankDecomposition(decomposition_rank=self.decomposition_rank, mode=mode)(outp)

        outp = tf.keras.layers.Flatten()(outp)

        bernoulli_probabilities = []
        for _ in range(self.num_classes):
            probability = LogisticRegression(initializer=tf.keras.initializers.HeUniform())(outp)
            bernoulli_probabilities.append(probability)

        output_probabilities = tf.keras.layers.Concatenate(axis=-1)(bernoulli_probabilities)

        return tf.keras.models.Model(a_input, output_probabilities, name=self.name)


# yolo = OrthMatchPursuit(n_rx=4, n_adc=256, rmin=0., rmax=5., theta_min=-60., theta_max=60., rsamp=40,
#                         theta_samp=60, slow_time_seq_len=100, backbone_name='yolo', b_size=16)
# yolo2 = yolo.get_model()
# print(yolo2.summary())
