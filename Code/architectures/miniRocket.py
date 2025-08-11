from experiment_environment.custom_items.logRegression import LogisticRegression
from experiment_environment.custom_items.utilities import combinations
from experiment_environment.custom_items.randomConv import RandomConv
import tensorflow as tf


# Custom Tensorflow re-implementation of Ignacio Oguiza, tsai, MiniRocket Pytorch
# URL: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKET_Pytorch.py
class MiniRocket:

    def __init__(self, n_rx, n_adc, ampphase, slow_time_seq_len, backbone_name, num_features=10000,
                 num_classes=8, max_dil_kernel=32, random_state=None):

        # Static defined settings
        self.kernel_size = 9
        self.num_kernels = 84

        # Dynamic defined settings
        self.c_in = n_rx * n_adc * ampphase  # Flat channel number
        self.seq_len = slow_time_seq_len  # Slow-time time-series sequence length
        self.n_features = num_features // self.num_kernels * self.num_kernels  # Output feature number
        self.max_dil = max_dil_kernel
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.rand_state = random_state
        tf.random.set_seed(self.rand_state)

        # Dilations, channel combinations, static kernels
        self._set_dilations(self.seq_len)
        if self.c_in > 1:
            self._set_channel_combinations(self.c_in)
        self.kernel_variable = tf.constant(self._pass_weights(shape=(self.kernel_size, 1, self.num_kernels * self.c_in), dtype=tf.float32))

    def get_model(self):

        batched_2d_input = tf.keras.layers.Input(shape=(self.seq_len, self.c_in))

        output_features = []
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):

            conv_output = RandomConv(kernels=self.num_kernels, channels=self.c_in, kernel_variable=self.kernel_variable,
                                     channel_combination=self.channel_combinations[i], padding=padding,
                                     dilation=dilation, num_feat_dilation=self.num_features_per_dilation[i], index=i)(batched_2d_input)
            output_features.append(conv_output)

        output = tf.keras.layers.Concatenate(axis=-1)(output_features)

        bernoulli_probabilities = []
        for _ in range(self.num_classes):

            probability = LogisticRegression(initializer=tf.keras.initializers.HeUniform(), n_features=self.n_features)(output)
            bernoulli_probabilities.append(probability)

        output_probabilities = tf.keras.layers.Concatenate(axis=-1)(bernoulli_probabilities)

        return tf.keras.models.Model(batched_2d_input, output_probabilities, name=self.backbone_name)

    def _pass_weights(self, shape=None, dtype=None):
        indices = tf.range(self.kernel_size)
        indices = combinations(indices, r=3, with_replacement=False)

        # Add an extra dimension to indices, turn to n-dimensions indices
        indices_last_dim = tf.cast(tf.expand_dims(indices, -1), dtype=tf.int32)
        indices_1st_dim = tf.reshape(tf.range(self.num_kernels, dtype=tf.int32), shape=(self.num_kernels, *tf.clip_by_value(tf.shape(indices_last_dim)[1:], clip_value_min=1, clip_value_max=1)))
        indices_1st_dim = tf.broadcast_to(indices_1st_dim, tf.shape(indices_last_dim))
        indices_2nd_dim = tf.zeros(shape=tf.shape(indices_last_dim), dtype=tf.int32)

        indices = tf.concat([indices_1st_dim, indices_2nd_dim, indices_last_dim], axis=-1)
        indices = tf.reshape(indices, shape=(-1, tf.shape(indices)[-1]))

        # Create weight parameters according to alpha = -1, beta = 2 value structure
        kernels = - tf.ones([self.num_kernels, 1, self.kernel_size])
        kernel_updates = 2 * tf.ones([indices.shape[0]], dtype=tf.float32)
        kernels = tf.tensor_scatter_nd_update(kernels, indices, kernel_updates)
        kernels = tf.tile(kernels, multiples=[self.c_in, 1, 1])
        kernels = tf.transpose(kernels, perm=[2, 1, 0])

        tf.debugging.assert_equal(kernels.shape, shape, message="Shape mismatch between supplied shape argument and shape of 'kernels'.")

        return tf.cast(kernels, dtype=dtype)

    def _set_dilations(self, input_length):
        num_features_per_kernel = self.n_features // self.num_kernels
        true_max_dilations_per_kernel = min(num_features_per_kernel, self.max_dil)
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel
        max_exponent = tf.math.log((input_length - 1) / (9 - 1)) / tf.math.log(2.0)

        dilations = tf.linspace(0.0, max_exponent, num=true_max_dilations_per_kernel)
        dilations = tf.cast(tf.pow(tf.cast(2.0, dilations.dtype), dilations), dtype=tf.int32)
        dilations, _, num_features_per_dilation = tf.unique_with_counts(dilations)

        num_features_per_dilation = tf.cast(tf.cast(num_features_per_dilation, dtype=tf.float32) * multiplier, tf.int32)
        remainder = num_features_per_kernel - tf.reduce_sum(num_features_per_dilation)
        i = 0

        def condition(curr_remainder, *args):
            return tf.greater(curr_remainder, 0)

        def body(curr_remainder, curr_i, curr_num_features_per_dilation):
            curr_num_features_per_dilation = tf.tensor_scatter_nd_add(curr_num_features_per_dilation, [[curr_i]], [1])
            curr_remainder -= 1
            curr_i = (curr_i + 1) % tf.size(curr_num_features_per_dilation)
            return curr_remainder, curr_i, curr_num_features_per_dilation

        remainder, i, num_features_per_dilation = tf.while_loop(condition, body,
                                                                [remainder, i, num_features_per_dilation], parallel_iterations=1)

        self.num_features_per_dilation = num_features_per_dilation
        self.num_dilations = tf.size(dilations)
        self.dilations = dilations
        self.padding = (((self.kernel_size - 1) * dilations) // 2)

    def _set_channel_combinations(self, num_channels):
        num_combinations = self.num_kernels * self.num_dilations
        max_num_channels = float(min(num_channels, 9))  # Changed from default 9 due to large channel nr.
        max_exponent_channels = tf.math.log(max_num_channels + 1.0) / tf.math.log(2.0)
        num_channels_per_combination = tf.cast(2 ** tf.random.uniform([num_combinations], 0, max_exponent_channels), dtype=tf.int32)
        channel_combinations = tf.zeros(shape=(num_combinations, 1, num_channels))

        channel_combinations = tf.map_fn(fn=self.random_choice_no_replacement_2d, elems=(channel_combinations, num_channels_per_combination),
                                         fn_output_signature=tf.TensorSpec(shape=(1, num_channels), dtype=tf.float32))
        channel_combinations = tf.expand_dims(tf.transpose(channel_combinations, perm=[1, 2, 0]), axis=-1)
        channel_combinations = tf.split(channel_combinations,
            num_or_size_splits=[self.num_kernels for _ in range(0, tf.shape(channel_combinations)[2] // self.num_kernels, 1)], axis=2)

        self.channel_combinations = channel_combinations

    # Multi-dimensional re-implementation of PaulG, Custom random without-replacement solution,
    # URL: https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    @staticmethod
    def random_choice_no_replacement_2d(elems):
        two_dim_input = elems[0]
        num_indices_idle = elems[1]
        input_shape = tf.shape(two_dim_input)
        input_length = input_shape[0]

        uniform_distribution = tf.random.uniform(shape=input_shape, minval=0, maxval=None, dtype=tf.float32, name=None)

        # grab the indices of the greatest num_indices_to_change values from the distribution for each row
        _, indices_to_change = tf.math.top_k(uniform_distribution, num_indices_idle)
        sorted_indices_to_change = tf.sort(indices_to_change)
        row_indices = tf.range(input_length)[:, tf.newaxis]
        row_indices = tf.tile(row_indices, [1, num_indices_idle])
        scatter_indices = tf.stack([row_indices, sorted_indices_to_change], axis=-1)

        mask = tf.zeros(input_shape, dtype=tf.float32)

        updates = tf.ones([input_length, num_indices_idle], dtype=tf.float32)

        mask = tf.tensor_scatter_nd_update(mask, scatter_indices, updates)
        zero_mask = tf.cast(tf.equal(mask, 0), tf.float32)

        elements_to_keep = tf.multiply(tf.cast(two_dim_input, dtype=zero_mask.dtype), zero_mask)
        return tf.add(mask, elements_to_keep)


# yolo = MiniRocket(n_rx=4, n_adc=256, ampphase=2, num_features=10000, max_dil_kernel=32,
#                   slow_time_seq_len=100, backbone_name='yolo', random_state=None)
# yolo2 = yolo.get_model()
# print(yolo2.summary())
