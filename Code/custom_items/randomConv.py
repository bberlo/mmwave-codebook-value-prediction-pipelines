import tensorflow_probability as tfp
import tensorflow as tf
import functools


# Custom Tensorflow re-implementation of Ignacio Oguiza, tsai, MiniRocket Pytorch, convolution
# URL: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKET_Pytorch.py
class RandomConv(tf.keras.layers.Layer):

    def __init__(self, kernels, channels, kernel_variable, channel_combination,
                 padding, dilation, num_feat_dilation, index):
        super(RandomConv, self).__init__()

        self.kernels = kernels
        self.channels = channels
        self.kernel_variable = kernel_variable
        self.channel_combination_tensor = channel_combination
        self.dilation = dilation
        self.num_feat_dilation = num_feat_dilation
        self.padding = padding
        self.index = index

    # Input_shape: (batch, seq-len, channels)
    def build(self, input_shape):

        self.padding_op = tf.keras.layers.ZeroPadding1D(
            padding=int(self.padding)
        )
        self.conv_op = functools.partial(
            tf.nn.conv1d,
            stride=1,
            padding='VALID',
            data_format='NWC',
            dilations=int(self.dilation),
            name='randConv_{}'.format(self.index)
        )

        self.bias = self.add_weight("bias", shape=(self.kernels, self.num_feat_dilation), dtype=tf.float32,
                                    initializer=tf.keras.initializers.Zeros(), trainable=False)
        self.channel_combination = self.add_weight("channel_combination", shape=self.channel_combination_tensor.shape,
                                    dtype=self.channel_combination_tensor.dtype,
                                    initializer=lambda shape, dtype: self.channel_combination_tensor, trainable=False)

        self.channel_combination_reduced = self.add_weight("channel_reduced", shape=(1, self.channels, 1, 1),
                                                           dtype=self.channel_combination_tensor.dtype, trainable=False,
                                                           initializer=lambda shape, dtype: tf.reduce_sum(self.channel_combination, axis=2, keepdims=True))
        self.kernel_variable_reduced = self.add_weight("kernel_reduced", shape=(9, 1, self.kernels * tf.math.count_nonzero(self.channel_combination_reduced)),
                                                       dtype=self.kernel_variable.dtype, trainable=False,
                                                       initializer=lambda shape, dtype: tf.boolean_mask(tf.reshape(self.kernel_variable, shape=(-1, 1, self.channels, self.kernels)), tf.math.greater(x=tf.tile(input=tf.reshape(self.channel_combination_reduced, shape=(-1,1)),multiples=(1,84)), y=tf.constant([[0.0]])), axis=2))

    def call(self, inputs, training=None, **kwargs):

        _padding1 = self.index % 2

        # Convolution
        inputs_p = tf.boolean_mask(inputs, tf.reshape(self.channel_combination_reduced, shape=(self.channels,)), axis=2)
        inputs_p = self.padding_op(inputs_p)
        inputs_c = self.conv_op(input=inputs_p, filters=self.kernel_variable_reduced)

        yolo5 = tf.boolean_mask(tf.transpose(self.channel_combination, perm=[0, 3, 1, 2]), tf.reshape(self.channel_combination_reduced, shape=(self.channels,)), axis=2)

        if self.channels > 1:

            inputs_c = tf.reshape(inputs_c, shape=(tf.shape(inputs_c)[0], -1, tf.cast(tf.math.count_nonzero(self.channel_combination_reduced), dtype=tf.int32), self.kernels))
            inputs_c = tf.multiply(inputs_c, yolo5)
            inputs_c = tf.reduce_sum(inputs_c, axis=2)

        # Bias
        if training:  # Set bias of Conv output and pass along
            bias_this_dilation = self._get_bias(inputs_c, self.num_feat_dilation)

            tf.debugging.assert_equal(self.bias.shape, bias_this_dilation.shape,
                message="Shape mismatch between bias variable: {} and new bias vector: {}".format(self.bias.shape, bias_this_dilation.shape))

            self.bias.assign(bias_this_dilation)

        else:
            bias_this_dilation = self.bias.value()

        # Features
        _features1 = self._get_ppvs(inputs_c[:, :, _padding1::2], bias_this_dilation[_padding1::2])
        _features2 = self._get_ppvs(inputs_c[:, self.padding:-self.padding, 1-_padding1::2], bias_this_dilation[1-_padding1::2])
        return tf.concat([_features1, _features2], axis=-1)

    def _get_bias(self, inputs, num_features_this_dilation):

        idxs = tf.random.uniform(shape=(self.kernels,), minval=0, maxval=inputs.shape[0], dtype=tf.int32)
        samples = tf.transpose(tf.linalg.diag_part(tf.transpose(tf.gather(params=inputs, indices=idxs, axis=0), perm=[0, 2, 1])))
        biases = tfp.stats.percentile(x=samples, q=self._get_quantiles(num_features_this_dilation), axis=1,
                                      interpolation='linear', keepdims=False)
        return tf.transpose(biases)

    @staticmethod
    def _get_ppvs(inputs, bias):
        inputs = tf.expand_dims(tf.transpose(inputs, perm=[0, 2, 1]), -1)
        bias = tf.reshape(bias, [1, tf.shape(bias)[0], 1, tf.shape(bias)[1]])
        return tf.reshape(tf.reduce_mean(tf.cast(inputs > bias, tf.float32), axis=2), [tf.shape(inputs)[0], -1])

    @staticmethod
    def _get_quantiles(n):
        range_tensor = tf.range(1, n + 1, dtype=tf.float32)
        golden_ratio = (tf.sqrt(5.0) + 1) / 2
        quantiles = (range_tensor * golden_ratio) % 1
        return quantiles * 100
