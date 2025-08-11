import tensorflow as tf


class LogisticRegression(tf.keras.layers.Layer):

    def __init__(self, initializer, n_features=None):
        super(LogisticRegression, self).__init__()

        self.initializer = initializer
        self.n_features = n_features

    def build(self, input_shape):

        if self.n_features is None:
            dim_1_shape = input_shape[-1]
        else:
            dim_1_shape = self.n_features

        self.activation_op = tf.keras.activations.sigmoid

        self.kernel = self.add_weight(name="kernel", shape=(dim_1_shape, 1), initializer=self.initializer,
                                      dtype=tf.float32, trainable=True)
        self.bias = self.add_weight(name="bias", shape=(), initializer=self.initializer,
                                    dtype=tf.float32, trainable=True)

    def call(self, inputs, **kwargs):

        outputs = tf.add(tf.matmul(inputs, self.kernel), self.bias)
        return self.activation_op(outputs)

    def get_config(self):

        config = super().get_config()
        config.update({
                "initializer": self.initializer,
                "n_features": self.n_features
            })
        return config
