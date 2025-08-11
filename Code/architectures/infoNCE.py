from experiment_environment.custom_items.utilities import correct_pad
import tensorflow as tf


# Implementation of EfficientNet B0 architecture, tuning custom architecture not considered
# Changes: 1) without squeeze-excitation module to prevent domain shift mitigation technique influence
# Expected input shape: (batch, pad(seq-len), n_adc, comb(1/2 * n_rx, ampphase))
class MobileNetV2Backbone:

    def __init__(self, backbone_name, input_shape=(224, 256, 4), dropout_rate=0.2):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='normal', seed=None)

        self.input_shape = input_shape
        self.backbone_name = backbone_name
        self.dropout_rate = dropout_rate
        self.drop_count = 0

    def get_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape, name="raw_input")

        x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(inp, 3))(inp)
        x = tf.keras.layers.Conv2D(32, (3, 3), (2, 2), 'valid', activation=None, use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

        # ---------- Mobile block set 1 ----------
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=32,
            outp_filters=16,
            exp_ratio=1,
            strides=1,
            batch_norm=True,
            id_skip=True
        )(x)
        self.drop_count += 1

        # ---------- Mobile block set 2 ----------
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=16,
            outp_filters=24,
            exp_ratio=6,
            strides=2,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=24,
            outp_filters=24,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1

        # ---------- Mobile block set 3 ----------
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=24,
            outp_filters=40,
            exp_ratio=6,
            strides=2,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=40,
            outp_filters=40,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1

        # ---------- Mobile block set 4 ----------
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=40,
            outp_filters=80,
            exp_ratio=6,
            strides=2,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=80,
            outp_filters=80,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=80,
            outp_filters=80,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1

        # ---------- Mobile block set 5 ----------
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=80,
            outp_filters=112,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=112,
            outp_filters=112,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=112,
            outp_filters=112,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1

        # ---------- Mobile block set 6 ----------
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=112,
            outp_filters=192,
            exp_ratio=6,
            strides=2,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=192,
            outp_filters=192,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=192,
            outp_filters=192,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=192,
            outp_filters=192,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)
        self.drop_count += 1

        # ---------- Mobile block set 7 ----------
        x = self.mobilev2_block(
            kernel_size=3,
            inp_filters=192,
            outp_filters=320,
            exp_ratio=6,
            strides=1,
            batch_norm=True,
            id_skip=True,
            drop_rate=self.dropout_rate * self.drop_count / 16
        )(x)

        x = tf.keras.layers.Conv2D(1280, (1, 1), (1, 1), 'same', activation=None, use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x_interm = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        return tf.keras.models.Model(inp, x_interm, name=self.backbone_name)

    def mobilev2_block(self, kernel_size, inp_filters, outp_filters, exp_ratio,
                       strides, id_skip=None, drop_rate=None, batch_norm=None):

            def block(inputs):
                res = inputs

                # Expansion
                x = tf.keras.layers.Conv2D(
                    filters=inp_filters * exp_ratio,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer)(inputs)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)
                x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

                # Depthwise convolution
                x = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    depthwise_initializer=self.kernel_initializer)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)
                x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

                # Output
                x = tf.keras.layers.Conv2D(
                    filters=outp_filters,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)

                if id_skip:

                    if x._type_spec.shape != res._type_spec.shape:

                        res = tf.keras.layers.Conv2D(
                            filters=x._type_spec.shape[-1],
                            kernel_size=(1, 1),
                            strides=strides,
                            padding='same',
                            activation=None,
                            use_bias=True,
                            kernel_initializer=self.kernel_initializer
                        )(res)

                        if batch_norm:
                            res = tf.keras.layers.BatchNormalization(axis=3)(res)

                    x = tf.keras.layers.Add()([x, res])

                if drop_rate:
                    x = tf.keras.layers.Dropout(drop_rate)(x)

                return x

            return block

# model = MobileNetV2Backbone('yolo').get_model()
# print(model.summary())
