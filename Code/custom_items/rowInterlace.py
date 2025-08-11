from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
import tensorflow as tf


class RowInterlace(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RowInterlace, self).__init__(**kwargs)
        self.axis = 0

    @shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.

        if len(input_shape) < 2 or not isinstance(input_shape[0], tuple):
            raise ValueError('A `RowInterlace` layer should be called on a list of '
                             f'at least 2 inputs. Received: input_shape={input_shape}')

        if all(shape is None for shape in input_shape):
            return

        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            shape_set.add(tuple(reduced_inputs_shapes[i]))

        if len(shape_set) != 1:
            err_msg = ('A `RowInterlace` layer requires inputs with matching shapes. '
                       f'Received: input_shape={input_shape}')

            # Make sure all the shapes have same ranks.
            ranks = set(len(shape) for shape in shape_set)
            if len(ranks) != 1:
                raise ValueError(err_msg)

            # Get the only rank for the set.
            (rank,) = ranks
            for axis in range(rank):
                # Skip the Nones in the shape since they are dynamic, also the axis for
                # concat has been removed above.
                unique_dims = set(shape[axis] for shape in shape_set if shape[axis] is not None)
                if len(unique_dims) > 1:
                    raise ValueError(err_msg)

    def call(self, inputs, **kwargs):
        batch_size, zdim = tf.shape(inputs[0])[0], tf.shape(inputs[0])[1]
        z = tf.concat(inputs, axis=self.axis)

        return tf.reshape(
                   tf.transpose(
                       tf.reshape(z, shape=[2, batch_size, zdim]),
                       perm=[1, 0, 2]
                   ),
                   shape=[batch_size * 2, -1]
        )

    def get_config(self):
        config = super(RowInterlace, self).get_config()
        config.update({
            'axis': self.axis
        })
        return config
