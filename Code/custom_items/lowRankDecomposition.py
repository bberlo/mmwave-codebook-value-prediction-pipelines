from functools import reduce
import tensorflow as tf
import operator


# Custom TensorFlow implementation of N-mode SVD to make output tensor tractable for logistic regression
# as proposed in Vasilescu et al., "Multilinear Analysis of Image Ensembles: TensorFaces",
# URL: https://link.springer.com/chapter/10.1007/3-540-47969-4_30

# Updated algorithm via operator interlacing as proposed by van Nieuwenhoven et al., "A New Truncation Strategy for the
# Higher-Order Singular Value Decomposition", URL: https://doi.org/10.1137/110836067
class LowRankDecomposition(tf.keras.layers.Layer):

    def __init__(self, decomposition_rank, mode):
        super(LowRankDecomposition, self).__init__()

        self.decomposition_rank = decomposition_rank
        self.mode = mode

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        mode_dim = input_shape.pop(self.mode)
        unfolded_shape = (mode_dim, reduce(operator.mul, input_shape))
        min_dim, max_dim = min(unfolded_shape), max(unfolded_shape)

        if self.decomposition_rank is None:
            self.decomposition_rank = max_dim

        if self.decomposition_rank > max_dim:
            raise ValueError("Trying to compute SVD with decomposition_rank={}, "
                 "which is larger than max(unfolded tensor shape)".format(self.decomposition_rank))

        self.full_matrices = True if self.decomposition_rank > min_dim else False

    # Shape: (Batch size, seq-len, nranges, nthetas)
    def call(self, inputs, **kwargs):

        shape_to_recover = tf.shape(inputs)
        unfold_perm = tf.concat([[self.mode], tf.range(self.mode), tf.range(self.mode + 1, tf.rank(inputs))], axis=0)

        # Matrix SVD for specified mode
        inp_unfolded = tf.reshape(tf.transpose(inputs, perm=unfold_perm), shape=(shape_to_recover[self.mode], -1))
        _, U, _ = tf.linalg.svd(inp_unfolded, full_matrices=self.full_matrices, compute_uv=True)
        U = U[:, :self.decomposition_rank]

        # SVD sign convention
        max_abs_cols = tf.argmax(tf.abs(U), axis=0, output_type=tf.int32)
        u_range = tf.range(start=0, limit=tf.shape(U)[1], delta=1, dtype=tf.int32)
        u_indices = tf.stack([max_abs_cols, u_range], axis=-1)
        signs = tf.expand_dims(tf.sign(tf.gather_nd(params=U, indices=u_indices)), axis=0)
        U = tf.multiply(U, signs)

        # Core tensor multiplication for specified mode (entire multiplication interlaces when layer added in sequence)
        shape_to_recover = tf.tensor_scatter_nd_update(shape_to_recover, indices=[[self.mode]], updates=[tf.shape(U)[0]])
        shape_to_recover = tf.concat([shape_to_recover[:self.mode], shape_to_recover[self.mode + 1:]], axis=0)
        shape_to_recover = tf.concat([[tf.shape(tf.transpose(U))[0]], shape_to_recover], axis=0)

        if self.mode > 0:
            fold_perm = tf.concat([tf.range(1, self.mode + 1), [0], tf.range(self.mode + 1, tf.rank(inputs))], axis=0)
        else:
            fold_perm = tf.range(tf.rank(inputs))

        interm_core_tensor_unfold = tf.matmul(U, inp_unfolded, transpose_a=True)
        interm_core_tensor = tf.transpose(tf.reshape(interm_core_tensor_unfold, shape_to_recover), perm=fold_perm)

        return interm_core_tensor
