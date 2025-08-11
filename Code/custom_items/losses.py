import tensorflow as tf


# Taken from: A Simple Framework for Contrastive Learning of Visual Representations, Chen et al.
# https://proceedings.mlr.press/v119/chen20j.html
@tf.function
def nt_xent_loss(pij, tau):
    batch_size_times_two = tf.shape(pij)[0]

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_projections, right_projections = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=pij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=pij)

    similarity_matrix = -1 \
        * tf.keras.losses.cosine_similarity(right_projections, left_projections, axis=-1) \
        * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    nominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    logit_1 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_1, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_1, tau)), axis=-1)
    ))

    nominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.range(1, batch_size_times_two, delta=2),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    logit_2 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_2, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_2, tau)), axis=-1)
    ))

    return tf.reduce_sum(logit_1 + logit_2) / tf.cast(batch_size_times_two, dtype=tf.float32)


# Taken from: Decoupled Contrastive Learning, Yeh et al.
# https://link.springer.com/chapter/10.1007/978-3-031-19809-0_38
@tf.function
def decoupl_nt_xent_loss(pij, tau):
    batch_size_times_two = tf.shape(pij)[0]

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_projections, right_projections = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=pij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=pij)

    similarity_matrix = -1 \
        * tf.keras.losses.cosine_similarity(right_projections, left_projections, axis=-1) \
        * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    nominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    denominator_1_upd = tf.scatter_nd(
        updates=nominator_1,
        shape=(tf.shape(denominator_1)[0], tf.shape(denominator_1)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1_decoupl = tf.subtract(denominator_1, denominator_1_upd)
    logit_1 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_1, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_1_decoupl, tau)), axis=-1)
    ))

    nominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.range(1, batch_size_times_two, delta=2),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    denominator_2_upd = tf.scatter_nd(
        updates=nominator_2,
        shape=(tf.shape(denominator_2)[0], tf.shape(denominator_2)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2_decoupl = tf.subtract(denominator_2, denominator_2_upd)
    logit_2 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_2, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_2_decoupl, tau)), axis=-1)
    ))

    return tf.reduce_sum(logit_1 + logit_2) / tf.cast(batch_size_times_two, dtype=tf.float32)
