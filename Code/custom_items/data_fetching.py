from experiment_environment.custom_items.utilities import infonce_window_pad_divisible_2
import sklearn.model_selection as sk
import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
import h5py
import math


def fetch_labels_indices(f_path, experiment_type=None, indices=None, fine_tune=False, seed=None, val_set_sampling=False):

    if indices is None and experiment_type == 'domain-leave-out':

        with h5py.File(f_path, 'r') as f:
            dset_1 = f['domain_labels']
            domain_labels = dset_1[:]

        return domain_labels

    if indices is None and experiment_type == 'random':

        with h5py.File(f_path, 'r') as f:
            dset_1 = f['class_labels']
            class_labels = dset_1[:]

        return class_labels

    if indices is not None:

        if fine_tune is True and val_set_sampling is True:
            split_nr = 2
        else:
            split_nr = 5

        with h5py.File(f_path, 'r') as f:
            dset_1 = f['class_labels']
            all_class_labels = dset_1[:]

        class_labels = all_class_labels[indices]
        k_fold_object = sk.StratifiedKFold(n_splits=split_nr, shuffle=True, random_state=seed)
        sparse_class_labels = np.argmax(class_labels, axis=1)

        train_indices, val_indices = next(k_fold_object.split(np.zeros_like(sparse_class_labels), sparse_class_labels))
        return train_indices, val_indices

    raise ValueError("Sampling scenario criteria, according to function parameter setting, were not met."
                     "Criteria are <experiment type>: domain-leave-out, random and <indices>: Array[int].")


def dataset_constructor_infonce(instances, f_path, subset_type, batch_size, data_format, observation_size,
                                observation_size_nearest_divisible, half_remainder, prediction_size,
                                overlap, seed=None, mask=None, order=None, order_dim=None, order_group_size=None):

    if data_format == 'ampphase':
        spec = {'/inputs': tf.TensorSpec(shape=(2000, 256, 4), dtype=tf.complex128),
                '/task_labels': tf.TensorSpec(shape=(2000, 8), dtype=tf.int8),
                '/domain_labels': tf.TensorSpec(shape=(240,), dtype=tf.int8),
                '/class_labels': tf.TensorSpec(shape=(5,), dtype=tf.int8)}
    else:
        raise ValueError("Unknown data_format. Allowed values: ampphase")

    blockage_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    blockage_inputs = blockage_hdf5('/inputs')
    blockage_task_labels = blockage_hdf5('/task_labels')

    if not isinstance(overlap, float) or not 0.0 <= overlap <= 1.0:
        raise ValueError("Overlap is a type float denoting window overlap probability (0-1)")
    else:
        overlap = 1.0 - overlap

    if mask is not None and not tf.is_tensor(mask):
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def get_frame(instance):

        inputs_complex_128 = blockage_inputs[instance]
        inputs_abs = tf.abs(inputs_complex_128)
        inputs_ang = tf.math.angle(inputs_complex_128)
        inputs = tf.cast(tf.stack([inputs_abs, inputs_ang], axis=-1), dtype=tf.float32)

        return inputs, blockage_task_labels[instance]

    def antenna_combination(dataset, size):

        def create_view_comb_batch(batch_inputs, batch_labels):

            reduced_input_batch_v1 = batch_inputs[:, :, :, :2, :]
            reduced_input_batch_v2 = batch_inputs[:, :, :, 2:, :]

            return (reduced_input_batch_v1, reduced_input_batch_v2), batch_labels

        dataset = dataset.batch(size).map(create_view_comb_batch).prefetch(20)
        return dataset

    def split_observation_prediction(dataset):

        def split_op(batch_inputs, batch_labels):

            observation_window = batch_inputs[:observation_size]
            prediction_window = batch_labels[observation_size:]

            return observation_window, prediction_window

        dataset = dataset.map(split_op)
        return dataset

    def apply_mask(x, mask):

        x_true_shape = tf.shape(x)
        if order:

            x_true = tf.gather(params=x, indices=order, axis=order_dim)
            x_true = tf.reshape(x_true, shape=tf.concat([tf.shape(x_true)[:order_dim], [order_group_size],
                                                         [tf.shape(x_true)[order_dim] // order_group_size],
                                                         tf.shape(x_true)[order_dim + 1:]], axis=0))
        else:
            x_true = x

        x_true = tf.multiply(tf.cast(mask,dtype=x_true.dtype), x_true)

        if order:
            x_true = tf.reshape(x_true, shape=x_true_shape)

        return x_true

    dset = tf.data.Dataset\
             .from_tensor_slices(instances)

    if subset_type == 'pre-train':

        dset = dset\
                 .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True, seed=seed)\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size, shift=math.floor(observation_size * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size), y.batch(observation_size))))\

        if mask is not None:

            dset = dset\
                     .map(lambda x, y: (apply_mask(x, mask), y))\

        dset = dset\
                 .map(lambda x, y: (infonce_window_pad_divisible_2(x, data_format=data_format, nearest_divisible=observation_size_nearest_divisible, half_remainder=half_remainder), y))\
                 .apply(transformation_func=lambda curr_dset: antenna_combination(curr_dset, batch_size))

    elif subset_type == 'pre-train-val':

        dset = dset\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size, shift=math.floor(observation_size * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size), y.batch(observation_size))))\
                 .map(lambda x, y: (infonce_window_pad_divisible_2(x, data_format=data_format, nearest_divisible=observation_size_nearest_divisible, half_remainder=half_remainder), y))\
                 .apply(transformation_func=lambda curr_dset: antenna_combination(curr_dset, batch_size))

    elif subset_type == 'fine-tune':

        dset = dset\
                 .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True, seed=seed)\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size + prediction_size, shift=math.floor((observation_size + prediction_size) * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size + prediction_size), y.batch(observation_size + prediction_size))))\
                 .apply(lambda curr_dset: split_observation_prediction(curr_dset))

        if mask is not None:

            dset = dset\
                     .map(lambda x, y: (apply_mask(x, mask), y))\

        dset = dset\
                 .map(lambda x, y: (infonce_window_pad_divisible_2(x, data_format=data_format, nearest_divisible=observation_size_nearest_divisible, half_remainder=half_remainder), y))\
                 .apply(transformation_func=lambda curr_dset: antenna_combination(curr_dset, batch_size))

    elif subset_type == 'fine-tune-val' or subset_type == 'test':

        dset = dset\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size + prediction_size, shift=math.floor((observation_size + prediction_size) * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size + prediction_size), y.batch(observation_size + prediction_size))))\
                 .apply(lambda curr_dset: split_observation_prediction(curr_dset))\
                 .map(lambda x, y: (infonce_window_pad_divisible_2(x, data_format=data_format, nearest_divisible=observation_size_nearest_divisible, half_remainder=half_remainder), y))\
                 .apply(transformation_func=lambda curr_dset: antenna_combination(curr_dset, batch_size))

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: pre-train(-val), fine-tune(-val), test.")

    return dset


def dataset_constructor_minirocket(instances, f_path, subset_type, batch_size, data_format, observation_size, prediction_size,
                                   overlap, seed=None, mask=None, order=None, order_dim=None, order_group_size=None):

    if data_format == 'ampphase':
        spec = {'/inputs': tf.TensorSpec(shape=(2000, 256, 4), dtype=tf.complex128),
                '/task_labels': tf.TensorSpec(shape=(2000, 8), dtype=tf.int8),
                '/domain_labels': tf.TensorSpec(shape=(240,), dtype=tf.int8),
                '/class_labels': tf.TensorSpec(shape=(5,), dtype=tf.int8)}
    else:
        raise ValueError("Unknown data_format. Allowed values: ampphase")

    blockage_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    blockage_inputs = blockage_hdf5('/inputs')
    blockage_task_labels = blockage_hdf5('/task_labels')

    if not isinstance(overlap, float) or not 0.0 <= overlap <= 1.0:
        raise ValueError("Overlap is a type float denoting window overlap probability (0-1)")
    else:
        overlap = 1.0 - overlap

    if mask is not None and not tf.is_tensor(mask):
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def get_frame(instance):

        inputs_complex_128 = blockage_inputs[instance]
        inputs_abs = tf.abs(inputs_complex_128)
        inputs_ang = tf.math.angle(inputs_complex_128)
        inputs = tf.cast(tf.stack([inputs_abs, inputs_ang], axis=-1), dtype=tf.float32)

        return inputs, blockage_task_labels[instance]

    def split_observation_prediction(dataset):

        def split_op(batch_inputs, batch_labels):

            observation_window = batch_inputs[:observation_size]
            prediction_window = batch_labels[observation_size:]

            return observation_window, prediction_window

        dataset = dataset.map(split_op)
        return dataset

    def apply_mask(x, mask):

        x_true_shape = tf.shape(x)
        if order:

            x_true = tf.gather(params=x, indices=order, axis=order_dim)
            x_true = tf.reshape(x_true, shape=tf.concat([tf.shape(x_true)[:order_dim], [order_group_size],
                                                         [tf.shape(x_true)[order_dim] // order_group_size],
                                                         tf.shape(x_true)[order_dim + 1:]], axis=0))
        else:
            x_true = x

        x_true = tf.multiply(tf.cast(mask,dtype=x_true.dtype), x_true)

        if order:
            x_true = tf.reshape(x_true, shape=x_true_shape)

        return x_true

    dset = tf.data.Dataset\
             .from_tensor_slices(instances)

    if subset_type == 'train':

        dset = dset\
                 .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True, seed=seed)\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size + prediction_size, shift=math.floor((observation_size + prediction_size) * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size + prediction_size), y.batch(observation_size + prediction_size))))\
                 .apply(lambda curr_dset: split_observation_prediction(curr_dset))

        if mask is not None:

            dset = dset\
                     .map(lambda x, y: (apply_mask(x, mask), y))

        dset = dset\
                 .batch(batch_size=batch_size, drop_remainder=True)\
                 .prefetch(20)

    elif subset_type == 'val' or subset_type == 'test':

        dset = dset\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size + prediction_size, shift=math.floor((observation_size + prediction_size) * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size + prediction_size), y.batch(observation_size + prediction_size))))\
                 .apply(lambda curr_dset: split_observation_prediction(curr_dset))\
                 .batch(batch_size=batch_size, drop_remainder=True)\
                 .prefetch(20)

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: train, val, test.")

    return dset


def dataset_constructor_orthmatchpursuit(instances, f_path, subset_type, batch_size, data_format, observation_size, prediction_size,
                                         overlap, seed=None, mask=None, order=None, order_dim=None, order_group_size=None):

    if data_format == 'ampphase':
        spec = {'/inputs': tf.TensorSpec(shape=(2000, 256, 4), dtype=tf.complex128),
                '/task_labels': tf.TensorSpec(shape=(2000, 8), dtype=tf.int8),
                '/domain_labels': tf.TensorSpec(shape=(240,), dtype=tf.int8),
                '/class_labels': tf.TensorSpec(shape=(5,), dtype=tf.int8)}
    else:
        raise ValueError("Unknown data_format. Allowed values: ampphase")

    blockage_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    blockage_inputs = blockage_hdf5('/inputs')
    blockage_task_labels = blockage_hdf5('/task_labels')

    if not isinstance(overlap, float) or not 0.0 <= overlap <= 1.0:
        raise ValueError("Overlap is a type float denoting window overlap probability (0-1)")
    else:
        overlap = 1.0 - overlap

    if mask is not None and not tf.is_tensor(mask):
        mask = tf.convert_to_tensor(mask, dtype=tf.float64)

    def get_frame(instance):

        return blockage_inputs[instance], blockage_task_labels[instance]

    def split_observation_prediction(dataset):

        def split_op(batch_inputs, batch_labels):

            observation_window = batch_inputs[:observation_size]
            prediction_window = batch_labels[observation_size:]

            return observation_window, prediction_window

        dataset = dataset.map(split_op)
        return dataset

    def apply_mask(x, mask):

        x_abs = tf.abs(x)
        x_ang = tf.math.angle(x)
        x_true = tf.stack([x_abs, x_ang], axis=-1)
        x_true_shape = tf.shape(x)

        if order:

            x_true = tf.gather(params=x_true, indices=order, axis=order_dim)
            x_true = tf.reshape(x_true, shape=tf.concat([tf.shape(x_true)[:order_dim], [order_group_size],
                                                         [tf.shape(x_true)[order_dim] // order_group_size],
                                                         tf.shape(x_true)[order_dim + 1:]], axis=0))

        x_true = tf.multiply(tf.cast(mask,dtype=x_true.dtype), x_true)
        x_true = tf.unstack(x_true, axis=-1)
        x_true = tf.complex(real=x_true[0], imag=tf.zeros(shape=tf.shape(x_true[0]), dtype=tf.float64)) * tf.exp(tf.complex(real=tf.zeros(shape=tf.shape(x_true[1]), dtype=tf.float64), imag=x_true[1]))

        x_true = tf.reshape(x_true, x_true_shape)
        return x_true

    dset = tf.data.Dataset\
             .from_tensor_slices(instances)

    if subset_type == 'train':

        dset = dset\
                 .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True, seed=seed)\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size + prediction_size, shift=math.floor((observation_size + prediction_size) * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size + prediction_size), y.batch(observation_size + prediction_size))))\
                 .apply(lambda curr_dset: split_observation_prediction(curr_dset))

        if mask is not None:

            dset = dset\
                     .map(lambda x, y: (apply_mask(x, mask), y))

        dset = dset\
                 .batch(batch_size=batch_size, drop_remainder=True)\
                 .prefetch(20)

    elif subset_type == 'val' or subset_type == 'test':

        dset = dset\
                 .repeat()\
                 .map(lambda x: get_frame(x))\
                 .unbatch()\
                 .window(size=observation_size + prediction_size, shift=math.floor((observation_size + prediction_size) * overlap), stride=1, drop_remainder=True)\
                 .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(observation_size + prediction_size), y.batch(observation_size + prediction_size))))\
                 .apply(lambda curr_dset: split_observation_prediction(curr_dset))\
                 .batch(batch_size=batch_size, drop_remainder=True)\
                 .prefetch(20)

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: train, val, test.")

    return dset


# instances = list(range(100))
# f_path = r'C:\TUe-PhD\UT-RS-3 Domain-Invariant Beam Adaptivity\experiment_environment\Datasets\codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
#
# dataset = dataset_constructor_infonce(instances, f_path, subset_type='pre-train', batch_size=12, data_format='ampphase', observation_size=400, observation_size_nearest_divisible=416, half_remainder=8, prediction_size=50, overlap=0.2)
# dataset = dataset_constructor_orthmatchpursuit(instances, f_path, subset_type='train', batch_size=12, data_format='ampphase', observation_size=400, prediction_size=50, overlap=0.2)
# dataset = dataset_constructor_minirocket(instances, f_path, subset_type='train', batch_size=12, data_format='ampphase', observation_size=400, prediction_size=50, overlap=0.2)
#
# for element in dataset:
#     print(element)
#     break
