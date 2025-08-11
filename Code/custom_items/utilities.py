import experiment_environment.custom_items.combis0_8 as cmb9
import experiment_environment.custom_items.combis0_9 as cmb10
import tensorflow as tf
import itertools

# https://github.com/tensorflow/tensorflow/issues/36327
def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(layer, batch_size=batch_size)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

    total_memory = (batch_size * shapes_mem_count + internal_model_mem_count + trainable_count + non_trainable_count)
    return round(total_memory * 1.15)  # To account for 10% discrepancy as indicated by author


# Keras EfficientNet application function retrieved from:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/__init__.py
def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else 1
    input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


# Assumes row-major flattening procedure for creation of domain label
# (see: https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays)
def domain_to_be_left_out_indices_calculation(dim_nr, dim_ind, unflat_domain_label_shape):
    base_dim_range_lists = [list(range(x)) for x in unflat_domain_label_shape]
    base_dim_range_lists[dim_nr] = [dim_ind]

    test_indices = [
        w + unflat_domain_label_shape[-1] * x + unflat_domain_label_shape[-1] * unflat_domain_label_shape[-2] * y +
        unflat_domain_label_shape[-1] * unflat_domain_label_shape[-2] * unflat_domain_label_shape[-3] * z for w in
        base_dim_range_lists[-1] for x in base_dim_range_lists[-2] for y in base_dim_range_lists[-3] for z in
        base_dim_range_lists[-4]]

    # test_indices.sort()
    # print(test_indices)

    return test_indices


def find_nearest_divisible(num, n):
    # Start from the next number
    num += 1

    # Keep incrementing the number until it is divisible by 2 for n times
    while True:
        temp = num
        count = 0

        # Keep dividing the number by 2 until it is no longer divisible
        while temp % 2 == 0:
            temp = temp // 2
            count += 1

        # If the number is divisible by 2 for n times, return the number
        if count >= n:
            return num

        # Otherwise, increment the number and continue the loop
        num += 1


# Substitute for itertools.batched in older Python versions < 3.12
# https://discuss.python.org/t/add-batching-function-to-itertools-module/19357/6
def grouper(iterable, n, *, incomplete='fill', fillvalue=None):

    """Collect data into non-overlapping fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF

    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    elif incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill or ignore')


# Re-implementation of C++ defined torch.combinations method, URL:
# https://github.com/pytorch/pytorch/blob/99af1b3ab03289e922b95252bab51758de6035c9/aten/src/ATen/native/Itertools.cpp
# Note: multiplication, index ops are applied to list of tensors (tf.meshgrid output).
# Therefore, code compiles correctly for structural tensor->op->tensor combinations.
def combinations_pytorch(input_tensor, r, with_replacement):

    tf.debugging.assert_equal(tf.rank(input_tensor), 1,
                              message="Expect a 1D vector, but got shape {}".format(input_tensor.shape))
    tf.debugging.assert_greater_equal(r, 0, message="Expect a non-negative number, but got {}".format(r))

    if r == 0:
        return tf.zeros([0], dtype=input_tensor.dtype)

    num_elements = tf.size(input_tensor)
    input_tensor_list = [input_tensor for _ in range(r)]
    grids = tf.meshgrid(*input_tensor_list, indexing='ij')

    range_tensor = tf.range(num_elements, dtype=tf.int64)
    range_tensor_list = [range_tensor for _ in range(r)]
    index_grids = tf.meshgrid(*range_tensor_list, indexing='ij')
    mask = tf.cast(tf.fill(index_grids[0].shape, True), dtype=tf.int64)

    if with_replacement:
        for i in range(r - 1):
            mask *= tf.cast(tf.greater_equal(index_grids[i], index_grids[i + 1]), dtype=tf.int64)

    else:
        for i in range(r - 1):
            mask *= tf.cast(tf.greater(index_grids[i], index_grids[i + 1]), dtype=tf.int64)

    for i in range(len(grids)):
        grids[i] = tf.boolean_mask(grids[i], mask)

    result = tf.stack(grids, axis=1)

    return tf.sort(tf.reshape(result, [-1, r]), axis=-1)


def combinations(input_tensor, r, with_replacement):

    tf.debugging.assert_equal(tf.rank(input_tensor), 1,
                              message="Expect a 1D vector, but got shape {}".format(input_tensor.shape))
    tf.debugging.assert_greater_equal(r, 0, message="Expect a non-negative number, but got {}".format(r))
    tf.debugging.assert_less_equal(r, tf.size(input_tensor), message="Expect a number <= to {}, but got {}".format(tf.size(input_tensor),r))
    tf.debugging.assert_less_equal(tf.size(input_tensor), 10, message="Due to memory issues, only implements for size 9 or 10, but got {}".format(tf.size(input_tensor)))
    tf.debugging.assert_greater_equal(tf.size(input_tensor), 9, message="Due to memory issues, only implements for size 9 or 10, but got {}".format(tf.size(input_tensor)))

    sz = tf.size(input_tensor)
    if r == 0 or sz < 9:
        return tf.zeros([0], dtype=input_tensor.dtype)
    if sz == 9:
      match r:
        case 1:
           res=cmb9.combi_1_of_9
        case 2:
           res=cmb9.combi_2_of_9
        case 3:
           res=cmb9.combi_3_of_9
        case 4:
           res=cmb9.combi_4_of_9
        case 5:
           res=cmb9.combi_5_of_9
        case 6:
           res=cmb9.combi_6_of_9
        case 7:
           res=cmb9.combi_7_of_9
        case 8:
           res=cmb9.combi_8_of_9
        case 9:
           res=cmb9.combi_9_of_9
        case _:
           res=cmb9.combi_1_of_9
    elif sz == 10:
      match r:
        case 1:
           res=cmb10.combi_1_of_10
        case 2:
           res=cmb10.combi_2_of_10
        case 3:
           res=cmb10.combi_3_of_10
        case 4:
           res=cmb10.combi_4_of_10
        case 5:
           res=cmb10.combi_5_of_10
        case 6:
           res=cmb10.combi_6_of_10
        case 7:
           res=cmb10.combi_7_of_10
        case 8:
           res=cmb10.combi_8_of_10
        case 9:
           res=cmb10.combi_9_of_10
        case 10:
           res=cmb10.combi_10_of_10
        case _:
           res=cmb10.combi_1_of_10
    else:
       res=cmb10.combi_1_of_10
    return tf.gather(input_tensor,res)


def calculate_flattened_indices(shape, dim, dim_idx):
    # Create a tensor of indices with the given shape
    tensor_indices = tf.reshape(tf.range(tf.reduce_prod(shape)), shape)

    # Create a mask that is True at the given dimension index and False elsewhere
    mask = tf.equal(tf.range(shape[dim]), dim_idx)

    # Apply the mask to the tensor indices
    masked_indices = tf.boolean_mask(tensor_indices, mask, axis=dim)

    # Flatten the masked indices
    flattened_indices = tf.reshape(masked_indices, [-1])

    return flattened_indices


def infonce_window_pad_divisible_2(x, data_format, nearest_divisible, half_remainder):

    if data_format == 'ampphase':
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.pad_to_bounding_box(x, half_remainder, 0, nearest_divisible, 256)
        return tf.transpose(x, perm=[1, 2, 0, 3])
    else:
        raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")


def set_masks_dim_idx(shape=None, dtype=None, dim_idx_enc=None, masks=None):

    mask_shape = tf.shape(masks)[1:]

    dim_idx_enc = tf.strings.as_string(dim_idx_enc)
    dim_idx_enc_1 = tf.strings.to_number(tf.strings.substr(dim_idx_enc, pos=0, len=1), out_type=tf.int32)
    dim_idx_enc_2 = tf.strings.to_number(tf.strings.substr(dim_idx_enc, pos=1, len=-1), out_type=tf.int32)

    indices_array = calculate_flattened_indices(mask_shape, dim_idx_enc_1, dim_idx_enc_2)
    indices_array = tf.broadcast_to(tf.reshape(indices_array, shape=(1, -1)),
                                    shape=(tf.shape(masks)[0], tf.size(indices_array)))
    batch_indices = tf.broadcast_to(
        tf.reshape(tf.range(start=0, limit=tf.shape(masks)[0], delta=1, dtype=tf.int32), shape=(-1, 1)),
        shape=(tf.shape(masks)[0], tf.shape(indices_array)[-1]))
    scatter_indices = tf.reshape(tf.stack([batch_indices, indices_array], axis=-1), shape=(-1, 2))
    updates = tf.ones(shape=(tf.shape(scatter_indices)[0],), dtype=tf.int32)

    masks = tf.reshape(masks, shape=(tf.shape(masks)[0], -1))
    masks = tf.tensor_scatter_nd_update(masks, scatter_indices, updates)
    masks = tf.reshape(masks, shape=(tf.shape(masks)[0], *mask_shape))
    return masks
