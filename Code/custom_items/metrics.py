from experiment_environment.custom_items.utilities import combinations, calculate_flattened_indices, set_masks_dim_idx
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked
import tensorflow_addons as tfa
from typing import Optional
from math import factorial
import tensorflow as tf
import functools


class MultiClassPrecision(tfa.metrics.FBetaScore):
    @typechecked
    def __init__(self, num_classes: FloatTensorLike, average: str = None, threshold: Optional[FloatTensorLike] = None,
            name: str = "precision", dtype: AcceptableDTypes = None, ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(self.weights_intermediate, tf.reduce_sum(self.weights_intermediate))
            precision = tf.reduce_sum(precision * weights)

        elif self.average is not None:  # [micro, macro]
            precision = tf.reduce_mean(precision)

        return precision

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


class MultiClassRecall(tfa.metrics.FBetaScore):
    @typechecked
    def __init__(self, num_classes: FloatTensorLike, average: str = None, threshold: Optional[FloatTensorLike] = None,
            name: str = "recall", dtype: AcceptableDTypes = None, ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def result(self):
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(self.weights_intermediate, tf.reduce_sum(self.weights_intermediate))
            recall = tf.reduce_sum(recall * weights)

        elif self.average is not None:  # [micro, macro]
            recall = tf.reduce_mean(recall)

        return recall

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


# Custom implemented shapley calculation according to https://christophm.github.io/interpretable-ml-book/shapley.html,
# chapter 9.5.3.1, equation 1. Feature subsets are treated as tensor index groups (comparable to super pixels for image data).
# Super pixel example given in 9.6.1 irrespective of methods in Lundberg et al.
# “A unified approach to interpreting model predictions.” Adv. NeurIPS (2017).
# Output metric is absolute contribution averaged across test set as explained in chapter 9.6.5.
class CategoricalShapleyImportance(tf.metrics.Metric):

    def __init__(self, dim=None, index=None, batch_size=None, data_shape=None, model_shape=None, classes=None, last_dims_included=None,
                 order=None, order_dim=None, order_group_size=None, average=None, name='shapley_per_feature', **kwargs):
        super().__init__(name=name, **kwargs)

        total_indices = sum(data_shape[- last_dims_included:])
        sample_sizes = list(range(0, total_indices, 1))
        sample_size_combinations = [factorial(total_indices) / (factorial(x) * factorial(total_indices - x)) for x in sample_sizes]
        masks_batch_size = int(sum(sample_size_combinations))
        self.my_batch_size = batch_size
        sample_sizes_expanded = [item for item, repeat in zip(sample_sizes, sample_size_combinations) for _ in range(int(repeat))]
        weights = [(factorial(x) * factorial(total_indices - x - 1)) / factorial(total_indices) for x in sample_sizes_expanded]

        def coalition_weights(shape=None, dtype=None):
            return tf.convert_to_tensor(weights, dtype=dtype)

        self.coalition_function = functools.partial(
            self.coalition_masks_no_test_feat,
            input_shape=data_shape,
            nr_last_dims_included=last_dims_included,
            dim_test=dim,
            index_test=index
        )
        self.coalition_masks = self.add_weight('coalition_masks', shape=(masks_batch_size, *data_shape),
                                    dtype=tf.int32, initializer=self.coalition_function)

        self.feat_set_function = functools.partial(
            set_masks_dim_idx,
            dim_idx_enc=pow(10, len(data_shape)) * dim + index,
            masks=self.coalition_masks
        )
        self.coalition_masks_feat = self.add_weight('coalition_masks_feat', shape=(masks_batch_size, *data_shape),
                                            dtype=tf.int32, initializer=self.feat_set_function)

        self.coalition_masks_weights = self.add_weight('coalition_masks_weights', shape=(masks_batch_size,),
                                            dtype=tf.float32, initializer=coalition_weights)

        self.average_op = tf.keras.layers.Average()
        self.average = average
        self.dim = dim
        self.order = order
        self.order_dim = order_dim
        self.order_group_size = order_group_size
        self.classes = classes
        self.model_shape = model_shape

        self.metric_state = self.add_weight('shapley_value', shape=(classes,), dtype=tf.float32,
                                            initializer='zeros')
        self.samples = self.add_weight('samples', shape=(), dtype=tf.int32,
                                       initializer='zeros')

    # Other: (batch, slow-time, fast-time, RX, type) - float32
    # OrthMatch: (batch, slow-time, fast-time, RX) - complex128
    # Masks: (combinations, slow-time, fast-time, RX, type) - int32
    def update_state(self, y_true=None, y_pred=None, sample_weight=None, modelref=None, x_true=None):

        group_batch_size = None
        is_complex = False

        if not tf.is_tensor(x_true):
            group_batch_size = tf.shape(x_true[0])[-2]
            split_size = len(x_true)
            x_true = tf.concat(x_true, axis=-2)

        if x_true.dtype == tf.dtypes.complex128:
            x_true_abs = tf.abs(x_true)
            x_true_ang = tf.math.angle(x_true)
            x_true = tf.stack([x_true_abs, x_true_ang], axis=-1)
            is_complex = True

        if self.order:

            tf.debugging.assert_equal(tf.shape(x_true)[self.order_dim], tf.size(self.order),
              message="Structure changes are not allowed due to order array being smaller than x_true dimension"
                      "being ordered. Please adjust self.order array appropriately.")
            tf.debugging.assert_equal(tf.shape(x_true)[self.order_dim] % self.order_group_size, 0,
              message="Size of dimension to be ordered not wholely divisible by group size due to non-zero remainder. "
                      "Please make sure that the ordered dimension is divisible by the group size.")

            x_true = tf.gather(params=x_true, indices=self.order, axis=self.order_dim)
            x_true = tf.reshape(x_true, shape=tf.concat([tf.shape(x_true)[:self.order_dim],
                                                         [self.order_group_size],
                                                         [tf.shape(x_true)[self.order_dim] // self.order_group_size],
                                                         tf.shape(x_true)[self.order_dim + 1:]], axis=0))

        tf.debugging.assert_equal(tf.shape(self.coalition_masks)[1:], tf.shape(x_true)[1:],
                                  message="Data_shape provided to create coalition masks does not match "
                                          "model input shape, excluding batch dimension. Please adjust data_shape to"
                                          "shape after ordering and grouping has taken place.")

        if is_complex:
            masks = tf.cast(self.coalition_masks, dtype=tf.float64)
            masks_feat = tf.cast(self.coalition_masks_feat, dtype=tf.float64)
        else:
            masks = tf.cast(self.coalition_masks, dtype=tf.float32)
            masks_feat = tf.cast(self.coalition_masks_feat, dtype=tf.float32)

        coalition_indices = tf.range(start=0, limit=tf.shape(masks)[0], delta=1, dtype=tf.int32)

        def map_func(elem):
            nonlocal modelref

            if group_batch_size is not None:

                masks_1 = tf.broadcast_to(tf.expand_dims(masks[elem],axis=0), shape=tf.concat([[tf.shape(x_true)[0]], tf.shape(masks[elem])], axis=0))
                yolo_1 = tf.split(tf.multiply(tf.cast(masks_1, dtype=x_true.dtype), x_true), num_or_size_splits=split_size, axis=-2)
                mask_y_pred_1 = modelref(tf.reshape(yolo_1[0], shape=tf.concat([[tf.shape(yolo_1[0])[0]], self.model_shape], axis=0)), training = False)
                mask_y_pred_2 = modelref(tf.reshape(yolo_1[1], shape=tf.concat([[tf.shape(yolo_1[1])[0]], self.model_shape], axis=0)), training = False)
                mask_y_pred = self.average_op([mask_y_pred_1, mask_y_pred_2])

                masks_feat_1 = tf.broadcast_to(tf.expand_dims(masks_feat[elem], axis=0), shape=tf.concat([[tf.shape(x_true)[0]], tf.shape(masks_feat[elem])], axis=0))
                yolo_2 = tf.split(tf.multiply(tf.cast(masks_feat_1, dtype=x_true.dtype), x_true), num_or_size_splits=split_size, axis=-2)
                mask_feat_y_pred_1 = modelref(tf.reshape(yolo_2[0], shape=tf.concat([[tf.shape(yolo_2[0])[0]], self.model_shape], axis=0)), training=False)
                mask_feat_y_pred_2 = modelref(tf.reshape(yolo_2[1], shape=tf.concat([[tf.shape(yolo_2[1])[0]], self.model_shape], axis=0)), training=False)
                mask_feat_y_pred = self.average_op([mask_feat_y_pred_1, mask_feat_y_pred_2])

            else:

                if is_complex:

                    mask_x = tf.multiply(tf.expand_dims(masks[elem], axis=0), x_true)
                    mask_x = tf.unstack(mask_x, axis=-1)
                    mask_x = tf.complex(real=mask_x[0],imag=tf.zeros(shape=tf.shape(mask_x[0]), dtype=tf.float64)) * tf.exp(tf.complex(real=tf.zeros(shape=tf.shape(mask_x[1]),dtype=tf.float64),imag=mask_x[1]))

                    mask_feat_x = tf.multiply(tf.expand_dims(masks_feat[elem], axis=0), x_true)
                    mask_feat_x = tf.unstack(mask_feat_x, axis=-1)
                    mask_feat_x = tf.complex(real=mask_feat_x[0],imag=tf.zeros(shape=tf.shape(mask_feat_x[0]), dtype=tf.float64)) * tf.exp(tf.complex(real=tf.zeros(shape=tf.shape(mask_feat_x[1]), dtype=tf.float64),imag=mask_feat_x[1]))

                    mask_y_pred = modelref(tf.reshape(mask_x, shape=tf.concat([[tf.shape(x_true)[0]], self.model_shape], axis=0)), training=False)
                    mask_feat_y_pred = modelref(tf.reshape(mask_feat_x, shape=tf.concat([[tf.shape(x_true)[0]], self.model_shape], axis=0)), training=False)

                else:

                    mask_y_pred = modelref(tf.reshape(tf.multiply(tf.expand_dims(masks[elem], axis=0), x_true), shape=tf.concat([[tf.shape(x_true)[0]], self.model_shape], axis=0)), training=False)
                    mask_feat_y_pred = modelref(tf.reshape(tf.multiply(tf.expand_dims(masks_feat[elem], axis=0), x_true), shape=tf.concat([[tf.shape(x_true)[0]], self.model_shape], axis=0)), training=False)

            mask_diff_y_pred = tf.multiply(self.coalition_masks_weights[elem], tf.subtract(mask_feat_y_pred, mask_y_pred))
            return mask_diff_y_pred

        diffshape = [self.my_batch_size, self.classes]
        diffspec = tf.TensorSpec(shape=diffshape, dtype=tf.float32)

        diffs = tf.map_fn(fn=map_func, elems=coalition_indices,
                          fn_output_signature=diffspec)

        diffs = tf.abs(tf.reduce_sum(diffs, axis=0))
        self.metric_state.assign_add(tf.reduce_sum(diffs, axis=0))
        self.samples.assign_add(tf.shape(x_true)[0])

    def result(self):

        if self.average:
            return tf.reduce_mean(tf.divide(self.metric_state, tf.cast(self.samples,dtype=tf.float32)))
        else:
            return tf.divide(self.metric_state, tf.cast(self.samples,dtype=tf.float32))

    # Feature coalition function that treats last n dim indices as features included in coalition generation
    @staticmethod
    def coalition_masks_no_test_feat(shape=None, dtype=None, input_shape=None,
                                     nr_last_dims_included=None, dim_test=None, index_test=None):

        if not tf.is_tensor(input_shape):
            input_shape = tf.convert_to_tensor(input_shape, dtype=tf.int32)

        # Generate all potential coalitions with marshal value for shape consistency (including dim_test, index_test)
        def cond(flat_features, shape_tensor, curr_i, max_shape):
            return tf.greater_equal(curr_i, tf.convert_to_tensor(0))

        def body(flat_features, shape_tensor, curr_i, max_shape):
            yolo = tf.range(start=0, limit=shape_tensor[curr_i], delta=1, dtype=tf.int32)
            yolo = tf.cast(yolo + (tf.pow(10, tf.size(shape_tensor)) * curr_i), dtype=tf.int32)
            size_diff = max_shape - tf.size(yolo)
            yolo_sized = tf.expand_dims(tf.concat([yolo, tf.zeros(shape=(size_diff,), dtype=tf.int32)], axis=0), axis=0)
            flat_features_updated = tf.tensor_scatter_nd_update(flat_features, [[curr_i]], yolo_sized)

            return [flat_features_updated, shape_tensor, tf.subtract(curr_i, 1), max_shape]

        end_flat_features, end_shape_tensor, end_i, end_max_shape = tf.while_loop(cond, body, [
            tf.zeros(shape=(tf.size(input_shape), tf.reduce_max(input_shape)), dtype=tf.int32),
            input_shape, tf.size(input_shape) - 1, tf.reduce_max(input_shape)],
                                                                                  parallel_iterations=1,
                                                                                  maximum_iterations=nr_last_dims_included)

        end_flat_features = tf.reshape(end_flat_features, shape=(-1,))

        # Marshal value filtering
        feature_mask_zero = tf.not_equal(end_flat_features, tf.convert_to_tensor([0], dtype=tf.int32))
        end_flat_features = tf.boolean_mask(end_flat_features, feature_mask_zero)

        # Feature under test filtering
        feature_mask_test = tf.not_equal(end_flat_features, [
            index_test + tf.cast((tf.pow(10, tf.size(end_shape_tensor)) * dim_test), dtype=tf.int32)])
        end_flat_features = tf.boolean_mask(end_flat_features, feature_mask_test)

        combination_sizes = tf.range(start=0, limit=tf.reduce_sum(end_shape_tensor[- nr_last_dims_included:]), delta=1,
                                     dtype=tf.int32)
        end_flat_features = tf.broadcast_to(tf.reshape(end_flat_features, shape=(1, -1)),
                                            shape=(tf.size(combination_sizes), tf.shape(end_flat_features)[-1]))
        mask_examples = tf.broadcast_to(
            tf.reshape(tf.zeros(shape=input_shape, dtype=tf.int32), shape=(1, *input_shape)),
            shape=(tf.size(combination_sizes), *input_shape))

        def map_func(elem):
            combination_size = elem[0]
            a_end_flat_features = elem[1]
            a_mask_example = elem[2]

            yolo = tf.strings.as_string(combinations(a_end_flat_features, r=combination_size, with_replacement=False))
            yolo_shape = tf.shape(yolo)

            if tf.size(yolo_shape) < 2:
                yolo_shape = tf.convert_to_tensor([1, 1], dtype=tf.int32)

            yolo_1 = tf.strings.to_number(tf.strings.substr(yolo, pos=0, len=1), out_type=tf.int32)
            yolo_2 = tf.strings.to_number(tf.strings.substr(yolo, pos=1, len=-1), out_type=tf.int32)

            def true_fn(dim, idx, mask_example):
                shapes = tf.broadcast_to(tf.reshape(tf.shape(mask_example), shape=(1, -1)),
                                         shape=(tf.size(dim), tf.size(tf.shape(mask_example))))
                dim = tf.reshape(dim, shape=(-1,))
                idx = tf.reshape(idx, shape=(-1,))
                mask_example = tf.broadcast_to(tf.reshape(mask_example, shape=(1, *tf.shape(mask_example))),
                                               shape=(tf.size(dim), *tf.shape(mask_example)))
                mask_example = tf.reshape(mask_example, shape=(tf.shape(mask_example)[0], -1))

                def map_func_b(elem_b):
                    indices_array = calculate_flattened_indices(elem_b[0], elem_b[1], elem_b[2])
                    updates = tf.ones_like(indices_array, dtype=tf.int32)
                    indices_array = tf.expand_dims(indices_array, axis=-1)

                    return tf.tensor_scatter_nd_update(elem_b[3], indices_array, updates)

                return_tensor = tf.map_fn(fn=map_func_b, elems=(shapes, dim, idx, mask_example),
                                          fn_output_signature=tf.TensorSpec(shape=(tf.shape(mask_example)[-1],),
                                                                            dtype=tf.int32))

                return return_tensor

            def false_fn(dim, idx, mask_example):
                return tf.reshape(mask_example, shape=(1, -1))

            mask_tensor = tf.cond(tf.size(yolo) > 0, true_fn=lambda: true_fn(yolo_1, yolo_2, a_mask_example),
                                  false_fn=lambda: false_fn(yolo_1, yolo_2, a_mask_example))
            mask_tensor = tf.reshape(mask_tensor, shape=(*yolo_shape, -1))
            mask_tensor = tf.reduce_max(mask_tensor, axis=1)
            return mask_tensor

        ragged_masks = tf.map_fn(fn=map_func, elems=(combination_sizes, end_flat_features, mask_examples),
                                 fn_output_signature=tf.RaggedTensorSpec(shape=(None, tf.reduce_prod(input_shape)),
                                                                         ragged_rank=0, dtype=tf.int32))
        ragged_masks = ragged_masks.merge_dims(0, 1)
        ragged_masks = tf.reshape(ragged_masks, shape=(-1, *input_shape))
        return ragged_masks
