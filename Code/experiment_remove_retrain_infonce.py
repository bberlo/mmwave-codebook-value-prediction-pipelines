from experiment_environment.custom_items.data_fetching import fetch_labels_indices, dataset_constructor_infonce
from experiment_environment.custom_items.utilities import find_nearest_divisible, grouper, set_masks_dim_idx
from experiment_environment.custom_items.metrics import MultiClassPrecision, MultiClassRecall
from experiment_environment.custom_items.anyLabelFinetuneModel import AnyLabelModel
from experiment_environment.custom_items.logRegression import LogisticRegression
from experiment_environment.custom_items.callbacks import WeightRestoreCallback
from experiment_environment.architectures.infoNCE import MobileNetV2Backbone
from experiment_environment.custom_items.pretrainModel import PretrainModel
from experiment_environment.custom_items.losses import decoupl_nt_xent_loss
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import itertools
import argparse
import datetime
import pickle
import os

# With feature 'superpixels' ranking, indexed by dimension and dimension index, based on Shapley importance value,
# successively remove features from input pipeline highrank-lowrank order with mask. Re-train with resulting pipeline
# and evaluate on old test input pipeline. Inspiration was drawn from Hooker et al.,
# "A Benchmark for Interpretability Methods in Deep Neural Networks", Adv. Neural Inform. Proc. Sys. 32 (NeurIPS 2019)

parser = argparse.ArgumentParser(description="Experiment automation setup script.")
parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
parser.add_argument('-o_s', '--observation_size', type=int, help='<Required> Size observation portion overlapping windows', required=True)
parser.add_argument('-p_s', '--prediction_size', type=int, help='<Required> Size prediction portion overlapping windows', required=True)
parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment', required=True)
parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=True)
parser.add_argument('-s', '--seed', type=int, help='<Required> Random seed used for random ops in the experiment', required=True)
parser.add_argument('-l_r', '--learning_rate', type=float, help='<Required> Learning rate to be used in the experiment', required=True)
parser.add_argument('-t', '--tau', type=float, help='<Required> Tau (temperature) used for ntxent loss in the experiment', required=True)
parser.add_argument('-o_l', '--overlap', type=float, help='<Required> Percentage 0-1 subsequent windows should overlap after dataset sample stitching', required=True)
parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
parser.add_argument('-f_p', '--file_path', help='<Required> File path to be used in the experiment', required=True)
parser.add_argument('-r_p', '--result_path', help='<Required> File path of Shapley importance results to be used in the experiment', required=True)
args = parser.parse_args()

# GPU config. for allocating limited amount of memory on a given device
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        tf.config.experimental.set_visible_devices(gpus[args.gpu], "GPU")
    except RuntimeError as e:
        print(e)

# Set logging level
tf.get_logger().setLevel("WARNING")

# Load training, validation, and test data
with open(args.file_path, 'rb') as handle:
    indices_types_dict = pickle.load(handle)

importance_results = pd.read_csv(args.result_path)
importance_results = importance_results.sort_values(by=importance_results.index[0], axis=1, ascending=False, inplace=False)
sort_importance_results_headers = importance_results.columns.values
dim_idx = [[int(x.split('_')[1]), int(x.split('_')[2])] for x in sort_importance_results_headers[:-1]]
accum_dim_idx = list(itertools.accumulate(dim_idx))
accum_dim_idx = [list(grouper(x, 2, incomplete='ignore')) for x in accum_dim_idx]

divisible_observation_size = find_nearest_divisible(args.observation_size, 5)
half_remainder = (divisible_observation_size - args.observation_size) // 2
data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
input_shape = (divisible_observation_size, 256, 4)
ft_shape = input_shape
frame_size = 2000
step_multiplier = frame_size // (args.observation_size + args.prediction_size)
class_nr = 8

train_instance_indices, val_instance_indices = fetch_labels_indices(f_path=data_fetch_path, indices=indices_types_dict["train_indices"], seed=args.seed)
test_instances = indices_types_dict["test_indices"]
os.remove(args.file_path)

_, fine_train_instance_indices = fetch_labels_indices(f_path=data_fetch_path, indices=indices_types_dict["train_indices"][train_instance_indices],
                                                      fine_tune=True, seed=args.seed)
_, fine_val_instance_indices = fetch_labels_indices(f_path=data_fetch_path, indices=indices_types_dict["train_indices"][val_instance_indices],
                                                    fine_tune=True, seed=args.seed, val_set_sampling=True)

# Mask variables
data_shape = (args.observation_size, 64, 4, 4, 2)  # Doesn't include batch dimension
data_shape_mask = (1, 64, 4, 4, 2)

order = list(range(0, 256, 1))  # Doesn't include batch dimension because dataset constructor applies prior to batching
order_group_size = 64
order_dim = 1

mask_list = []
for dim_idx_remove in accum_dim_idx:

    dim_idx_remove = tf.convert_to_tensor(dim_idx_remove, dtype=tf.int32)
    mask = tf.zeros(shape=(tf.shape(dim_idx_remove)[0], *data_shape_mask), dtype=tf.int32)

    mask_updated = tf.map_fn(fn=lambda x: set_masks_dim_idx(dim_idx_enc=tf.convert_to_tensor(pow(10, len(data_shape)), dtype=tf.int32) * x[1][0] + x[1][1], masks=tf.expand_dims(x[0], axis=0)),
                             elems=(mask, dim_idx_remove), fn_output_signature=tf.TensorSpec(shape=(1, *mask.shape[1:]), dtype=mask.dtype))
    mask_updated = tf.squeeze(mask_updated, axis=1)
    mask_updated = tf.reduce_max(mask_updated, axis=0, keepdims=False)
    mask_updated = tf.subtract(tf.ones_like(mask_updated), mask_updated)
    mask_list.append(mask_updated)

# Evaluation results for successive left out input features procedure
masked_results = []
for curr_mask in mask_list:

    # -------------------------------------------- Pre-training -------------------------------------------------------
    inp = tf.keras.layers.Input(shape=input_shape)
    cnn_extractors = MobileNetV2Backbone(input_shape=input_shape, backbone_name=args.model_name + "_backbone").get_model()
    enc = cnn_extractors(inp)

    proj = tf.keras.layers.Dense(450, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(enc)
    proj = tf.keras.layers.Activation(tf.keras.activations.relu)(proj)
    proj = tf.keras.layers.Dense(300, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(proj)
    proj = tf.keras.layers.Activation(tf.keras.activations.relu)(proj)
    proj = tf.keras.layers.Dense(150, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(proj)

    pre_train_model = PretrainModel(inp, proj, name=args.model_name)
    pre_train_model.compile(optimizer=tfa.optimizers.SGDW(weight_decay=1e-6, learning_rate=args.learning_rate * 5, momentum=0.95, nesterov=True),
                           sim_loss_fn=decoupl_nt_xent_loss, tau=args.tau, model_shape=input_shape)
    # -----------------------------------------------------------------------------------------------------------------

    pre_train_set = dataset_constructor_infonce(indices_types_dict["train_indices"][train_instance_indices], data_fetch_path,
                       'pre-train', args.batch_size, 'ampphase', args.observation_size, divisible_observation_size,
                       half_remainder, args.prediction_size, args.overlap, args.seed, curr_mask, order, order_dim, order_group_size)

    pre_val_set = dataset_constructor_infonce(indices_types_dict["train_indices"][val_instance_indices], data_fetch_path,
                       'pre-train-val', args.batch_size, 'ampphase', args.observation_size, divisible_observation_size,
                       half_remainder, args.prediction_size, args.overlap, args.seed, curr_mask, order, order_dim, order_group_size)

    pre_callback_objects = [
        WeightRestoreCallback(monitor='val_loss', min_delta=0, restore_best_weights=True)
    ]

    _ = pre_train_model.fit(x=pre_train_set, epochs=args.epoch_size, steps_per_epoch=(len(train_instance_indices) * step_multiplier) // args.batch_size,
                                           verbose=2, callbacks=pre_callback_objects, validation_data=pre_val_set, validation_freq=1,
                                           validation_steps=(len(val_instance_indices) * step_multiplier) // args.batch_size)

    # ------------------------------------------------- Fine-tuning ---------------------------------------------------
    ft_inp = tf.keras.layers.Input(shape=ft_shape)
    ft_cnn_extractor = MobileNetV2Backbone(input_shape=ft_shape, backbone_name=args.model_name + "_backbone_fine-tune").get_model()
    ft_enc = ft_cnn_extractor(ft_inp)

    bernoulli_probabilities = []
    for _ in range(class_nr):
        probability = LogisticRegression(initializer=tf.keras.initializers.HeUniform())(ft_enc)
        bernoulli_probabilities.append(probability)
    output_probabilities = tf.keras.layers.Concatenate(axis=-1)(bernoulli_probabilities)

    fine_tune_model = AnyLabelModel(ft_inp, output_probabilities, name=args.model_name + "_fine-tune")
    fine_tune_model.layers[1].set_weights(pre_train_model.layers[1].get_weights())
    for layer_new in fine_tune_model.layers[1].layers:
        layer_new.trainable = False

    fine_tune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
                            model_shape=ft_shape, feature_importance=False)
    # -----------------------------------------------------------------------------------------------------------------

    fine_train_set = dataset_constructor_infonce(indices_types_dict["train_indices"][train_instance_indices][fine_train_instance_indices], data_fetch_path,
                       'fine-tune', args.batch_size // 2, 'ampphase', args.observation_size, divisible_observation_size,
                       half_remainder, args.prediction_size, args.overlap, args.seed, curr_mask, order, order_dim, order_group_size)

    fine_val_set = dataset_constructor_infonce(indices_types_dict["train_indices"][val_instance_indices][fine_val_instance_indices], data_fetch_path,
                       'fine-tune-val', args.batch_size // 2, 'ampphase', args.observation_size, divisible_observation_size,
                       half_remainder, args.prediction_size, args.overlap, args.seed, curr_mask, order, order_dim, order_group_size)

    fine_callback_objects = [
        WeightRestoreCallback(monitor='val_loss', min_delta=0, restore_best_weights=True)
    ]

    _ = fine_tune_model.fit(x=fine_train_set, epochs=args.epoch_size, steps_per_epoch=(len(fine_train_instance_indices) * step_multiplier) // (args.batch_size // 2),
                                           verbose=2, callbacks=fine_callback_objects, validation_data=fine_val_set, validation_freq=1,
                                           validation_steps=(len(fine_val_instance_indices) * step_multiplier) // (args.batch_size // 2))

    test_set = dataset_constructor_infonce(test_instances, data_fetch_path,
                       'test', args.batch_size // 2, 'ampphase', args.observation_size, divisible_observation_size,
                       half_remainder, args.prediction_size, args.overlap, args.seed, curr_mask, order, order_dim, order_group_size)

    fine_tune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
                            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), MultiClassPrecision(num_classes=class_nr, average='weighted', threshold=0.5),
                                    MultiClassRecall(num_classes=class_nr, average='weighted', threshold=0.5), tfa.metrics.F1Score(num_classes=class_nr, average='weighted', threshold=0.5)],
                            model_shape=ft_shape, feature_importance=False)
    _, acc, precision, recall, f_score = fine_tune_model.evaluate(x=test_set, steps=(len(test_instances) * step_multiplier) // (args.batch_size // 2), verbose=2)
    masked_results.append(pd.DataFrame(data=[[acc, precision, recall, f_score]], columns=['A', 'P', 'R', 'F']))

if not os.path.exists(os.path.abspath('results/remove-retrain/')):
    os.mkdir(path=os.path.abspath('results/remove-retrain/'))

results_frame = pd.concat(masked_results, axis=0)
results_frame.to_csv("results/remove-retrain/{}_results_ps_{}_ol_{}_cvs_{}_{}.csv".format(args.model_name, str(args.prediction_size), str(int(args.overlap * 100.0)), str(args.crossval_split), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
