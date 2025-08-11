from experiment_environment.custom_items.data_fetching import fetch_labels_indices, dataset_constructor_orthmatchpursuit
from experiment_environment.custom_items.metrics import MultiClassPrecision, MultiClassRecall
from experiment_environment.architectures.orthMatchPursuit import OrthMatchPursuit
from experiment_environment.custom_items.callbacks import WeightRestoreCallback
from experiment_environment.custom_items.anyLabelModel import AnyLabelModel
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import argparse
import datetime
import pickle
import os

parser = argparse.ArgumentParser(description="Experiment automation setup script.")
parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
parser.add_argument('-o_s', '--observation_size', type=int, help='<Required> Size observation portion overlapping windows', required=True)
parser.add_argument('-p_s', '--prediction_sizes', nargs='+', type=int, help='<Required> Sizes prediction portion overlapping windows', required=True)
parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment', required=True)
parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=True)
parser.add_argument('-s', '--seed', type=int, help='<Required> Random seed used for random ops in the experiment', required=True)
parser.add_argument('-l_r', '--learning_rate', type=float, help='<Required> Learning rate to be used in the experiment', required=True)
parser.add_argument('-o_l', '--overlap', type=float, help='<Required> Percentage 0-1 subsequent windows should overlap after dataset sample stitching', required=True)
parser.add_argument('-do_t', '--domain_type', help='<Required> Domain type: user, position, orientation', required=False)
parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
parser.add_argument('-d_f_n', '--domain_factor_name', help='<Required> Set domain factor name to be used in the experiment', required=True)
parser.add_argument('-f_p', '--file_path', help='<Required> File path to be used in the experiment', required=True)
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

data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
n_rx = 4
n_adc = 256
input_shape = (args.observation_size, n_adc, n_rx)
frame_size = 2000
class_nr = 8

train_instance_indices, val_instance_indices = fetch_labels_indices(f_path=data_fetch_path, indices=indices_types_dict["train_indices"], seed=args.seed)
test_instances = indices_types_dict["test_indices"]
os.remove(args.file_path)

frame_ablation_results = []
for prediction_size in args.prediction_sizes:

    step_multiplier = frame_size // (args.observation_size + prediction_size)

    # -------------------------------------------- Training -----------------------------------------------------------
    inp = tf.keras.layers.Input(shape=input_shape)
    mpursuit_model = OrthMatchPursuit(n_rx=n_rx, n_adc=n_adc, rmin=0., rmax=5., theta_min=-70., theta_max=70.,
                                      rsamp=100, theta_samp=100, slow_time_seq_len=args.observation_size,
                                      backbone_name=args.model_name + "_backbone", num_classes=class_nr,
                                      b_size=args.batch_size).get_model()
    outp = mpursuit_model(inp)

    train_model = AnyLabelModel(inp, outp, name=args.model_name)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
                        model_shape=input_shape, feature_importance=False)
    # -----------------------------------------------------------------------------------------------------------------

    train_set = dataset_constructor_orthmatchpursuit(indices_types_dict["train_indices"][train_instance_indices], data_fetch_path,
                       'train', args.batch_size, 'ampphase', args.observation_size, prediction_size, args.overlap,
                       args.seed, None, None, None, None)

    val_set = dataset_constructor_orthmatchpursuit(indices_types_dict["train_indices"][val_instance_indices], data_fetch_path,
                       'val', args.batch_size, 'ampphase', args.observation_size, prediction_size, args.overlap,
                       args.seed, None, None, None, None)

    callback_objects = [
        WeightRestoreCallback(monitor='val_loss', min_delta=0, restore_best_weights=True)
    ]

    _ = train_model.fit(x=train_set, epochs=args.epoch_size, steps_per_epoch=(len(train_instance_indices) * step_multiplier) // args.batch_size,
                                           verbose=2, callbacks=callback_objects, validation_data=val_set, validation_freq=1,
                                           validation_steps=(len(val_instance_indices) * step_multiplier) // args.batch_size)

    test_set = dataset_constructor_orthmatchpursuit(test_instances, data_fetch_path,
                       'test', args.batch_size, 'ampphase', args.observation_size, prediction_size, args.overlap,
                       args.seed, None, None, None, None)

    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
                            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), MultiClassPrecision(num_classes=class_nr, average='weighted', threshold=0.5),
                                    MultiClassRecall(num_classes=class_nr, average='weighted', threshold=0.5), tfa.metrics.F1Score(num_classes=class_nr, average='weighted', threshold=0.5)],
                            model_shape=input_shape, feature_importance=False)
    _, acc, precision, recall, f_score = train_model.evaluate(x=test_set, steps=(len(test_instances) * step_multiplier) // args.batch_size, verbose=2)
    frame_ablation_results.append(pd.DataFrame(data=[[acc, precision, recall, f_score]], columns=['A', 'P', 'R', 'F']))

if not os.path.exists(os.path.abspath('results/frame-ablation/')):
    os.mkdir(path=os.path.abspath('results/frame-ablation/'))

results_frame = pd.concat(frame_ablation_results, axis=0)
results_frame.to_csv("results/frame-ablation/{}_results_ps_{}_ol_{}_dfn_{}_cvs_{}_{}.csv".format(args.model_name, "_".join([str(x) for x in args.prediction_sizes]), str(int(args.overlap * 100.0)), args.domain_factor_name, str(args.crossval_split), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
