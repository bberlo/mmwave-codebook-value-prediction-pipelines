from experiment_environment.custom_items.metrics import MultiClassPrecision, MultiClassRecall
from experiment_environment.architectures.infoNCE_tuning import MobileNetV2Backbone
from experiment_environment.custom_items.anyLabelFinetuneModel import AnyLabelModel
from experiment_environment.custom_items.data_fetching import fetch_labels_indices
from experiment_environment.custom_items.logRegression import LogisticRegression
from experiment_environment.custom_items.utilities import find_nearest_divisible
from experiment_environment.custom_items.tuners import HyperbandSizeFiltering
from kerastuner_tensorboard_logger import TensorBoardLogger, setup_tb
from multiprocess import set_start_method
import tensorflow_addons as tfa
import tensorflow as tf
import argparse
import pickle
import uuid
import os


# ------------------------------------ STD TRAINING WITH REAL AMPPHASE DATA -------------------------------------


def build_model(hp):
    uuid_string = str(uuid.uuid4())

    std_extractor = MobileNetV2Backbone(hp, input_shape=(256, 256, 4), backbone_name='backbone_tuning_{}'.format(uuid_string)).get_model()
    inp = tf.keras.layers.Input(shape=(256, 256, 4))
    enc_o = std_extractor(inp)

    bernoulli_probabilities = []
    for _ in range(8):
        probability = LogisticRegression(initializer=tf.keras.initializers.HeUniform())(enc_o)
        bernoulli_probabilities.append(probability)
    output_probabilities = tf.keras.layers.Concatenate(axis=-1)(bernoulli_probabilities)

    complete_model = AnyLabelModel(inp, output_probabilities, name='tuning_{}'.format(uuid_string))

    complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
                            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                                     MultiClassPrecision(num_classes=8, average='weighted', threshold=0.5),
                                     MultiClassRecall(num_classes=8, average='weighted', threshold=0.5),
                                     tfa.metrics.F1Score(num_classes=8, average='weighted', threshold=0.5)],
                            model_shape=(256, 256, 4), feature_importance=False)

    return complete_model


# ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment automation setup script.")
    parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
    parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
    parser.add_argument('-o_s', '--observation_size', type=int, help='<Required> Size observation portion overlapping windows', required=True)
    parser.add_argument('-p_s', '--prediction_size', type=int, help='<Required> Size prediction portion overlapping windows', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment', required=True)
    parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=True)
    parser.add_argument('-s', '--seed', type=int, help='<Required> Random seed used for random ops in the experiment', required=True)
    parser.add_argument('-l_r', '--learning_rate', type=float, help='<Required> Learning rate to be used in the experiment', required=True)
    parser.add_argument('-o_l', '--overlap', type=float, help='<Required> Percentage 0-1 subsequent windows should overlap after dataset sample stitching', required=True)
    parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
    parser.add_argument('-f_p', '--file_path', help='<Required> File path to be used in the experiment', required=True)
    args = parser.parse_args()

    # Prevent main process from clogging up GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as e:
            print(e)

    # Load training, validation, and test data
    with open(args.file_path, 'rb') as handle:
        indices_types_dict = pickle.load(handle)

    divisible_observation_size = find_nearest_divisible(args.observation_size, 5)
    half_remainder = (divisible_observation_size - args.observation_size) // 2
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    frame_size = 2000
    step_multiplier = frame_size // (args.observation_size + args.prediction_size)

    train_instance_indices, val_instance_indices = fetch_labels_indices(f_path=data_fetch_path, indices=indices_types_dict["train_indices"], seed=args.seed)
    test_instances = indices_types_dict["test_indices"]
    os.remove(args.file_path)

    set_start_method("spawn")

    tuner = HyperbandSizeFiltering(
            hypermodel=build_model,
            objective='val_loss',
            max_epochs=50,
            factor=3,
            hyperband_iterations=1,
            seed=42,
            directory=r'Streams',
            project_name=args.model_name,
            logger=TensorBoardLogger(metrics=["val_loss"], logdir='Streams/{}-hparams'.format(args.model_name))
    )
    setup_tb(tuner)
    tuner.search(x=indices_types_dict["train_indices"][train_instance_indices], epochs=args.epoch_size, steps_per_epoch=(len(indices_types_dict["train_indices"][train_instance_indices]) * step_multiplier) // args.batch_size, verbose=2, gpu=args.gpu,
                 validation_data=indices_types_dict["train_indices"][val_instance_indices], validation_steps=(len(indices_types_dict["train_indices"][val_instance_indices]) * step_multiplier) // args.batch_size,
                 validation_freq=1, batch_size=args.batch_size, dataset_filepath=args.data_file_path, o_size=args.observation_size, o_size_nd=divisible_observation_size, half_remainder=half_remainder, p_size=args.prediction_size, overlap=args.overlap,
                 seed=args.seed)
