from experiment_environment.custom_items.utilities import domain_to_be_left_out_indices_calculation
from experiment_environment.custom_items.data_fetching import fetch_labels_indices
from sklearn.model_selection import StratifiedKFold
from simple_slurm import Slurm
import numpy as np
import itertools
import argparse
import pickle
import uuid

# Command prompt settings for experiment automation
parser = argparse.ArgumentParser(description='Experiment automation setup script.')
parser.add_argument('-m_n', '--model_name', help='<Required> Model name used in the experiment: infonce, minirocket, ompursuit', required=True)
parser.add_argument('-t', '--type', help='<Required> Experiment type: leave-out, shapley, remove-retrain, frame-ablation', required=True)
parser.add_argument('-do_t', '--domain_type', help='<Required> Domain type: random, orientation, position, radar', required=False)
parser.add_argument('-r_p', '--result_path', help='<Required> File path of Shapley importance results to be used in the experiment', required=False)
parser.add_argument('-g', '--gpu', type=int, help='When tuning, set GPU since only one cross validation fold is used', required=False)
parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=False)
parser.add_argument('-s', '--seed', type=int, help='Pseudo-random seed to be used in the experiment', required=True)
args = parser.parse_args()

# WIN HPC slurm cluster configuration
cluster_config_obj = Slurm(
    '--job_name', args.model_name,
    '--nodes', '1',
    '--ntasks', '3',
    '--partition', 'mcs.gpu.q',
    '--error', 'slurm-%j.err',
    '--output', 'slurm-%j.out',
    '--time', '1-0',
    '--constraint', '2080ti',
    '--gres', 'gpu:1'
)

# -----------------------------
# Varying leave-out experiment settings
# -----------------------------
if args.model_name == 'infonce' and args.type == 'leave-out' and args.domain_type == 'random':
    experiment_type = 'random'
    script_name = 'experiment_leave_out_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'leave-out' and args.domain_type == 'random':
    experiment_type = 'random'
    script_name = 'experiment_leave_out_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'leave-out' and args.domain_type == 'random':
    experiment_type = 'random'
    script_name = 'experiment_leave_out_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'infonce' and args.type == 'leave-out' and args.domain_type == 'orientation':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 0
    CUR_DOMAIN_FACTOR_TOTAL_NR = 8
    prediction_size = 40
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'leave-out' and args.domain_type == 'orientation':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 0
    CUR_DOMAIN_FACTOR_TOTAL_NR = 8
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'leave-out' and args.domain_type == 'orientation':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 0
    CUR_DOMAIN_FACTOR_TOTAL_NR = 8
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'infonce' and args.type == 'leave-out' and args.domain_type == 'position':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 1
    CUR_DOMAIN_FACTOR_TOTAL_NR = 15
    prediction_size = 40
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'leave-out' and args.domain_type == 'position':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 1
    CUR_DOMAIN_FACTOR_TOTAL_NR = 15
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'leave-out' and args.domain_type == 'position':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 1
    CUR_DOMAIN_FACTOR_TOTAL_NR = 15
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'infonce' and args.type == 'leave-out' and args.domain_type == 'radar':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 2
    CUR_DOMAIN_FACTOR_TOTAL_NR = 2
    prediction_size = 40
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'leave-out' and args.domain_type == 'radar':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 2
    CUR_DOMAIN_FACTOR_TOTAL_NR = 2
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'leave-out' and args.domain_type == 'radar':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_leave_out_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 2
    CUR_DOMAIN_FACTOR_TOTAL_NR = 2
    prediction_size = 40
    tau = None
    domain_type = args.domain_type
    result_path = None

# ----------------------------------
# Varying frame-ablation experiment settings
# ----------------------------------
elif args.model_name == 'infonce' and args.type == 'frame-ablation' and args.domain_type == 'random':
    experiment_type = 'random'
    script_name = 'experiment_frame_ablation_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'frame-ablation' and args.domain_type == 'random':
    experiment_type = 'random'
    script_name = 'experiment_frame_ablation_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'frame-ablation' and args.domain_type == 'random':
    experiment_type = 'random'
    script_name = 'experiment_frame_ablation_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'infonce' and args.type == 'frame-ablation' and args.domain_type == 'orientation':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 0
    CUR_DOMAIN_FACTOR_TOTAL_NR = 8
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'frame-ablation' and args.domain_type == 'orientation':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 0
    CUR_DOMAIN_FACTOR_TOTAL_NR = 8
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'frame-ablation' and args.domain_type == 'orientation':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 0
    CUR_DOMAIN_FACTOR_TOTAL_NR = 8
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'infonce' and args.type == 'frame-ablation' and args.domain_type == 'position':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 1
    CUR_DOMAIN_FACTOR_TOTAL_NR = 15
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'frame-ablation' and args.domain_type == 'position':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 1
    CUR_DOMAIN_FACTOR_TOTAL_NR = 15
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'frame-ablation' and args.domain_type == 'position':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 1
    CUR_DOMAIN_FACTOR_TOTAL_NR = 15
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'infonce' and args.type == 'frame-ablation' and args.domain_type == 'radar':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 2
    CUR_DOMAIN_FACTOR_TOTAL_NR = 2
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = 0.1
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'frame-ablation' and args.domain_type == 'radar':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 2
    CUR_DOMAIN_FACTOR_TOTAL_NR = 2
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'frame-ablation' and args.domain_type == 'radar':
    experiment_type = 'domain-leave-out'
    script_name = 'experiment_frame_ablation_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    domain_label_struct = (8, 15, 2, 1)
    CUR_DOMAIN_FACTOR = 2
    CUR_DOMAIN_FACTOR_TOTAL_NR = 2
    prediction_size = [10, 20, 40, 80, 120, 160, 200, 240, 320, 400]
    tau = None
    domain_type = args.domain_type
    result_path = None

# --------------------------------------
# Varying Shapley importance experiment settings
# --------------------------------------
elif args.model_name == 'infonce' and args.type == 'shapley':
    experiment_type = 'random'
    script_name = 'experiment_shapley_importance_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = None
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = 0.1
    domain_type = None
    result_path = None

elif args.model_name == 'minirocket' and args.type == 'shapley':
    experiment_type = 'random'
    script_name = 'experiment_shapley_importance_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = None
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = None
    domain_type = None
    result_path = None

elif args.model_name == 'ompursuit' and args.type == 'shapley':
    experiment_type = 'random'
    script_name = 'experiment_shapley_importance_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = None
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = None
    domain_type = None
    result_path = None

# --------------------------------------
# Varying remove-retrain experiment settings
# --------------------------------------
elif args.model_name == 'infonce' and args.type == 'remove-retrain':
    experiment_type = 'random'
    script_name = 'experiment_remove_retrain_infonce.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = None
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = 0.1
    domain_type = None
    result_path = args.result_path

elif args.model_name == 'minirocket' and args.type == 'remove-retrain':
    experiment_type = 'random'
    script_name = 'experiment_remove_retrain_minirocket.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = None
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = None
    domain_type = None
    result_path = args.result_path

elif args.model_name == 'ompursuit' and args.type == 'remove-retrain':
    experiment_type = 'random'
    script_name = 'experiment_remove_retrain_orthmatpur.py'
    data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
    CUR_DOMAIN_FACTOR_NAME = None
    domain_label_struct = None
    CUR_DOMAIN_FACTOR = None
    CUR_DOMAIN_FACTOR_TOTAL_NR = None
    prediction_size = 40
    tau = None
    domain_type = None
    result_path = args.result_path

else:
    raise ValueError("Correct combination of model_name, type, and domain_type was not set. Use -h flag to get access"
                     "to possible values. Please note that shapley and remove-retrain do not require domain_type to"
                     "be set.")

# Create train/test subset split generators with len equal to desired folds or number of domain factors of speficic type
# Splitting based on standard folds of random domains not considered due to potential for class label imbalances
if experiment_type == 'random':
    class_labels = fetch_labels_indices(f_path=data_fetch_path, experiment_type=experiment_type)
    class_labels = np.argmax(class_labels, axis=1) + 1

    splits_size = 5
    kfold_obj = StratifiedKFold(n_splits=splits_size, shuffle=True, random_state=args.seed)
    split_generator = kfold_obj.split(X=np.zeros_like(class_labels), y=class_labels)

else:
    domain_labels = fetch_labels_indices(f_path=data_fetch_path, experiment_type=experiment_type)
    domain_labels = np.argmax(domain_labels, axis=1) + 1

    test_types_list = [list(map(lambda y: y + 1, domain_to_be_left_out_indices_calculation(CUR_DOMAIN_FACTOR, x,
    domain_label_struct))) for x in range(CUR_DOMAIN_FACTOR_TOTAL_NR)]
    splits_size = len(test_types_list)
    train_types_list = [list(set(range(1, np.prod(domain_label_struct).item() + 1, 1)) - set(x)) for x in test_types_list]

    def indices_generator(test_list, train_list):

        for test_types, train_types in zip(test_list, train_list):

            yield np.where(np.isin(domain_labels, test_elements=np.asarray(train_types)))[0], \
                  np.where(np.isin(domain_labels, test_elements=np.asarray(test_types)))[0]

    split_generator = indices_generator(test_types_list, train_types_list)

# No splitting involved because slurm sbatch calls are individually scheduled
# Not run in a process immediately
split_nrs = range(splits_size)
GPU_DEVICE = args.gpu

# Reduce split_generator to specific split
if args.crossval_split:

    split_nrs_index = split_nrs.index(args.crossval_split)
    split_nrs = [args.crossval_split]
    split_generator = itertools.islice(split_generator, split_nrs_index, split_nrs_index + 1, 1)

argument_list = [
    'python', script_name,
    '-e_s', '10' if args.model_name == 'ompursuit' else '100',
    '-b_s', '32' if args.model_name == 'infonce' else '16',
    '-o_s', '250',
    '-p_s', str(prediction_size) if type(prediction_size) == int else " ".join([str(x) for x in prediction_size]),
    '-g', str(GPU_DEVICE),
    '-s', str(args.seed),
    '-l_r', '0.0001',
    '-t', str(tau),
    '-o_l', '0.3',
    '-do_t', domain_type,
    '-m_n', args.model_name,
    '-d_f_n', domain_type,
    '-r_p', result_path
]

none_indices = {i for i, val in enumerate(argument_list) if val is None}
argument_list = [val for i, val in enumerate(argument_list) if i not in none_indices and i + 1 not in none_indices]

for split_nr, (train_indices, test_indices) in zip(split_nrs, split_generator):

    file_path = 'tmp/' + uuid.uuid4().hex + '.pickle'
    with open(file_path, 'wb') as handle:
        pickle.dump(obj={'train_indices': train_indices, 'test_indices': test_indices},
                    file=handle, protocol=pickle.HIGHEST_PROTOCOL)

    argument_list_extended = argument_list.copy()
    argument_list_extended.extend([
        '-cv_s', str(split_nr),
        '-f_p', file_path,
    ])

    if not CUR_DOMAIN_FACTOR_NAME:
        CUR_DOMAIN_FACTOR_NAME = 'none'

    cluster_config_obj.sbatch(
        run_cmd=' '.join(argument_list_extended),
        shell='/bin/bash'
    )
