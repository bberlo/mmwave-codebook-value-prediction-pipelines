from experiment_environment.custom_items.data_fetching import fetch_labels_indices
from sklearn.model_selection import StratifiedKFold
from subprocess import run
import numpy as np
import itertools
import argparse
import pickle
import uuid

# Command prompt settings for experiment automation
parser = argparse.ArgumentParser(description='Experiment automation setup script.')
parser.add_argument('-g', '--gpu', type=int, help='When tuning, set GPU since only one cross validation fold is used', required=False)
parser.add_argument('-s', '--seed', type=int, help='Pseudo-random seed to be used in the experiment', required=True)
args = parser.parse_args()

# -----------------------------
# Tuning experiment settings
# -----------------------------
experiment_type = 'random'
script_name = 'experiment_tuning_infonce.py'
data_fetch_path = r'Datasets/codebook-value-prediction-domain-leave-out-benchmark-complex.hdf5'
CUR_DOMAIN_FACTOR_NAME = None
domain_label_struct = None
CUR_DOMAIN_FACTOR = None
CUR_DOMAIN_FACTOR_TOTAL_NR = None
prediction_size = 40
tau = None
domain_type = None
result_path = None

# Create train/test subset split generators with len equal to desired folds or number of domain factors of speficic type
# Splitting based on standard folds of random domains not considered due to potential for class label imbalances
class_labels = fetch_labels_indices(f_path=data_fetch_path, experiment_type=experiment_type)
class_labels = np.argmax(class_labels, axis=1) + 1

splits_size = 5
kfold_obj = StratifiedKFold(n_splits=splits_size, shuffle=True, random_state=args.seed)
split_generator = kfold_obj.split(X=np.zeros_like(class_labels), y=class_labels)

split_nrs = [0]
split_generator = itertools.islice(split_generator, 0, 1, 1)

argument_list = [
    'python', script_name,
    '-e_s', '100',
    '-b_s', '32',
    '-o_s', '250',
    '-p_s', str(prediction_size),
    '-g', str(args.gpu),
    '-s', str(args.seed),
    '-l_r', '0.0001',
    '-t', str(tau),
    '-o_l', '0.3',
    '-do_t', domain_type,
    '-m_n', 'infonce',
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

    with open('Streams/{}_{}_{}_{}_stdout.txt'.format('infonce', 'tuning', CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stdoutFile, \
             open('Streams/{}_{}_{}_{}_stderr.txt'.format('infonce', 'tuning', CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stderrFile:

        run(argument_list_extended, stdout=stdoutFile, stderr=stderrFile)
