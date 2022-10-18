from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import CompactL2L2
from neural_ik.metrics import gamma_dx, gamma_dy, gamma_dz, angle_axis_l2
from neural_ik.models.momentum_recurrent_grad_boost import momentum_recurrent_grad_boost
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.tf_transformations import tf_compact

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()
LOGDIR = Path('../logs').absolute()
HISTS_DIR = Path('../hists').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '10k'

BATCH_SIZE = 1

tf.config.set_visible_devices([], 'GPU')
tf.debugging.disable_check_numerics()
print(tf.config.list_physical_devices())


def prepare_data(kin_model, batch_size):
    kin = load(kin_model, batch_size)
    with open(PATH_TO_DATA / f'{KINEMATIC_NAME}_train_{DATASET_SIZE_SUF}.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    return thetas, thetas_seed, iso_transforms


def main():
    kin_model = f"{KINEMATIC_NAME}_robot"
    thetas, thetas_seed, iso_transforms = prepare_data(kin_model, BATCH_SIZE)

    kin = load(kin_model, BATCH_SIZE)

    metrics = [gamma_dx, gamma_dy, gamma_dz, angle_axis_l2]
    errors_data = {m.__name__: [] for m in metrics}

    for theta, seed, iso in zip(thetas, thetas_seed, iso_transforms):
        gamma_seed = tf_compact(kin.forward(seed))
        gamma = tf_compact(kin.forward(theta))
        for m in metrics:
            errors_data[m.__name__].append(float(m(gamma_seed, gamma)))

    idx = 0
    for m_name, vals in errors_data.items():
        errors_data[m_name] = np.array(vals)
    for m_name, vals in errors_data.items():
        idx += 1
        axs = plt.subplot(len(errors_data), 1, idx)
        if idx == len(errors_data):
            axs.set_ylabel('Sample')
        axs.set_ylabel(m_name)
        axs.grid()

        plt.scatter(np.arange(0, len(vals)), vals, marker='.', linewidths=1)
        mean = np.repeat(vals.mean(axis=0), len(vals))
        mx_y = np.max(vals, axis=0)
        std = np.repeat(vals.std(axis=0), len(vals))
        plt.errorbar(np.arange(0, len(vals)), mean, yerr=std, fmt='go--', alpha=0.05)
        plt.text(0, mx_y, f'Mean: {mean[0]:.3f}')
        plt.text(0, mx_y - 0.02, f'StdDev: {std[0]:.3f}')
    plt.show()


if __name__ == '__main__':
    main()
