from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tqdm import tqdm

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.metrics import gamma_dx, gamma_dy, gamma_dz, angle_axis_l2
from tf_kinematics.kinematic_models_io import load

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '2k'
BATCH_SIZE = 32
metrics = [gamma_dx, gamma_dy, gamma_dz, angle_axis_l2]
tf.keras.backend.set_floatx('float64')
tf.config.set_visible_devices([], 'GPU')


def prepare_model() -> keras.Model:
    # model: keras.Model = NewtonRecurrentBoost(f'{KINEMATIC_NAME}_robot', BATCH_SIZE, 10)
    model: keras.Model = load_model(PATH_TO_MODELS / 'momentum_recurrent_grad_boost_bs32__kuka_10k___100ITERS_BIG_1_1')
    model.compile(loss='mse', metrics=metrics)
    return model


def prepare_data(kin_model, batch_size):
    kin = load(kin_model, batch_size)
    path_to_data = PATH_TO_DATA / f'{KINEMATIC_NAME}_test_{DATASET_SIZE_SUF}.csv'
    print(f"Loading {path_to_data}")
    with open(path_to_data, mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)

    n = len(thetas)
    n -= n % batch_size

    x = [thetas_seed[:n], iso_transforms[:n]]
    y = tf.zeros(shape=(n, 6), dtype=tf.float64)
    return x, y


def main():
    kin_model = f'{KINEMATIC_NAME}_robot'

    model = prepare_model()
    x, y_true = prepare_data(kin_model, BATCH_SIZE)

    y_pred = model.predict(x=x, batch_size=BATCH_SIZE, verbose=1)
    errors_data = {m.__name__: [] for m in metrics}
    for i in tqdm(range(len(y_true))):
        for m in metrics:
            errors_data[m.__name__].append(float(m(y_true[i], y_pred[i])))
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
