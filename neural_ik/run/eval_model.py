from pathlib import Path

import keras
import tensorflow as tf
from keras.models import load_model

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.metrics import gamma_xyz_max, gamma_andle_axis_max
from neural_ik.models.newton_dnn_grad_boost import newton_dnn_grad_boost
from tf_kinematics.kinematic_models_io import load

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '10k'


def prepare_model() -> keras.Model:
    model: keras.Model = load_model(PATH_TO_MODELS / 'newton_dnn_grad_boost_kuka__0_2.h5')
    model.compile(loss='mse', metrics=[gamma_xyz_max, gamma_andle_axis_max])
    model.summary()
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
    y = tf.zeros(shape=(n, 6), dtype=float)
    return x, y


def main():
    batch_size = 32
    kin_model = f'{KINEMATIC_NAME}_robot'

    model = prepare_model()
    x, y = prepare_data(kin_model, batch_size)

    eval_res = model.evaluate(x=x, y=y, batch_size=batch_size)
    print(eval_res)


if __name__ == '__main__':
    main()
