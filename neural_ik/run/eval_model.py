from pathlib import Path

import keras

from data.data_io import read_csv
import tensorflow as tf

from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_gamma_diff, first_gamma_diff, gamma_xyz_max, gamma_andle_axis_max
from tf_kinematics.kinematic_models_io import load, omnipointer_robot
from keras.models import load_model

from tf_kinematics.layers.iso_layers import IsometryCompact
from tf_kinematics.layers.solve_layers import SolveIterGrad

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'omnipointer'
DATASET_SIZE_SUF = '2k'


def prepare_model() -> keras.Model:
    model: keras.Model = load_model(PATH_TO_MODELS / 'residual_solver_dnn_dist_omnipointer__0.1.hdf5',
                                    custom_objects={
                                        'IsometryCompact': IsometryCompact,
                                        'SolveIterGrad': SolveIterGrad,
                                        'omnipointer_robot': omnipointer_robot
                                    })
    model.compile(loss='mse', metrics=[gamma_xyz_max, gamma_andle_axis_max])
    model.summary()
    return model


def prepare_data(kin_model, batch_size):
    kin = load(kin_model, batch_size)
    with open(PATH_TO_DATA / f'{KINEMATIC_NAME}_test_{DATASET_SIZE_SUF}.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    x = [thetas_seed, iso_transforms]
    y = tf.zeros(shape=(len(thetas), 6), dtype=float)
    return x, y


def main():
    batch_size = 1
    kin_model = 'kuka_robot'

    model = prepare_model()
    x, y = prepare_data(kin_model, batch_size)

    eval_res = model.evaluate(x=x, y=y, batch_size=batch_size)
    print(eval_res)


if __name__ == '__main__':
    main()
