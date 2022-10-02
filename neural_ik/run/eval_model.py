from pathlib import Path

import keras

from data.data_io import read_csv
import tensorflow as tf

from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_one_abs, max_diff_abs, first_one_abs
from tf_kinematics.iso_layers import IsometryCompact, IsometryInverse, CompactL2Norm
from tf_kinematics.kin_layers import ForwardKinematics
from tf_kinematics.kinematic_models import load
from keras.models import load_model

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()


def main():
    batch_size = 1
    kin_model = 'kuka_robot'
    kin = load(kin_model, batch_size)

    model_dist: keras.Model = load_model(PATH_TO_MODELS / 'residual_acos_fk_dnn_dist_kuka_robot_0.1.hdf5')
    model_dist.compile(loss=PowWeightedMSE(), metrics=['mse', 'mae', last_one_abs, first_one_abs])

    with open(PATH_TO_DATA / 'kuka_test_10k.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    y = tf.zeros(shape=(len(thetas), model_dist.output_shape[1]), dtype=float)

    hist = model_dist.evaluate(x=[thetas_seed, iso_transforms], y=y, batch_size=batch_size)
    print(hist)

    # model_ik: keras.Model = tf.keras.models.load_model(PATH_TO_MODELS / 'residual_fk_dnn_ik_kuka_robot_0.1.hdf5')
    # model_ik.outputs = model_ik.outputs[-1]
    # model_ik.compile(loss='mse', metrics=['mse', 'mae', max_diff_abs])
    # hist = model_ik.evaluate(x=[thetas_seed, iso_transforms], y=thetas, batch_size=batch_size)
    # print(hist)


if __name__ == '__main__':
    main()
