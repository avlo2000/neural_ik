from pathlib import Path

import keras

from data.data_io import read_csv
import tensorflow as tf

from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_one_abs, first_one_abs
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models_io import load
from keras.models import load_model

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()


def main():
    batch_size = 1
    kin_model = 'kuka_robot'
    kin = load(kin_model, batch_size)

    model_dist: keras.Model = load_model(PATH_TO_MODELS / 'residual_fk_dnn_dist_kuka_robot__catch_NaNs.hdf5')
    model_dist.compile(loss=PowWeightedMSE(2.7), metrics=[last_one_abs, first_one_abs])

    with open(PATH_TO_DATA / 'kuka_train_10k.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    x = [thetas_seed, iso_transforms]
    y = tf.zeros(shape=(len(thetas), model_dist.outputs[0].shape[-1]), dtype=tf.float32)

    tag = '__catch_NaNs'
    model_dist_path = PATH_TO_MODELS / f'{model_dist.name}_{kin_model}_{tag}_tuned.hdf5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dist_path, 'val_loss', verbose=1, save_best_only=False)

    history = model_dist.fit(x=x, y=y,
                             epochs=100, batch_size=batch_size,
                             validation_split=0.1, callbacks=[checkpoint])

    model_dist.save(model_dist_path)

    plot_training_history(history.history, PATH_TO_PICS / f'{model_dist.name}_{kin_model}_{tag}.png')


if __name__ == '__main__':
    main()
