from pathlib import Path

import keras

from data.data_io import read_csv
import tensorflow as tf

from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_gamma_diff, first_gamma_diff
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models_io import load
from keras.models import load_model

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '10k'


def prepare_data(kin_model, batch_size, validation_split):
    kin = load(kin_model, batch_size)
    with open(PATH_TO_DATA / f'{KINEMATIC_NAME}_train_{DATASET_SIZE_SUF}.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    n = len(thetas)

    val_size = int(n * validation_split)
    val_size -= val_size % batch_size
    train_size = n - val_size
    train_size -= train_size % batch_size

    x = (thetas_seed[:train_size], iso_transforms[:train_size])
    y = tf.zeros(shape=(train_size, 6), dtype=float)
    x_val = (thetas_seed[train_size: train_size + val_size], iso_transforms[train_size: train_size + val_size])
    y_val = tf.zeros(shape=(val_size, 6), dtype=float)

    return x, y, x_val, y_val


def train_model(model, x, y, x_val, y_val, batch_size):
    tag = '_0_3'
    epochs = 100

    model_path = PATH_TO_MODELS / f'{model.name}_{KINEMATIC_NAME}_{tag}.h5'
    model_checkpoint_path = PATH_TO_MODELS / f'{model.name}_{DATASET_SIZE_SUF}__{KINEMATIC_NAME}__{tag}__checkpoint.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, 'val_loss', verbose=1, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, mode='min', restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1, update_freq='batch', log_dir=LOGDIR)

    history = model.fit(x=x, y=y, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        validation_batch_size=batch_size,
                        callbacks=[early_stopping, checkpoint, tensorboard_callback],
                        workers=12, use_multiprocessing=True)

    model.save(model_path)
    plot_training_history(history.history, PATH_TO_PICS / f'{model.name}_{KINEMATIC_NAME}_{tag}.png')

def main():
    batch_size = 1
    kin_model = 'kuka_robot'
    kin = load(kin_model, batch_size)

    model_dist: keras.Model = load_model(PATH_TO_MODELS / 'residual_fk_dnn_dist_kuka_robot__catch_NaNs.hdf5')
    model_dist.compile(loss=PowWeightedMSE(2.7), metrics=[last_gamma_diff, first_gamma_diff])

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
