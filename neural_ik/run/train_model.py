from pathlib import Path

import tensorflow as tf

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_gamma_diff, first_gamma_diff, gamma_xyz_norm, gamma_andle_axis_norm, max_diff_abs, \
    gamma_xyz_max, gamma_andle_axis_max
from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.models.residual_solver_dnn import residual_solver_dnn
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models_io import load

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()
KINEMATIC_NAME = 'omnipointer'
DATASET_SIZE_SUF = '2k'


tf.debugging.disable_check_numerics()


def prepare_model(kin_model, batch_size):
    blocks_count = 16
    model = residual_solver_dnn(kin_model, batch_size, blocks_count=blocks_count)
    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='mse', metrics=[gamma_xyz_max, gamma_andle_axis_max])
    model.summary()
    return model


def prepare_data(kin_model, batch_size):
    kin = load(kin_model, batch_size)
    with open(PATH_TO_DATA / f'{KINEMATIC_NAME}_train_{DATASET_SIZE_SUF}.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    x = [thetas_seed, iso_transforms]
    y = tf.zeros(shape=(len(thetas), 6), dtype=float)
    return x, y


def train_model(model, x, y, batch_size):
    tag = '_0.1'
    epochs = 50

    model_path = PATH_TO_MODELS / f'{model.name}_{KINEMATIC_NAME}_{tag}.hdf5'
    model_checkpoint_path = PATH_TO_MODELS / f'{model.name}__{KINEMATIC_NAME}__{tag}__checkpoint.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, 'val_loss', verbose=1, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, mode='min',  min_delta=0.001,
                                                      restore_best_weights=True)

    history = model.fit(x=x, y=y,
                        epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, callbacks=[early_stopping, checkpoint],
                        workers=6, use_multiprocessing=True)

    model.save(model_path)
    plot_training_history(history.history, PATH_TO_PICS / f'{model.name}_{KINEMATIC_NAME}_{tag}.png')


def main():
    kin_model = f"{KINEMATIC_NAME}_robot"
    batch_size = 1

    model = prepare_model(kin_model, batch_size)
    x, y = prepare_data(kin_model, batch_size)
    train_model(model, x, y, batch_size)


if __name__ == '__main__':
    main()
