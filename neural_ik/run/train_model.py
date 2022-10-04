from pathlib import Path

import tensorflow as tf

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_gamma_diff, first_gamma_diff, gamma_xyz_norm, gamma_andle_axis_norm
from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.models.residual_newton_iter_percept import residual_newton_iter_percept
from neural_ik.models.residual_solver_dnn import residual_solver_dnn
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models_io import load


PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()

if __name__ == '__main__':
    batch_size = 1
    kin_model = 'kuka_robot'
    kin = load(kin_model, batch_size)

    blocks_count = 32
    model_dist, model_ik = residual_solver_dnn(kin_model, batch_size, blocks_count=blocks_count)
    opt = tf.keras.optimizers.RMSprop()
    model_dist.compile(optimizer=opt, loss='mse')
    model_dist.summary()

    with open(PATH_TO_DATA / 'kuka_train_10k.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    x = [thetas_seed, iso_transforms]
    y = tf.zeros(shape=(len(thetas), 6), dtype=float)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_last_one_abs',
                                                      patience=5,
                                                      mode='min',
                                                      min_delta=0.001)
    tag = '_0.1'
    model_dist_path = PATH_TO_MODELS / f'{model_dist.name}_{kin_model}_{tag}.hdf5'
    model_ik_path = PATH_TO_MODELS / f'{model_ik.name}_{kin_model}_{tag}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dist_path, 'val_loss', verbose=1, save_best_only=False)

    history = model_dist.fit(x=[thetas_seed, iso_transforms], y=y,
                             epochs=50, batch_size=batch_size,
                             validation_split=0.1, callbacks=[early_stopping, checkpoint])

    model_dist.save(model_dist_path)
    model_ik.save(model_ik_path)

    plot_training_history(history.history, PATH_TO_PICS / f'{model_dist.name}_{kin_model}_{tag}.png')
