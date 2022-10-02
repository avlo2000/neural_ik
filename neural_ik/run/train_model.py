from pathlib import Path

import tensorflow as tf

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik.metrics import last_one_abs, first_one_abs
from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models import load


PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()

if __name__ == '__main__':
    batch_size = 1
    kin_model = 'kuka_robot'
    kin = load(kin_model, batch_size)

    blocks_count = 30
    model_dist, model_ik = residual_fk_dnn(kin_model, batch_size, blocks_count=blocks_count, corrector_units=32)
    model_dist.summary()

    opt = tf.keras.optimizers.RMSprop()
    loss = PowWeightedMSE(base=1.5)

    model_dist.compile(optimizer=opt, loss=loss, metrics=['mse', 'mae', last_one_abs, first_one_abs])

    with open(PATH_TO_DATA / 'kuka_test_10k.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    x = [thetas_seed, iso_transforms]
    y = tf.zeros(shape=(len(thetas), blocks_count), dtype=float)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_last_one_abs',
                                                      patience=3,
                                                      mode='min',
                                                      min_delta=0.01)
    tag = '_catch_NaNs'
    model_dist_path = PATH_TO_MODELS / f'{model_dist.name}_{kin_model}_{tag}.hdf5'
    model_ik_path = PATH_TO_MODELS / f'{model_ik.name}_{kin_model}_{tag}.hdf5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_dist_path, 'val_loss', verbose=1, save_best_only=False)

    history = model_dist.fit(x=[thetas_seed, iso_transforms], y=y,
                             epochs=50, batch_size=batch_size,
                             validation_split=0.1, callbacks=[early_stopping, checkpoint])

    model_dist.save(model_dist_path)
    model_ik.save(model_ik_path)

    plot_training_history(history.history, PATH_TO_PICS / f'{model_dist.name}_{kin_model}_{tag}.png')
