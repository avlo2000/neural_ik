from pathlib import Path

import tensorflow as tf

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import PowWeightedMSE
from neural_ik import metrics
from neural_ik.models.newton_linear_grad_boost import newton_linear_grad_boost
from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.models.newton_dnn_grad_boost import newton_dnn_grad_boost
from neural_ik.models.newton_iso44_grad_boost import newton_iso44_grad_boost
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models_io import load

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()
LOGDIR = Path('../logs').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '100k'


tf.debugging.disable_check_numerics()


def prepare_model(kin_model, batch_size):
    blocks_count = 32
    model = newton_dnn_grad_boost(kin_model, batch_size, blocks_count=blocks_count)
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.x, metrics.y, metrics.z, metrics.angle_axis_l2])
    model.summary()
    return model


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
    tag = '_0_2'
    epochs = 32

    model_path = PATH_TO_MODELS / f'{model.name}_{KINEMATIC_NAME}_{tag}.h5'
    model_checkpoint_path = PATH_TO_MODELS / f'{model.name}__{KINEMATIC_NAME}__{tag}__checkpoint.h5'
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
    kin_model = f"{KINEMATIC_NAME}_robot"
    batch_size = 32

    model = prepare_model(kin_model, batch_size)
    x, y, x_val, y_val = prepare_data(kin_model, batch_size, 0.3)
    train_model(model, x, y, x_val, y_val, batch_size)


if __name__ == '__main__':
    main()
