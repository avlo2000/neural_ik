from pathlib import Path

import json
import tensorflow as tf

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset

from inference.gd_solver import GDModel
from inference.momentum_solver import MomentumModel
from neural_ik.losses import CompactXYZL2CosAA, CompactL2L2, CompactL4L4
from neural_ik import metrics
from neural_ik.models.adam_recurrent_grad_boost import adam_recurrent_grad_boost
from neural_ik.models.momentum_recurrent_grad_boost import momentum_recurrent_grad_boost
from neural_ik.models.gd_recurrent_grad_boost import gd_recurrent_grad_boost
from neural_ik.models.qp_recurrent_grad_boost import qp_recurrent_grad_boost
from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.models.newton_dnn_grad_boost import newton_dnn_grad_boost

from neural_ik.models.simpe_dnn import simple_dnn
from neural_ik.visual import plot_training_history
from tf_kinematics.kinematic_models_io import load

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
PATH_TO_PICS = Path('../pics').absolute()
LOGDIR = Path('../logs').absolute()
HISTS_DIR = Path('../hists').absolute()
KINEMATIC_NAME = 'dof2d'
DATASET_SIZE_SUF = '10k'

N_ITERS = 15
BATCH_SIZE = 1

tf.config.set_visible_devices([], 'GPU')
tf.debugging.disable_check_numerics()
tf.keras.backend.set_floatx('float64')
print(tf.config.list_physical_devices())


def prepare_model(kin_model, batch_size):
    model = momentum_recurrent_grad_boost(kin_model, batch_size, N_ITERS)
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.gamma_dx, metrics.gamma_dy])
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


def prepare_test_data(kin_model, batch_size):
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


def train_model(model, x, y, x_val, y_val, batch_size):
    tag = '_experiment'
    epochs = 5

    model_full_name = f'{model.name}_bs{BATCH_SIZE}__{KINEMATIC_NAME}_{DATASET_SIZE_SUF}__{tag}'
    model_path = PATH_TO_MODELS / model_full_name
    model_checkpoint_path = PATH_TO_MODELS / f'{model_full_name}__checkpoint'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, 'val_loss',
                                                    verbose=1, save_best_only=True, save_weights_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=7, mode='min', restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1, update_freq='batch', log_dir=LOGDIR)

    history = model.fit(x=x, y=y, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        validation_batch_size=batch_size,
                        callbacks=[early_stopping, checkpoint, tensorboard_callback],
                        workers=12, use_multiprocessing=True)

    model.save(model_path, save_format="tf")

    return history.history, model_full_name


def main():
    kin_model = f"{KINEMATIC_NAME}_robot"

    model = prepare_model(kin_model, BATCH_SIZE)
    x, y, x_val, y_val = prepare_data(kin_model, BATCH_SIZE, 0.3)
    history, model_full_name = train_model(model, x, y, x_val, y_val, BATCH_SIZE)

    x_test, y_test = prepare_test_data(kin_model, BATCH_SIZE)

    eval_res = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE, return_dict=True)
    print(f"Trained model:\n {eval_res}")
    # path_to_history_json = HISTS_DIR / f"{model_full_name}.json"
    # with open(path_to_history_json, mode='w') as f:
    #     json.dump(eval_res, f)

    adam_model = MomentumModel(kin_model, BATCH_SIZE, N_ITERS)
    adam_model.compile(loss=CompactL2L2(1.0, 0.0), metrics=[metrics.gamma_dx, metrics.gamma_dy,
                                                            metrics.gamma_dz, metrics.angle_axis_l2])
    eval_res = adam_model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE, return_dict=True)
    print(f"Momentum model:\n {eval_res}")

    # adam_model = AdamModel(kin_model, BATCH_SIZE, N_ITERS)
    # adam_model.compile(loss=CompactL2L2(1.0, 0.0), metrics=[metrics.gamma_dx, metrics.gamma_dy,
    #                                                         metrics.gamma_dz, metrics.angle_axis_l2])
    # eval_res = adam_model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE, return_dict=True)
    # print(f"Adam model:\n {eval_res}")

    plot_training_history(history, PATH_TO_PICS / f'{model_full_name}.png')


if __name__ == '__main__':
    main()
