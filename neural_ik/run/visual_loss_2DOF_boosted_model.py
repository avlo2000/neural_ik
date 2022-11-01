from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from neural_ik.metrics import gamma_dx, gamma_dy
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.dlkinematics import DLKinematics

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '2k'
BATCH_SIZE = 32
metrics = [gamma_dx, gamma_dy]
tf.keras.backend.set_floatx('float64')
tf.config.set_visible_devices([], 'GPU')

model: keras.Model = load_model(PATH_TO_MODELS / 'momentum_recurrent_grad_boost_bs1__dof2d_10k___experiment')


def func(theta1: tf.Tensor, theta2: tf.Tensor, y_goal: tf.Tensor, loss_fn):
    original_shape = theta1.shape
    goal_compact = tf.reshape(tf_compact(y_goal), [1, 1, 6])

    @tf.function
    def fn(th):
        t1, t2 = th
        thetas = tf.reshape(tf.convert_to_tensor([t1, t2], dtype=tf.float64), shape=(1, 2))
        model.n_iters = 1
        goal_actual = model((thetas, y_goal))
        goal_actual = tf.reshape(goal_actual, [1, 1, 6])
        ls = loss_fn(goal_compact, goal_actual)
        return ls

    theta1 = tf.reshape(theta1, shape=[-1])
    theta2 = tf.reshape(theta2, shape=[-1])
    loss = []
    for i in tqdm(range(theta1.shape[0])):
        loss.append(fn([theta1[i], theta2[i]]))

    return tf.reshape(tf.convert_to_tensor(loss), shape=original_shape)


def loss_l_inf(y, y_goal):
    return tf.reduce_max(tf.abs(y - y_goal), axis=2)


def loss_l4(y, y_goal):
    return tf.math.pow(tf.reduce_sum((y - y_goal) ** 4, axis=2), 1 / 4)


def loss_l2(y, y_goal):
    return tf.math.pow(tf.reduce_sum((y - y_goal) ** 2, axis=2), 1 / 2)


def loss_l1(y, y_goal):
    return tf.reduce_sum(tf.abs(y - y_goal), axis=2)


def loss_l1l1(y, y_goal):
    return loss_l1(y[..., :3], y_goal[..., :3]) + loss_l1(y[..., 3:], y_goal[..., 3:])


def loss_l2l2(y, y_goal):
    return loss_l2(y[..., :3], y_goal[..., :3]) + loss_l2(y[..., 3:], y_goal[..., 3:])


def main():
    kin = load('dof2d_robot', 1)

    fig = plt.figure(figsize=(16, 14))

    theta1 = tf.range(-np.pi, np.pi + .1, 0.1, dtype=tf.float64)
    theta2 = tf.range(-np.pi, np.pi + .1, 0.1, dtype=tf.float64)

    theta1, theta2 = tf.meshgrid(theta1, theta2)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    loss = func(theta1, theta2, tf.constant([1.0, 0.0, 0.0, 0.2,
                                             0.0, 1.0, 0.0, 1.0,
                                             0.0, 0.0, 1.0, 0.0,
                                             0.0, 0.0, 0.0, 1.0,
                                             ], dtype=tf.float64, shape=(1, 4, 4)), loss_l2)
    surf = ax.plot_surface(theta1, theta2, loss, cmap='Reds')
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('loss_l1l1', labelpad=50)

    fig.colorbar(surf)
    plt.show()


if __name__ == '__main__':
    main()
