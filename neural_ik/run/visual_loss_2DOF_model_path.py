import keras
import numpy as np
import tensorflow as tf
from keras.saving.save import load_model
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.dlkinematics import DLKinematics
from pathlib import Path

from neural_ik.metrics import gamma_dx, gamma_dy
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.dlkinematics import DLKinematics

tf.config.set_visible_devices([], 'GPU')

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '2k'
BATCH_SIZE = 32
metrics = [gamma_dx, gamma_dy]
tf.keras.backend.set_floatx('float64')
tf.config.set_visible_devices([], 'GPU')


def model_path(y_goal: tf.Tensor, lr_model, beta_model, grad_model):
    theta_seed = tf.reshape(tf.convert_to_tensor([0, 0], dtype=tf.float64), shape=(1, 2))
    gamma = tf_compact(y_goal)
    losses = []
    n_iters = 10

    momentum = tf.zeros_like(theta_seed)
    for _ in range(n_iters):
        grad = grad_model([gamma, theta_seed])
        grad_gamma_and_seed = tf.concat([grad, gamma, theta_seed], axis=1)

        lr = lr_model(grad_gamma_and_seed)
        beta = beta_model(grad_gamma_and_seed)

        lr = tf.clip_by_value(lr, clip_value_min=0.0001, clip_value_max=6.9999)
        beta = tf.clip_by_value(beta, clip_value_min=0.5001, clip_value_max=0.9999)
        momentum = (1.0 - beta) * momentum + beta * grad

        theta_seed = theta_seed - lr * momentum
        print(theta_seed.numpy())
    return losses


def func(theta1: tf.Tensor, theta2: tf.Tensor, kin: DLKinematics, y_goal: tf.Tensor, loss_fn):
    original_shape = theta1.shape

    def fk(th):
        t1, t2 = th
        thetas = tf.convert_to_tensor([t1, t2], dtype=tf.float64)
        return tf_compact(kin.forward(tf.reshape(thetas, [-1])))

    theta1 = tf.reshape(theta1, shape=[-1])
    theta2 = tf.reshape(theta2, shape=[-1])
    y_actual = tf.vectorized_map(fk, (theta1, theta2))
    y_actual = tf.reshape(y_actual, [original_shape[0], original_shape[1], 6])
    y_goal = tf.reshape(y_goal, [1, 1, 6])
    loss = loss_fn(y_goal, y_actual)
    return loss


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

    model: keras.Model = load_model(PATH_TO_MODELS / 'momentum_recurrent_grad_boost_bs1__dof2d_10k___experiment')
    lr_model = model.lr_corrector
    beta_model = model.beta_corrector
    grad = SolveCompactIterGrad('mse', 'dof2d_robot', 1)

    fig = plt.figure(figsize=(16, 14))

    theta1 = tf.range(-np.pi, np.pi + .1, 0.1)
    theta2 = tf.range(-np.pi, np.pi + .1, 0.1)

    theta1, theta2 = tf.meshgrid(theta1, theta2)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    y_goal = tf.constant([1.0, 0.0, 0.0, 1.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0,
                          ], dtype=tf.float64, shape=(1, 4, 4))

    model_path(y_goal, lr_model, beta_model, grad)
    loss = func(theta1, theta2, kin, y_goal, loss_l2)
    surf = ax.plot_surface(theta1, theta2, loss, cmap='Reds')
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('loss_l1l1', labelpad=50)

    # ax = fig.add_subplot(1, 2, 2)
    # ax.quiver(theta1, theta2, grad_u, grad_v)
    # ax.set_xlabel('grad_theta1', labelpad=50)
    # ax.set_ylabel('grad_theta2', labelpad=50)

    fig.colorbar(surf)
    plt.show()


if __name__ == '__main__':
    main()
