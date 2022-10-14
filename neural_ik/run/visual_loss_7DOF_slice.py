import numpy as np
import tensorflow as tf
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.dlkinematics import DLKinematics


def func(theta1: tf.Tensor, theta2: tf.Tensor, kin: DLKinematics, y_goal: tf.Tensor, loss_fn):
    original_shape = theta1.shape
    theta1 = tf.reshape(theta1, shape=[-1])
    theta2 = tf.reshape(theta2, shape=[-1])

    def fk(th):
        t1, t2 = th
        thetas = tf.convert_to_tensor([t1, t2, 0.0, 0.0, 1.0, 1.0, 1.0])
        return tf_compact(kin.forward(tf.reshape(thetas, [-1])))
    y_actual = tf.vectorized_map(fk, (theta1, theta2))
    y_actual = tf.reshape(y_actual, [original_shape[0], original_shape[1], 6])
    y_goal = tf.reshape(y_goal, [1, 1, 6])
    loss = loss_fn(y_goal, y_actual)
    return loss


def loss_l_inf(y, y_goal):
    return tf.reduce_sum((y - y_goal) ** 50, axis=2)


def loss_l4(y, y_goal):
    return tf.reduce_sum((y - y_goal) ** 4, axis=2)


def loss_l2(y, y_goal):
    return tf.reduce_sum((y - y_goal) ** 2, axis=2)


def loss_l1(y, y_goal):
    return tf.reduce_sum(tf.abs(y - y_goal), axis=2)


def loss_l1l1(y, y_goal):
    return 10*tf.reduce_sum(tf.abs(y[..., :3] - y_goal[..., :3]), axis=2) + \
           tf.reduce_sum(tf.abs(y[..., 3:] - y_goal[..., 3:]), axis=2)


def main():
    kin = load('kuka_robot', 1)

    fig = plt.figure(figsize=(16, 14))

    theta1 = tf.range(-np.pi, np.pi + .1, 0.02)
    theta2 = tf.range(-np.pi, np.pi + .1, 0.02)

    theta1, theta2 = tf.meshgrid(theta1, theta2)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    loss = func(theta1, theta2, kin, tf.constant([1.0, 1.0, 1.0, 1.0, 0.0, 1.0]), loss_l1l1)
    ax.plot_surface(theta1, theta2, loss, cmap='Reds')
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('loss_l1l1', labelpad=50)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    loss = func(theta1, theta2, kin, tf.constant([1.0, 1.0, 1.0, 1.0, 0.0, 1.0]), loss_l4)
    surf = ax.plot_surface(theta1, theta2, loss, cmap='Reds')

    fig.colorbar(surf)
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('loss_l4', labelpad=50)
    plt.show()


if __name__ == '__main__':
    main()
