import numpy as np
import tensorflow as tf
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.dlkinematics import DLKinematics

tf.config.set_visible_devices([], 'GPU')


def func(theta1: tf.Tensor, theta2: tf.Tensor, kin: DLKinematics, y_goal: tf.Tensor, loss_fn):
    original_shape = theta1.shape

    def fk(th):
        t1, t2 = th
        thetas = tf.convert_to_tensor([t1, t2], dtype=tf.float64)
        return tf_compact(kin.forward(tf.reshape(thetas, [-1])))

    with tf.GradientTape() as tape:
        theta1 = tf.reshape(theta1, shape=[-1])
        theta2 = tf.reshape(theta2, shape=[-1])
        tape.watch(theta1)
        tape.watch(theta2)
        y_actual = tf.vectorized_map(fk, (theta1, theta2))
        y_actual = tf.reshape(y_actual, [original_shape[0], original_shape[1], 6])
        y_goal = tf.reshape(y_goal, [1, 1, 6])
        loss = loss_fn(y_goal, y_actual)
    grad1, grad2 = tape.gradient(loss, [theta1, theta2])
    grad1 = tf.reshape(grad1, original_shape)
    grad2 = tf.reshape(grad2, original_shape)
    return loss, grad1, grad2


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

    theta1 = tf.range(-np.pi, np.pi + .1, 0.1)
    theta2 = tf.range(-np.pi, np.pi + .1, 0.1)

    theta1, theta2 = tf.meshgrid(theta1, theta2)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    loss, grad_u, grad_v = func(theta1, theta2, kin, tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float64),
                                loss_l2)
    surf = ax.plot_surface(theta1, theta2, loss, cmap='Reds')
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('loss_l1l1', labelpad=50)

    ax = fig.add_subplot(1, 2, 2)
    ax.quiver(theta1, theta2, grad_u, grad_v)
    ax.set_xlabel('grad_theta1', labelpad=50)
    ax.set_ylabel('grad_theta2', labelpad=50)

    fig.colorbar(surf)
    plt.show()




if __name__ == '__main__':
    main()
