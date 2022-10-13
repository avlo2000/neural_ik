import numpy as np
import tensorflow as tf
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.dlkinematics import DLKinematics


def func(theta1: tf.Tensor, theta2: tf.Tensor, kin: DLKinematics):
    def fk(t1, t2):
        thetas = tf.convert_to_tensor([t1, t2, 1.0, 1.0, 1.0, 1.0, 1.0])
        return kin.forward(thetas)
    z = fk(theta1, theta2)
    return z


def loss_entropy(y, y_goal):
    return -np.sum(y * np.log(y_goal))


def loss_l_inf(y, y_goal):
    return np.sum((y - y_goal) ** 50)


def loss_l4(y, y_goal):
    return np.sum((y - y_goal) ** 4)


def loss_l2(y, y_goal):
    return np.sum((y - y_goal) ** 2)


def loss_l1(y, y_goal):
    return np.sum(np.abs(y - y_goal))


def loss_l1l1(y, y_goal):
    return np.sum(np.abs(y - y_goal))


def main():
    kin = load('kuka_robot', 1)

    fig = plt.figure(figsize=(16, 14))
    ax = plt.axes(projection='3d')

    theta1 = tf.range(-np.pi, np.pi + .1, 0.1)
    theta2 = tf.range(-np.pi, np.pi + .1, 0.1)

    theta1, theta2 = tf.meshgrid(theta1, theta2)
    func(theta1, theta2, kin)

    # fig.colorbar(surf)
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('loss', labelpad=50)
    plt.show()


if __name__ == '__main__':
    main()
