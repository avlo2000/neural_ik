import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def func(theta1, theta2, theta3, y_goal):
    r1 = np.array([[np.cos(theta1), -np.sin(theta1), 1.0],
                   [np.sin(theta1), np.cos(theta1), 0.0],
                   [0.0, 0.0, 1.0]])
    r2 = np.array([[np.cos(theta2), -np.sin(theta2), 1.0],
                   [np.sin(theta2), np.cos(theta2), 1.0],
                   [0.0, 0.0, 1.0]])
    r3 = np.array([[np.cos(theta3), -np.sin(theta3), 1.0],
                   [np.sin(theta3), np.cos(theta3), 1.0],
                   [0.0, 0.0, 1.0]])
    r = r1 @ r2 @ r3
    y = (r @ np.ones(shape=(3, 1)))
    ang_cos = (np.trace(r) - 1.0) / 2
    y[2, 0] = ang_cos

    z = loss_l2(y[:3], y_goal)
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


def main():
    fig = plt.figure(figsize=(16, 14))
    ax = plt.axes(projection='3d')

    theta1 = np.arange(-np.pi, np.pi + .1, 0.1)
    theta2 = np.arange(-np.pi, np.pi + .1, 0.1)
    theta3 = np.arange(-np.pi, np.pi + .1, 0.1)
    x, y, z = np.meshgrid(theta1, theta2, theta3)
    loss = func(x, y, z, np.array([1.0, 1.0, 1.0]))

    surf = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=loss.flatten(), cmap='Reds', alpha=0.6)

    # Set axes label
    ax.set_xlabel('theta1', labelpad=50)
    ax.set_ylabel('theta2', labelpad=50)
    ax.set_zlabel('theta3', labelpad=50)

    fig.colorbar(surf)

    plt.show()


if __name__ == '__main__':
    main()
