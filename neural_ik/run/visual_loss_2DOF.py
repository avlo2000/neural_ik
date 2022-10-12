import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def func(theta1, theta2, y_goal):
    r1 = np.array([[np.cos(theta1), -np.sin(theta1), 1.0],
                   [np.sin(theta1), np.cos(theta1), 0.0],
                   [0.0, 0.0, 1.0]])
    r2 = np.array([[np.cos(theta2), -np.sin(theta2), 1.0],
                   [np.sin(theta2), np.cos(theta2), 0.0],
                   [0.0, 0.0, 1.0]])
    y = (r1 @ r2 @ np.ones(shape=(3, 1)))

    z = loss_l2(y[:2], y_goal)
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

    surf = None
    for step in np.arange(0, 5.0, 1.0):
        x, y = np.meshgrid(theta1, theta2)
        z = func(x, y, np.array([step, step]))
        surf = ax.plot_surface(x, y, z, cmap='Reds')

    fig.colorbar(surf)

    plt.show()


if __name__ == '__main__':
    main()
