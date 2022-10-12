import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def func(theta1, theta2, y_goal):
    r1 = np.array([[np.cos(theta1), -np.sin(theta1), 1.0],
                   [np.sin(theta1), np.cos(theta1), 0.0],
                   [0.0, 0.0, 1.0]])
    r2 = np.array([[np.cos(theta2), -np.sin(theta2), 1.0],
                   [np.sin(theta2), np.cos(theta2), 1.0],
                   [0.0, 0.0, 1.0]])
    y = (r1 @ r2 @ np.ones(shape=(3, 1)))

    z = loss_l4(y[:2], y_goal)
    return z


def loss_l4(y, y_goal):
    return np.sum((y - y_goal) ** 4)


def loss_l2(y, y_goal):
    return np.sum((y - y_goal) ** 2)


def loss_l1(y, y_goal):
    return np.sum(np.abs(y - y_goal))


def main():
    fig = plt.figure(figsize=(16, 14))
    ax = plt.axes(projection='3d')

    theta1 = np.arange(-5, 5.1, 0.2)
    theta2 = np.arange(-5, 5.1, 0.2)

    x, y = np.meshgrid(theta1, theta2)
    z = func(x, y, np.array([1.0, 1.0]))

    surf = ax.plot_surface(x, y, z, cmap=plt.cm.cividis)

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()


if __name__ == '__main__':
    main()
