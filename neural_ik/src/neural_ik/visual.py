import dataclasses
from typing import Mapping, Iterable

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider


def plot_training_history(history: Mapping[str, Iterable[float]], save_path=None):
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    train_hist = dict()
    for key, val in history.items():
        if not key.startswith('val_'):
            train_hist[key] = val

    n = len(train_hist)

    idx = 1
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    for m_name, m_nums in train_hist.items():
        axs = plt.subplot(n, 1, idx)
        idx += 1

        axs.set_ylabel(m_name)
        axs.grid()

        plt.plot(m_nums, label="train")
        if 'val_'+m_name in history:
            m_nums_val = history['val_'+m_name]
            plt.plot(m_nums_val, label="valid")
        plt.legend(loc="upper left")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_robot_trajectory(robot, trj: list, num_segments: int):
    axes = plt.axes([0.15, .06, 0.75, 0.02])
    slider = Slider(axes, "progress", 0., 1., valinit=0)

    x, y, z = [], [], []
    for i in range(num_segments + 1):
        robot.forward(trj[i])
        x.append(robot.end_frame.t_3_1[0, 0])
        y.append(robot.end_frame.t_3_1[1, 0])
        z.append(robot.end_frame.t_3_1[2, 0])

    def update(_):
        robot.forward(trj[int(np.floor(slider.val * num_segments))])
        robot.draw()
        robot.ax.plot_wireframe(x, y, np.array([z]), color="lightblue")
        robot.figure.canvas.draw_idle()

    slider.on_changed(update)
    robot.forward(trj[0])
    robot.draw()
    robot.ax.plot_wireframe(x, y, np.array([z]), color="lightblue")
    plt.show()


