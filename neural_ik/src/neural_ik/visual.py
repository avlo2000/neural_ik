import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider


def plot_training_history(history, save_path=None):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    if save_path is not None:
        plt.savefig(save_path)
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


