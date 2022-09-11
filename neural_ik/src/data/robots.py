from visual_kinematics.RobotSerial import RobotSerial
import numpy as np


def arm7dof():
    dh_params = np.array([[0.34, 0., -np.pi / 2, 0.],
                          [0.1, 0., np.pi / 2, 0.],
                          [0.4, 0., -np.pi / 2, 0.],
                          [0.1, 0., np.pi / 2, 0.],
                          [0.4, 0., -np.pi / 2, 0.],
                          [0.1, 0., np.pi / 2, 0.],
                          [0.126, 0., 0., 0.]])

    robot = RobotSerial(dh_params)
    return robot


def arm6dof():
    dh_params = np.array([[0.34, 0., -np.pi / 2, 0.],
                          [0.1, 0., np.pi / 2, 0.],
                          [0.4, 0., -np.pi / 2, 0.],
                          [0.1, 0., np.pi / 2, 0.],
                          [0.4, 0., -np.pi / 2, 0.],
                          [0.1, 0., np.pi / 2, 0.]])

    robot = RobotSerial(dh_params)
    return robot

