import abc
from typing import Optional, Callable

import numpy as np
from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.Frame import Frame

LossFunction = Callable[[Frame, Frame], float]


def zero_loss_fn(*_) -> float:
    return 0.0


class IKSolver(abc.ABC):
    def __init__(self, robot: Optional[RobotSerial], loss_fn: LossFunction = zero_loss_fn):
        self.robot = robot
        self.__loss_fn = loss_fn
        self.__loss = 0.0

    def solve(self, pose: Frame) -> Optional[np.ndarray]:
        q = self._solve(pose)
        if q is None or self.robot is None:
            return q
        actual_pose = self.robot.forward(q)
        self.__loss = self.__loss_fn(actual_pose, pose)
        return q

    @abc.abstractmethod
    def _solve(self, pose: Frame) -> Optional[np.ndarray]:
        pass

    @property
    def loss(self) -> float:
        return self.__loss
