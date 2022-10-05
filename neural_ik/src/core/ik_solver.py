import abc
from typing import Optional, Callable

import numpy as np
from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.Frame import Frame

LossFunction = Callable[[Frame, Frame], float]


def zero_loss_fn(*_) -> float:
    return 0.0


class IKSolver(abc.ABC):
    def __init__(self, loss_fn: LossFunction = zero_loss_fn):
        self.__loss_fn = loss_fn
        self.__loss = 0.0

    @abc.abstractmethod
    def solve(self, pose: Frame) -> Optional[np.ndarray]:
        pass

    @property
    def loss(self) -> float:
        return self.__loss
