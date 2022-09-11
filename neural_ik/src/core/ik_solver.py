import abc
from typing import Optional

import numpy as np
from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.Frame import Frame


class IKSolver(abc.ABC):
    def __init__(self, robot: Optional[RobotSerial]):
        self._robot = robot

    @abc.abstractmethod
    def solve(self, pose: Frame) -> Optional[np.ndarray]:
        pass
