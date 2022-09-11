from typing import Optional

import numpy as np
from visual_kinematics import Frame

from core.ik_solver import IKSolver


class VisualKinematicsIKSolver(IKSolver):

    def solve(self, pose: Frame) -> Optional[np.ndarray]:
        self._robot.inverse(pose)
        return self._robot.axis_values
