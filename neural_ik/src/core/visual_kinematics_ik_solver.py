from typing import Optional

import numpy as np
from visual_kinematics import Frame

from core.ik_solver import IKSolver


class VisualKinematicsIKSolver(IKSolver):

    def _solve(self, pose: Frame) -> Optional[np.ndarray]:
        self.robot.inverse(pose)
        if self.robot.is_reachable_inverse:
            return self.robot.axis_values
        return None
