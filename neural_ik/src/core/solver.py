import abc
from typing import Optional

import numpy as np
from visual_kinematics.Frame import Frame


class Solver(abc.ABC):
    @abc.abstractmethod
    def solve_ik(self, *args) -> Optional[np.ndarray]:
        pass

    @abc.abstractmethod
    def solve_fk(self, theta: np.ndarray) -> Frame:
        pass
