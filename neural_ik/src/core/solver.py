import abc
from typing import Optional, Mapping

import numpy as np
from visual_kinematics.Frame import Frame


class Solver(abc.ABC):
    @abc.abstractmethod
    def solve_ik(self, *args, **kwargs) -> (Optional[np.ndarray], Mapping[str, float]):
        pass

    @abc.abstractmethod
    def solve_fk(self, theta: np.ndarray) -> Frame:
        pass
