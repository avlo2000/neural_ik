from abc import ABC

import numpy as np
from visual_kinematics import Frame

from core.solver import Solver
from tf_kinematics.kinematic_models_io import load


class FKSolver(Solver, ABC):
    def __init__(self, kin_model_ident: str):
        self._kin_model_ident = kin_model_ident
        self._kernel = load(kin_model_ident, 1)

    def solve_fk(self, theta: np.ndarray) -> Frame:
        tf_iso = self._kernel.forward(theta)
        return Frame(np.squeeze(tf_iso.numpy()))
