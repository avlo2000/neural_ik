import dataclasses
from time import sleep
from typing import Optional
from unittest import TestCase

import numpy as np
from visual_kinematics import Frame

from core.ik_solver import IKSolver
from core.evaluate import evaluate
from core.evaluate_report import EvaluateReport


class IKSolverMock(IKSolver):
    def __init__(self):
        super().__init__(None)

    def _solve(self, pose: Frame) -> Optional[np.ndarray]:
        sleep(0.01)
        if pose is None:
            return None
        return np.zeros(6)


class TestEvaluate(TestCase):

    def test_evaluate_returns_correct_report(self):
        reachable = 100
        nones = 50
        frames = [Frame(np.identity(4))] * reachable + [None] * nones
        q_states = [np.zeros(6)] * (reachable + nones)

        report = evaluate(IKSolverMock(), frames, q_states)

        self.assertEqual(report.total, reachable + nones)
        self.assertEqual(report.found, reachable)
        self.assertTrue((report.losses == 0.).all())
        self.assertTrue((report.times > 0).all())

