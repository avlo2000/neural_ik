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
        self.assertTrue((report.losses > 0).all())
        self.assertTrue(report.losses.sum() > 0)
        self.assertTrue((report.times > 0).all())

    def test_evaluate_report_has_correct_repr(self):
        report = EvaluateReport(100, 50, np.ones(100), np.ones(100))
        self.assertEqual(dataclasses.asdict(report),
                         {
                             'found': 100,
                             'total': 50,
                             'average_time': np.ones(100),
                             'average_loss': np.ones(100)
                         })
