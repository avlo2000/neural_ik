from time import sleep
from typing import Optional
import numpy as np

from core.solver import Solver
from core.solver_evaluator import SolverEvaluator
from unittest import TestCase

from visual_kinematics import Frame

from data.data_io import frame_to_vec, vec_to_frame


def zero_error_fn(*_) -> float:
    return 0.0


class SolverMock(Solver):
    def __init__(self):
        super().__init__()

    def solve_ik(self, pose) -> Optional[np.ndarray]:
        sleep(0.01)
        if pose is None:
            return None
        return np.zeros(6)

    def solve_fk(self, theta: np.ndarray) -> Frame:
        return Frame.i_4_4()


class TestEvaluate(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 0.0001

    def test_evaluate_returns_correct_report(self):
        reachable = 100
        nones = 50
        frames = [[Frame(np.identity(4))]] * reachable + [[None]] * nones
        q_states = [np.zeros(6)] * (reachable + nones)

        evaluator = SolverEvaluator(zero_error_fn, SolverMock())
        report = evaluator.evaluate(frames, q_states)

        self.assertEqual(report.total, reachable + nones)
        self.assertEqual(report.found, reachable)
        self.assertTrue((report.times > 0).all())

    def test_frame_to_vec_transitivity(self):
        frame = Frame.from_r_3([0.30369251, 0.68088033, 0.98290186], [[0.56932648],
                                                                      [0.39061527],
                                                                      [0.0554877]])
        vec = frame_to_vec(frame)
        frame_back = vec_to_frame(vec)

        self.assertTrue(frame.distance_to(frame_back) <= self.eps)

    def test_vec_to_frame_transitivity(self):
        vec = np.asarray([0.15180924, 0.56683175, 0.34126947, 0.00432348, 0.6785767, 0.55012793, 0.76428717])
        frame = vec_to_frame(vec)
        vec_back = frame_to_vec(frame)

        self.assertTrue(np.max(vec - vec_back) <= self.eps)
