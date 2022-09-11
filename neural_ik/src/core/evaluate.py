import numpy as np
import time
from typing import Sequence
from visual_kinematics.Frame import Frame

from core.ik_solver import IKSolver
from core.evaluate_report import EvaluateReport


def distance_q_l2(q1: np.ndarray, q2: np.ndarray) -> float:
    return np.linalg.norm(q1 - q2)


def distance_q_max(q1: np.ndarray, q2: np.ndarray) -> float:
    return np.max(np.abs(q1 - q2))


def evaluate(solver: IKSolver, x_seq: Sequence[Frame], y_seq: Sequence[np.ndarray]) -> EvaluateReport:
    total = len(x_seq)
    found = 0
    total_time = 0.0
    total_loss = 0.0

    t0 = time.time()
    for x, y_actual in zip(x_seq, y_seq):
        y = solver.solve(x)
        found += 1 if y is not None else 0
        total_loss += distance_q_max(y_actual, y) if y is not None else 0
    t1 = time.time()
    total_time += t1 - t0

    return EvaluateReport(found, total, total_time / total, total_loss / total)
