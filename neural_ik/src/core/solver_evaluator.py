import time
from typing import Callable, Sequence, Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from visual_kinematics import Frame

from core.evaluate_report import EvaluateReport
from core.solver import Solver


ErrorFunction = Callable[[Frame, Frame], float]


class SolverEvaluator:
    def __init__(self, error_fn: ErrorFunction, solver: Solver):
        self.__error_fn = error_fn
        self.__solver = solver

    def evaluate(self, x_seq: Sequence[Any], y_seq: Sequence[np.ndarray]):
        total = len(x_seq)
        found = 0
        times = []

        for x_args, y_actual in tqdm(zip(x_seq, y_seq)):
            t0 = time.time_ns()
            y = self.__solver.solve_ik(*x_args)
            t1 = time.time_ns()
            times.append((t1 - t0) / (10 ** 9))

            found += 1 if y is not None else 0

        return EvaluateReport(found, total, np.asarray(times))

