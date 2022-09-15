import numpy as np
import time
from typing import Sequence

import tqdm
from visual_kinematics.Frame import Frame

from core.ik_solver import IKSolver
from core.evaluate_report import EvaluateReport


def evaluate(solver: IKSolver, x_seq: Sequence[Frame], y_seq: Sequence[np.ndarray]) -> EvaluateReport:
    total = len(x_seq)
    found = 0
    times = []
    losses = []

    for x, y_actual in tqdm.tqdm(zip(x_seq, y_seq)):
        t0 = time.time()
        y = solver.solve(x)
        t1 = time.time()
        times.append(t1 - t0)

        found += 1 if y is not None else 0
        losses.append(solver.loss)

    return EvaluateReport(found, total, np.asarray(times), np.asarray(losses))
