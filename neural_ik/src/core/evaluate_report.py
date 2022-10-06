from dataclasses import dataclass

import numpy as np


@dataclass(init=True, repr=True, frozen=False)
class EvaluateReport:
    found: int
    total: int
    times: np.ndarray


