from dataclasses import dataclass


@dataclass(init=True, repr=True, frozen=False)
class EvaluateReport:
    found: int
    total: int
    average_time: float
    average_loss: float

