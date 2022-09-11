import matplotlib.pyplot as plt

from core.evaluate_report import EvaluateReport
from core.evaluate import evaluate
from core.visual_kinematics_ik_solver import VisualKinematicsIKSolver

from data.robots import arm6dof
from data.dataset import read, vec_to_frame


def main():
    ik_solver = VisualKinematicsIKSolver(arm6dof())
    frames_raw, q_states = read(r"C:\Users\Pavlo\source\repos\math\neural_ik\data\train_fk_ds_0_0_1.csv")
    frames = list(map(vec_to_frame, frames_raw))
    report = evaluate(ik_solver, frames, q_states)


if __name__ == '__main__':
    main()