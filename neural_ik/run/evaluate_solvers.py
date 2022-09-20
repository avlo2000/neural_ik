from core.evaluate_metrics import distance_as_dual_quat_norm
from core.evaluate import evaluate
from core.visual_kinematics_ik_solver import VisualKinematicsIKSolver

from data.robots import arm6dof
from data.data_io import read, vec_to_frame


def main():
    robot = arm6dof()
    ik_solver = VisualKinematicsIKSolver(robot, distance_as_dual_quat_norm)
    frames_raw, q_states = read(r"C:\Users\Pavlo\source\repos\math\neural_ik\data\test_fk_ds_0_0_2.csv")
    frames = list(map(vec_to_frame, frames_raw))
    report = evaluate(ik_solver, frames, q_states)
    print(report)


if __name__ == '__main__':
    main()
