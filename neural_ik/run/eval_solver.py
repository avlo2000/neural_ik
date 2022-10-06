from pathlib import Path

from core.evaluate_error_functions import distance_as_dual_quat_norm
from core.solver_evaluator import SolverEvaluator

from data.data_io import read_csv
from data.tf_kin_data import rawdata_to_dataset

from inference.adam_solver import AdamSolver
from inference.newtor_solver import NewtonSolver

from tf_kinematics.kinematic_models_io import load

PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()
KINEMATIC_NAME = 'kuka'
DATASET_SIZE_SUF = '2k'


def prepare_data(kin_model):
    kin = load(kin_model, 1)
    path_to_data = PATH_TO_DATA / f'{KINEMATIC_NAME}_test_{DATASET_SIZE_SUF}.csv'
    print(f"Loading {path_to_data}")
    with open(path_to_data, mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    x = list(zip(iso_transforms, thetas_seed))
    y = thetas
    return x, y


def main():
    kin_model = f"{KINEMATIC_NAME}_robot"
    solver = NewtonSolver(32, 1e-6, kin_model)
    evaluator = SolverEvaluator(distance_as_dual_quat_norm, solver)

    x, y = prepare_data(kin_model)
    report = evaluator.evaluate(x, y)
    print(report)


if __name__ == '__main__':
    main()
