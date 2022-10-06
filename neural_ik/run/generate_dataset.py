from pathlib import Path

from data.data_io import write_csv
from data.tf_kin_data import generate_with_theta_seed
from tf_kinematics import kinematic_models_io


PATH_TO_DATA = Path('../data').absolute()
KINEMATIC_NAME = 'kuka'
SIZE = 10_000
TRAIN_TEST_SPLIT = 0.6
SEED_MAX_DIST = 0.1


def main():
    kin = kinematic_models_io.load(f'{KINEMATIC_NAME}_robot', 1)
    train_size = int(SIZE * TRAIN_TEST_SPLIT)

    feature_names, raw_data = generate_with_theta_seed(kin, SIZE, SEED_MAX_DIST)

    size_suf = str(SIZE // 1000) + 'k'
    with open(PATH_TO_DATA/f'{KINEMATIC_NAME}_train_{size_suf}.csv', 'w') as file:
        write_csv(feature_names, raw_data[:train_size], file)
    with open(PATH_TO_DATA/f'{KINEMATIC_NAME}_test_{size_suf}.csv', 'w') as file:
        write_csv(feature_names, raw_data[train_size:], file)


if __name__ == '__main__':
    main()
