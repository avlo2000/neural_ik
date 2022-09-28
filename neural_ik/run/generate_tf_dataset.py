from data.data_io import write
from data.tf_kin_data import generate_with_theta_seed
from tf_kinematics import kinematic_models


def main():
    kin = kinematic_models.omnipointer_robot(1)
    size = 8_000
    seed_multiplier = 0.1

    feature_names, raw_data = generate_with_theta_seed(kin, size, seed_multiplier)
    with open(r'C:\Users\Pavlo\source\repos\math\neural_ik\data\omnipointer_train_10k.csv', 'w') as file:
        write(feature_names, raw_data, file)


if __name__ == '__main__':
    main()
