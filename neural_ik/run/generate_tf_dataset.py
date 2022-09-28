import math
from itertools import product

from tqdm import tqdm
from data.dataset import write
from tf_kinematics import kinematic_models

import tensorflow as tf


def rand_angles(shape: tuple):
    return tf.random.uniform(shape=shape, minval=-math.pi, maxval=math.pi)


def main():
    kin = kinematic_models.kuka_robot(1)
    size = 1000
    seed_multiplier = 0.1

    thetas_seed = []
    thetas = []
    iso_transforms = []

    for _ in tqdm(range(size)):
        thetas.append(rand_angles(shape=(1, kin.dof)))
        thetas_seed.append(thetas[-1] + seed_multiplier * rand_angles(shape=(1, kin.dof)))
        iso_transforms.append(kin.forward(tf.reshape(thetas[-1], [-1])))

    feature_names = [f'theta_{i}' for i in range(kin.dof)] + \
                    [f'thetas_seed_{i}' for i in range(kin.dof)] + \
                    [f'm_{i}{j}' for i, j in product(range(4), repeat=2)]
    raw_data = []
    for theta, theta_seed, iso in zip(thetas, thetas_seed, iso_transforms):
        raw = tf.concat([theta, theta_seed, tf.reshape(iso, [1, -1])], axis=1).numpy()
        raw_data.append(raw)
    write(feature_names, raw_data, r'C:\Users\Pavlo\source\repos\math\neural_ik\data\testtest.csv')


if __name__ == '__main__':
    main()
