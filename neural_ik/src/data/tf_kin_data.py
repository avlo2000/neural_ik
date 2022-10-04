import os
from itertools import product
from typing import Iterable, Any
from tqdm import tqdm
from tf_kinematics.dlkinematics import DLKinematics
import tensorflow as tf


def rand_state(kin: DLKinematics):
    return tf.random.uniform(shape=(1, kin.dof), minval=kin.limits[0], maxval=kin.limits[1])


def rawdata_to_dataset(kin: DLKinematics, feat_names: Iterable[str], raw_data: Iterable[Iterable[Any]]) -> \
                                                                                    (tf.Tensor, tf.Tensor, tf.Tensor):
    feature_names = [f'theta_{i}' for i in range(kin.dof)] + \
                    [f'thetas_seed_{i}' for i in range(kin.dof)] + \
                    [f'm_{i}{j}' for i, j in product(range(4), repeat=2)]
    assert feature_names == feat_names

    thetas = []
    thetas_seed = []
    iso_transforms = []

    for sample in tqdm(raw_data):
        thetas.append(tf.convert_to_tensor(sample[:kin.dof]))
        thetas_seed.append(tf.convert_to_tensor(sample[kin.dof:2*kin.dof]))
        iso_transforms.append(tf.reshape(tf.convert_to_tensor(sample[2*kin.dof:]), shape=(4, 4)))

        tf.debugging.check_numerics(thetas[-1], f"thetas has NaNs")
        tf.debugging.check_numerics(thetas_seed[-1], f"thetas_seed has NaNs")
        tf.debugging.check_numerics(iso_transforms[-1], f"iso_transforms has NaNs")

    thetas = tf.squeeze(tf.stack(thetas))
    thetas_seed = tf.squeeze(tf.stack(thetas_seed))
    iso_transforms = tf.squeeze(tf.stack(iso_transforms))
    return thetas, thetas_seed, iso_transforms


def generate_with_theta_seed(kin: DLKinematics, size: int, seed_multiplier: float) -> (Iterable[str],
                                                                                       Iterable[Iterable[Any]]):
    thetas = []
    thetas_seed = []
    iso_transforms = []

    for _ in tqdm(range(size)):
        thetas.append(rand_state(kin))
        thetas_seed.append(thetas[-1] + seed_multiplier * rand_state(kin))
        iso_transforms.append(kin.forward(tf.reshape(thetas[-1], [-1])))

    feature_names = [f'theta_{i}' for i in range(kin.dof)] + \
                    [f'thetas_seed_{i}' for i in range(kin.dof)] + \
                    [f'm_{i}{j}' for i, j in product(range(4), repeat=2)]
    raw_data = []
    for theta, theta_seed, iso in zip(thetas, thetas_seed, iso_transforms):
        raw = tf.squeeze(tf.concat([theta, theta_seed, tf.reshape(iso, [1, -1])], axis=1)).numpy().tolist()
        raw_data.append(raw)
    return feature_names, raw_data
