from typing import Iterable
from keras import layers
from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.kin_layers import ForwardKinematics, IsometryWeightedL2Norm, IsometryInverse


LayerList = Iterable[layers.Layer]


def fk_theta_iters_dist(kin: DLKinematics, theta_iters: LayerList, gamma_goal_inv: IsometryInverse) -> layers.Layer:
    fk_iters = [ForwardKinematics(kin)(theta_iter) for theta_iter in theta_iters]
    iso_diffs = [layers.Multiply()([fk_iter, gamma_goal_inv]) for fk_iter in fk_iters]
    norms = [IsometryWeightedL2Norm(1.0, 1.0)(iso_diff) for iso_diff in iso_diffs]
    concat_norms = layers.Concatenate(axis=1)(norms)
    return concat_norms

