from typing import Iterable
from keras import layers
from tf_kinematics.kin_layers import ForwardKinematics
from tf_kinematics.iso_layers import CompactL2Norm, IsometryInverse, IsometryCompact, IsometryMul

LayerList = Iterable[layers.Layer]


def fk_theta_iters_dist(kin_model_name: str, batch_size: int,
                        theta_iters: LayerList, iso_gaol: layers.Layer) -> layers.Layer:
    fk_iters = [ForwardKinematics(kin_model_name, batch_size)(theta_iter) for theta_iter in theta_iters]
    compacts = [IsometryCompact()(fk_iter) for fk_iter in fk_iters]
    compacts_diff = [layers.Subtract()([compact, IsometryCompact()(iso_gaol)]) for compact in compacts]
    norms = [CompactL2Norm(1.0, 1.0)(compact) for compact in compacts_diff]
    concat_norms = layers.Concatenate(axis=1)(norms)
    return concat_norms
