from typing import Iterable
from keras import layers
from keras import activations
from keras.engine.keras_tensor import KerasTensor

from tf_kinematics.kin_layers import ForwardKinematics
from tf_kinematics.iso_layers import CompactL2Norm, IsometryCompact, CompactDiff

import tensorflow as tf

LayerList = Iterable[KerasTensor]


def theta_iters_dist(kin_model_name: str, batch_size: int,
                     theta_iters: LayerList, iso_gaol: KerasTensor) -> KerasTensor:
    fk_iters = [ForwardKinematics(kin_model_name, batch_size)(theta_iter) for theta_iter in theta_iters]
    compacts = [IsometryCompact()(fk_iter) for fk_iter in fk_iters]
    compacts_diff = [layers.Subtract()([compact, IsometryCompact()(iso_gaol)]) for compact in compacts]
    norms = [CompactL2Norm(1.0, 1.0)(compact) for compact in compacts_diff]
    concat_norms = layers.Concatenate(axis=1)(norms)
    return concat_norms


def fk_compact_iters_dist(fk_compact_iters: LayerList, iso_gaol: KerasTensor) -> KerasTensor:
    iso_goal_compact = IsometryCompact()(iso_gaol)
    compacts_diff = [CompactDiff()([compact, iso_goal_compact]) for compact in fk_compact_iters]
    norms = [CompactL2Norm(1.0, 1.0)(compact) for compact in compacts_diff]
    concat_norms = layers.Concatenate(axis=1)(norms)
    return concat_norms


def dnn_block(dof, hidden: Iterable[int], x: KerasTensor) -> KerasTensor:
    activation_fn = tf.nn.leaky_relu
    for h in hidden:
        x = layers.Dense(h, activation=activation_fn)(x)
    x = layers.Dense(dof, activation=activation_fn)(x)
    return x


def linear_identity(x: KerasTensor) -> KerasTensor:
    activation_fn = activations.linear
    k_initializer = tf.keras.initializers.Identity()
    b_initializer = tf.keras.initializers.Zeros()
    x = layers.Dense(x.shape[1], activation=activation_fn,
                     kernel_initializer=k_initializer,
                     bias_initializer=b_initializer)(x)
    return x
