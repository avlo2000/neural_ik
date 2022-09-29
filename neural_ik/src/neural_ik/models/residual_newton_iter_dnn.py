from typing import Iterable

import tensorflow as tf
from keras import layers
from keras import Model
from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.layers import IsometryCompact, IsometryInverse, NewtonIter
from neural_ik.models.common import fk_theta_iters_dist


def residual_newton_iter_dnn(kin: DLKinematics, *, blocks_count: int, corrector_units: int) -> (Model, Iterable[str]):
    assert blocks_count > 0
    activation_fn = tf.nn.tanh

    theta_input = tf.keras.Input(kin.dof)
    iso_input = tf.keras.Input((4, 4))
    gamma_compact = IsometryCompact()(iso_input)
    iso_inv = IsometryInverse(name='iso_inv')(iso_input)

    theta_layers_names = []

    def corrector_block(in_layer: layers.Layer, idx: int) -> layers.Layer:
        theta_layers_names.append(f'decoder_{idx}')
        enc = layers.Dense(corrector_units, activation=activation_fn, name=f'encoder_{idx}')(in_layer)
        dec = layers.Dense(kin.dof, activation=activation_fn, name=theta_layers_names[-1])(enc)
        return dec

    def residual_block(theta_in: layers.Layer, idx: int):
        newton = NewtonIter(kin, learning_rate=0.1, return_diff=False, name=f'newton_iter_{idx}') \
            ([gamma_compact, theta_in])
        theta_out = corrector_block(newton, idx)
        return theta_out

    theta_iters = [residual_block(theta_input, 0)]
    for i in range(blocks_count - 1):
        theta_iters.append(residual_block(theta_iters[-1], i + 1))

    concat_norms = fk_theta_iters_dist(kin, theta_iters, iso_inv)

    model_dist = Model(inputs=[theta_input, iso_input], outputs=concat_norms, name="residual_newton_iter_dnn_dist")
    model_ik = Model(inputs=[theta_input, iso_input], outputs=theta_iters, name="residual_newton_iter_dnn_ik")
    return model_dist, model_ik
