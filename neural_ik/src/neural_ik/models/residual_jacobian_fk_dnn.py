import tensorflow as tf
from keras import layers
from keras import Model

from neural_ik.models.common import fk_theta_iters_dist
from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.layers import ForwardKinematics, IsometryCompact, IsometryInverse


def residual_fk_dnn(kin: DLKinematics, blocks_count: int, corrector_units: int) -> (Model, layers.Layer):
    assert blocks_count > 0
    activation_fn = tf.nn.tanh

    theta_input = tf.keras.Input(kin.dof)
    iso_input = tf.keras.Input((4, 4))
    iso_inv = IsometryInverse()(iso_input)

    def corrector_block(concat_layer: layers.Concatenate) -> layers.Layer:
        enc = layers.Dense(corrector_units, activation=activation_fn)(concat_layer)
        dec = layers.Dense(kin.dof, activation=activation_fn)(enc)
        return dec

    def fk_diff(theta: layers.Layer):
        fk = ForwardKinematics(kin)(theta)
        diff = layers.Multiply()([fk, iso_inv])
        return diff

    def residual_block(theta_in: layers.Layer):
        diff = fk_diff(theta_in)
        gamma_diff_compact = IsometryCompact()(diff)

        concat = layers.Concatenate()([theta_in, gamma_diff_compact])
        theta_out = corrector_block(concat)
        return theta_out

    theta_iters = [residual_block(theta_input)]
    for i in range(blocks_count - 1):
        theta_iters.append(residual_block(theta_iters[-1]))

    concat_norms = fk_theta_iters_dist(kin, theta_iters, iso_inv)

    model_dist = Model(inputs=[theta_input, iso_input], outputs=concat_norms, name="residual_fk_dnn_dist")
    model_ik = Model(inputs=[theta_input, iso_input], outputs=theta_iters, name="residual_fk_dnn_ik")
    return model_dist, model_ik
