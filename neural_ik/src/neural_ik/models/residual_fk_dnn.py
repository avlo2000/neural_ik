import tensorflow as tf
from keras import layers
from keras import Model

from neural_ik.models.common import theta_iters_dist
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff


def residual_fk_dnn(kin_model_name: str, batch_size: int, blocks_count: int, corrector_units: int) -> Model:
    assert blocks_count > 0
    activation_fn = tf.nn.tanh

    dof = load_kin(kin_model_name, batch_size).dof
    theta_input = tf.keras.Input(dof)
    iso_goal_input = tf.keras.Input((4, 4))

    def corrector_block(concat_layer: layers.Concatenate) -> layers.Layer:
        enc = layers.Dense(corrector_units, activation=activation_fn)(concat_layer)
        dec = layers.Dense(dof, activation=activation_fn)(enc)
        return dec

    def fk_diff(theta: layers.Layer):
        fk = ForwardKinematics(kin_model_name, batch_size)(theta)
        fk_compact = IsometryCompact()(fk)
        iso_goal_compact = IsometryCompact()(iso_goal_input)
        gamma_diff_compact = Diff()([iso_goal_compact, fk_compact])
        return gamma_diff_compact

    def residual_block(theta_in: layers.Layer):
        gamma_diff_compact = fk_diff(theta_in)

        concat = layers.Concatenate()([theta_in, gamma_diff_compact])
        theta_out = corrector_block(concat)
        return theta_out

    theta_iters = [residual_block(theta_input)]
    for i in range(blocks_count - 1):
        theta_iters.append(residual_block(theta_iters[-1]))

    concat_norms = theta_iters_dist(kin_model_name, batch_size, theta_iters, iso_goal_input)

    model_dist = Model(inputs=[theta_input, iso_goal_input], outputs=concat_norms, name="residual_fk_dnn_dist")
    model_ik = Model(inputs=[theta_input, iso_goal_input], outputs=theta_iters, name="residual_fk_dnn_ik")
    return model_dist, model_ik
