import tensorflow as tf
from keras import layers
from keras import Model

from neural_ik.models.common import fk_theta_iters_dist
from tf_kinematics.kinematic_models import load as load_kin
from tf_kinematics.kin_layers import ForwardKinematics, JacobianForwardKinematics
from tf_kinematics.iso_layers import IsometryCompact, IsometryInverse


def residual_jacobian_fk_dnn(kin_model_name: str, batch_size: int, *, blocks_count: int, corrector_units: int) -> (Model, Model):
    assert blocks_count > 0
    activation_fn = tf.nn.leaky_relu

    dof = load_kin(kin_model_name, batch_size).dof
    theta_input = tf.keras.Input(dof)
    iso_input = tf.keras.Input((4, 4))
    iso_inv = IsometryInverse()(iso_input)

    def corrector_block(concat_layer: layers.Concatenate) -> layers.Layer:
        enc = layers.Dense(corrector_units, activation=activation_fn)(concat_layer)
        dec = layers.Dense(dof, activation=activation_fn)(enc)
        return dec

    def residual_block(theta_in: layers.Layer):
        jac_fk = JacobianForwardKinematics(kin_model_name, batch_size)(theta_in)
        flat = layers.Flatten()(jac_fk)

        concat = layers.Concatenate()([theta_in, flat])
        theta_out = corrector_block(concat)
        return theta_out

    theta_iters = [residual_block(theta_input)]
    for i in range(blocks_count - 1):
        theta_iters.append(residual_block(theta_iters[-1]))

    concat_norms = fk_theta_iters_dist(kin_model_name, batch_size, theta_iters, iso_inv)

    model_dist = Model(inputs=[theta_input, iso_input], outputs=concat_norms, name="residual_jacobian_fk_dnn_dist")
    model_ik = Model(inputs=[theta_input, iso_input], outputs=theta_iters, name="residual_jacobian_fk_dnn_ik")
    return model_dist, model_ik
