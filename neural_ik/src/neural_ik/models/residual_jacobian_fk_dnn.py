import tensorflow as tf
from keras import layers
from keras import Model
from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.layers import ForwardKinematics, IsometryCompact, IsometryInverse


def residual_fk_dnn(kin: DLKinematics, blocks_count: int, corrector_units: int) -> (Model, layers.Layer):
    assert blocks_count > 0
    activation_fn = tf.nn.tanh

    theta_input = tf.keras.Input(kin.dof)
    gamma_input = tf.keras.Input((4, 4))
    gamma_inv = IsometryInverse()(gamma_input)

    def corrector_block(concat_layer: layers.Concatenate) -> layers.Layer:
        enc = layers.Dense(corrector_units, activation=activation_fn)(concat_layer)
        dec = layers.Dense(kin.dof, activation=activation_fn)(enc)
        return dec

    def fk_diff(theta: layers.Layer):
        fk = ForwardKinematics(kin)(theta)
        diff = layers.Multiply()([fk, gamma_inv])
        return diff

    def residual_block(theta_in: layers.Layer):
        diff = fk_diff(theta_in)
        gamma_diff_compact = IsometryCompact()(diff)

        concat = layers.Concatenate()([theta_in, gamma_diff_compact])
        theta_out = corrector_block(concat)
        return theta_out

    theta_res = residual_block(theta_input)
    for _ in range(blocks_count):
        theta_res = residual_block(theta_res)
    diff = fk_diff(theta_res)

    model = Model(inputs=[theta_input, gamma_input], outputs=diff, name="residual_fk_dnn")
    return model, theta_res
