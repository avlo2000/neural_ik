import tensorflow as tf
from keras import layers
from keras import Model
from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.layers import ForwardKinematics, IsometryCompact, IsometryInverse


def residual_fk_dnn(kin: DLKinematics) -> (Model, layers.Layer):
    theta_input = tf.keras.Input(kin.dof)
    gamma_input = tf.keras.Input((4, 4))
    gamma_inv = IsometryInverse()(gamma_input)

    def corrector_block(concat_layer: layers.Concatenate) -> layers.Layer:
        enc = layers.Dense(10, activation=tf.nn.tanh)(concat_layer)
        dec = layers.Dense(kin.dof, activation=tf.nn.tanh)(enc)
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
    theta_res = residual_block(theta_res)
    theta_res = residual_block(theta_res)
    theta_res = residual_block(theta_res)
    theta_res = residual_block(theta_res)
    diff = fk_diff(theta_res)

    model = Model(inputs=[theta_input, gamma_input], outputs=diff, name="residual_fk_dnn")
    return model, theta_res


def simple_dnn(input_dim: int, dof: int) -> tf.keras.Model:
    input_layer = tf.keras.Input(input_dim)

    x = layers.Dense(50, activation=tf.nn.relu)(input_layer)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(100, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(200, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(100, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(50, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)

    output_layer = layers.Dense(dof, activation=tf.nn.tanh)(x)

    model = tf.keras.Model(input_layer, output_layer, name="simple_dnn")
    return model


def converging_dnn(input_dim: int, dof: int) -> tf.keras.Model:
    cart_layer_input = tf.keras.Input(input_dim)
    prev_state_input = tf.keras.Input(dof)

    cart_dense = layers.Dense(100)(cart_layer_input)
    prev_state_dense = layers.Dense(100)(prev_state_input)
    concat = layers.Concatenate()([cart_dense, prev_state_dense])

    x = layers.Dense(200, activation=tf.nn.relu)(concat)
    x = layers.Concatenate()([x, prev_state_dense])

    x = layers.Dense(200, activation=tf.nn.relu)(x)
    x = layers.Concatenate()([x, prev_state_dense])

    x = layers.Dense(200, activation=tf.nn.relu)(x)
    x = layers.Concatenate()([x, prev_state_dense])

    output_layer = layers.Dense(dof, activation=tf.nn.relu)(x)

    model = tf.keras.Model(inputs=[cart_layer_input, prev_state_input], outputs=output_layer, name="converging_dnn")
    return model



