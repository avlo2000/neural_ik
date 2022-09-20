import tensorflow as tf

from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.layers import ForwardKinematics, IsometryCompact, IsometryInverse


def fk_dnn(kin: DLKinematics) -> tf.keras.Model:
    input_layer = tf.keras.Input(kin.dof)
    x = ForwardKinematics(kin)(input_layer)
    x = IsometryInverse()(x)
    output_layer = IsometryCompact()(x)
    model = tf.keras.Model(input_layer, output_layer, name="fk_dnn")

    return model


def simple_dnn(input_dim: int, dof: int) -> tf.keras.Model:
    input_layer = tf.keras.Input(input_dim)

    x = tf.keras.layers.Dense(50, activation=tf.nn.relu)(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(200, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(50, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output_layer = tf.keras.layers.Dense(dof, activation=tf.nn.tanh)(x)

    model = tf.keras.Model(input_layer, output_layer, name="simple_dnn")
    return model


def converging_dnn(input_dim: int, dof: int) -> tf.keras.Model:
    cart_layer_input = tf.keras.Input(input_dim)
    prev_state_input = tf.keras.Input(dof)

    cart_dense = tf.keras.layers.Dense(100)(cart_layer_input)
    prev_state_dense = tf.keras.layers.Dense(100)(prev_state_input)
    concat = tf.keras.layers.Concatenate()([cart_dense, prev_state_dense])

    x = tf.keras.layers.Dense(200, activation=tf.nn.relu)(concat)
    x = tf.keras.layers.Concatenate()([x, prev_state_dense])

    x = tf.keras.layers.Dense(200, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Concatenate()([x, prev_state_dense])

    x = tf.keras.layers.Dense(200, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Concatenate()([x, prev_state_dense])

    output_layer = tf.keras.layers.Dense(dof, activation=tf.nn.relu)(x)

    model = tf.keras.Model(inputs=[cart_layer_input, prev_state_input], outputs=output_layer, name="converging_dnn")
    return model


def fk_dist_estimator(input_dim: int, dof: int) -> tf.keras.Model:
    cart_layer_input = tf.keras.Input(input_dim)
    q_input = tf.keras.Input(dof)


