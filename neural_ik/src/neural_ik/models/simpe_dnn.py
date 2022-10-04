import tensorflow as tf
from keras import layers

from tf_kinematics.layers.iso_layers import IsometryInverse, IsometryCompact
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.kinematic_models_io import load as load_kin


def simple_dnn(kin_model_name: str, batch_size: int) -> tf.keras.Model:
    dof = load_kin(kin_model_name, batch_size).dof
    theta_input = tf.keras.Input(dof)
    iso_input = tf.keras.Input((4, 4))

    activation_fn = tf.nn.relu

    x = IsometryCompact()(iso_input)
    x = layers.Concatenate()([x, theta_input])

    x = layers.Dense(64, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation=activation_fn)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation=activation_fn)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation=activation_fn)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    theta_out = layers.Dense(dof, activation=activation_fn)(x)
    iso_out = ForwardKinematics(kin_model_name, batch_size)(theta_out)
    iso_inv = IsometryInverse()(iso_input)
    iso_diff = layers.Multiply() //ELEMENTWISE!!!!!!!!!!

    model = tf.keras.Model(inputs=[theta_input, iso_input], output_layer, name="simple_dnn")
    return model
