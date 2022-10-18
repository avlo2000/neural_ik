import tensorflow as tf
from keras import layers

from tf_kinematics.layers.iso_layers import IsometryInverse, IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics, LimitsLerp
from tf_kinematics.kinematic_models_io import load as load_kin


def simple_dnn(kin_model_name: str, batch_size: int) -> tf.keras.Model:
    dof = load_kin(kin_model_name, batch_size).dof

    theta_input = tf.keras.Input(dof)
    iso_goal = tf.keras.Input((4, 4))
    iso_goal_compact = IsometryCompact()(iso_goal)

    activation_fn = tf.nn.tanh

    x = IsometryCompact()(iso_goal)
    x = layers.Concatenate()([x, theta_input])

    x = layers.Dense(128, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(256, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(512, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(1028, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(1028, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(512, activation=activation_fn)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation=activation_fn)(x)
    x = layers.Dropout(0.2)(x)

    theta_out = layers.Dense(dof, activation=tf.nn.tanh)(x)
    theta_out = LimitsLerp(-1.0, 1.0, kin_model_name, batch_size)(theta_out)

    fk_iso = ForwardKinematics(kin_model_name, batch_size)(theta_out)
    fk_compact = IsometryCompact()(fk_iso)
    fk_diff = Diff()([fk_compact, iso_goal_compact])

    model = tf.keras.Model(inputs=[theta_input, iso_goal], outputs=fk_diff, name="simple_dnn")
    return model
