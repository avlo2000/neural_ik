import tensorflow as tf
from keras import layers


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
