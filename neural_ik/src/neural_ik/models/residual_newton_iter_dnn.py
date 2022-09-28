import tensorflow as tf
from keras import layers
from keras import Model
from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.layers import ForwardKinematics, IsometryCompact, IsometryInverse, NewtonIter


def residual_newton_iter_dnn(kin: DLKinematics, blocks_count: int, corrector_units: int) -> (Model, layers.Layer):
    assert blocks_count > 0
    activation_fn = tf.nn.tanh

    theta_input = tf.keras.Input(kin.dof)
    gamma_input = tf.keras.Input((4, 4))
    gamma_compact = IsometryCompact()(gamma_input)

    def corrector_block(in_layer: layers.Layer, idx: int) -> layers.Layer:
        enc = layers.Dense(corrector_units, activation=activation_fn, name=f'encoder_{idx}')(in_layer)
        dec = layers.Dense(kin.dof, activation=activation_fn, name=f'decoder_{idx}')(enc)
        return dec

    def residual_block(theta_in: layers.Layer, idx: int):
        newton = NewtonIter(kin, learning_rate=0.1, return_diff=False, name=f'newton_iter_{idx}') \
            ([gamma_compact, theta_in])
        theta_out = corrector_block(newton, idx)
        return theta_out

    theta_res = residual_block(theta_input, 0)
    for i in range(blocks_count):
        theta_res = residual_block(theta_res, i + 1)

    gamma_inv = IsometryInverse(name='gamma_inv')(gamma_input)
    fk = ForwardKinematics(kin, name='final_fk')(theta_res)
    diff = layers.Multiply(name='final_diff')([fk, gamma_inv])

    model = Model(inputs=[theta_input, gamma_input], outputs=diff, name="residual_newton_iter_dnn")
    return model, theta_res
