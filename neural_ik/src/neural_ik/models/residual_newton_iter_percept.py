import tensorflow as tf
from keras import Model
from keras.engine.keras_tensor import KerasTensor

from neural_ik.layers import WeightedSum
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.iso_layers import IsometryCompact
from tf_kinematics.layers.kin_layers import NewtonIter
from neural_ik.models.common import fk_compact_iters_dist, dnn_block


def residual_newton_iter_percept(kin_model_name: str, batch_size: int, *, blocks_count: int) -> (Model, Model):
    assert blocks_count > 0

    dof = load_kin(kin_model_name, batch_size).dof
    theta_input = tf.keras.Input(dof)
    iso_goal_input = tf.keras.Input((4, 4))
    gamma_goal = IsometryCompact()(iso_goal_input)

    def residual_block(theta_iter: KerasTensor) -> (KerasTensor, KerasTensor):
        lr = dnn_block(dof, [32], theta_iter)
        d_theta, gamma_fk_iter = NewtonIter(kin_model_name, batch_size)([gamma_goal, theta_iter])
        theta_out = WeightedSum()([lr, d_theta, theta_iter])
        return theta_out, gamma_fk_iter

    theta, gamma_fk = residual_block(theta_input)
    theta_iters = [theta]
    fk_iters = [gamma_fk]
    for i in range(blocks_count - 1):
        theta, gamma_fk = residual_block(theta)
        theta_iters.append(theta)
        fk_iters.append(gamma_fk)

    concat_norms = fk_compact_iters_dist([fk_iters[-1]], iso_goal_input)

    model_dist = Model(inputs=[theta_input, iso_goal_input], outputs=concat_norms, name="residual_newton_iter_percept_dist")
    model_ik = Model(inputs=[theta_input, iso_goal_input], outputs=theta_iters, name="residual_newton_iter_percept_ik")

    return model_dist, model_ik
