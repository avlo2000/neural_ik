import tensorflow as tf
from keras import layers
from keras import Model
from keras.engine.keras_tensor import KerasTensor

from neural_ik.models.common import theta_iters_dist, dnn_block
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.solve_layers import SolveIterGrad
from tf_kinematics.layers.iso_layers import IsometryCompact


def residual_solver_dnn(kin_model_name: str, batch_size: int, blocks_count: int) -> (Model, Model):
    assert blocks_count > 0

    dof = load_kin(kin_model_name, batch_size).dof
    theta_input = tf.keras.Input(dof)
    iso_goal_input = tf.keras.Input((4, 4))
    iso_goal_compact = IsometryCompact()(iso_goal_input)

    def residual_block(theta_in: KerasTensor):
        solve_iter = SolveIterGrad("mse", "rmsprop", kin_model_name, batch_size)([iso_goal_compact, theta_in])
        theta_out = dnn_block(dof, (16, 32), solve_iter)
        return theta_out

    theta_iters = [residual_block(theta_input)]
    for i in range(blocks_count - 1):
        theta_iters.append(residual_block(theta_iters[-1]))

    concat_norms = theta_iters_dist(kin_model_name, batch_size, theta_iters, iso_goal_input)

    model_dist = Model(inputs=[theta_input, iso_goal_input], outputs=concat_norms, name="residual_solver_dnn_dist")
    model_ik = Model(inputs=[theta_input, iso_goal_input], outputs=theta_iters, name="residual_solver_dnn_ik")
    return model_dist, model_ik
