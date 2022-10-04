import tensorflow as tf
from keras import layers
from keras import Model
from keras import losses
from keras.engine.keras_tensor import KerasTensor

from neural_ik.layers import Sum, WeightedSum
from neural_ik.models.common import theta_iters_dist, dnn_block, linear_identity
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.solve_layers import SolveIterGrad
from tf_kinematics.layers.iso_layers import IsometryCompact, CompactMSE, CompactDiff
from tf_kinematics.layers.kin_layers import ForwardKinematics


def residual_solver_dnn(kin_model_name: str, batch_size: int, blocks_count: int) -> (Model, Model):
    assert blocks_count > 0

    dof = load_kin(kin_model_name, batch_size).dof
    theta_input = tf.keras.Input(dof)
    iso_goal_input = tf.keras.Input((4, 4))
    iso_goal_compact = IsometryCompact()(iso_goal_input)

    def residual_block(theta_in: KerasTensor):
        grad = SolveIterGrad("mse", kin_model_name, batch_size)([iso_goal_compact, theta_in])
        smart_lr = dnn_block(dof, (16, 32, 16), theta_in)
        return WeightedSum()([grad, smart_lr, theta_in])

    theta_iter = residual_block(theta_input)
    for i in range(blocks_count - 1):
        theta_iter = residual_block(theta_iter)

    fk_iso = ForwardKinematics(kin_model_name, batch_size)(theta_iter)
    fk_compact = IsometryCompact()(fk_iso)
    fk_diff = CompactDiff()([fk_compact, iso_goal_compact])

    model_dist = Model(inputs=[theta_input, iso_goal_input], outputs=fk_diff, name="residual_solver_dnn_dist")
    model_ik = Model(inputs=[theta_input, iso_goal_input], outputs=theta_iter, name="residual_solver_dnn_ik")
    return model_dist, model_ik
