import tensorflow as tf
from keras import Model
from keras import layers
from keras.engine.keras_tensor import KerasTensor

from neural_ik.layers import WeightedSum, Sum, GradOpt
from neural_ik.losses import CompactL2L2
from neural_ik.models.common import dnn_block, decorate_model_in_out
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad


@decorate_model_in_out
def newton_dnn_grad_boost(kin_model_name: str, batch_size: int, blocks_count: int) -> (Model, Model):
    assert blocks_count > 0

    dof = load_kin(kin_model_name, batch_size).dof
    theta_seed_input = tf.keras.Input(dof)
    iso_goal_input = tf.keras.Input((4, 4))
    iso_goal_compact = IsometryCompact()(iso_goal_input)

    def residual_block(theta_in: KerasTensor, name=None):
        grad = SolveCompactIterGrad(CompactL2L2(1.0, 10.0), kin_model_name, batch_size)([iso_goal_compact, theta_in])
        theta_dtheta = layers.Concatenate()([iso_goal_compact, grad, theta_in])
        smart_lr = dnn_block(dof, (64, 64, 16), theta_dtheta)

        if name is None:
            return GradOpt()([grad, smart_lr, theta_in])
        return GradOpt(name=name)([grad, smart_lr, theta_in])

    theta_iter = residual_block(theta_seed_input)
    for i in range(blocks_count - 2):
        theta_iter = residual_block(theta_iter)
    theta_iter = residual_block(theta_iter, "final_ik")

    fk_iso = ForwardKinematics(kin_model_name, batch_size)(theta_iter)
    fk_compact = IsometryCompact()(fk_iso)
    fk_diff = Diff()([fk_compact, iso_goal_compact])

    return [theta_seed_input, iso_goal_input], fk_diff
