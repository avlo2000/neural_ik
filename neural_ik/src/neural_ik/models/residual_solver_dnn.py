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
    theta_seed_input = tf.keras.Input(dof)
    iso_goal_input = tf.keras.Input((4, 4))
    iso_goal_compact = IsometryCompact()(iso_goal_input)

    def residual_block(theta_in: KerasTensor, name=None):
        grad = SolveIterGrad("mse", kin_model_name, batch_size)([iso_goal_compact, theta_in])
        smart_lr = dnn_block(dof, (16, 32, 16), theta_in)

        if name is None:
            return WeightedSum()([grad, smart_lr, theta_in])
        return WeightedSum(name=name)([grad, smart_lr, theta_in])

    theta_iter = residual_block(theta_seed_input)
    for i in range(blocks_count - 2):
        theta_iter = residual_block(theta_iter)
    theta_iter = residual_block(theta_iter, "final")

    fk_iso = ForwardKinematics(kin_model_name, batch_size)(theta_iter)
    fk_compact = IsometryCompact()(fk_iso)
    fk_diff = CompactDiff()([fk_compact, iso_goal_compact])

    model = Model(inputs=[theta_seed_input, iso_goal_input], outputs=fk_diff, name="residual_solver_dnn_dist")
    return model
