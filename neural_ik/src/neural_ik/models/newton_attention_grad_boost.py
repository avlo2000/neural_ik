import tensorflow as tf
from keras import layers
from keras import Model
from keras import losses
from keras.engine.keras_tensor import KerasTensor

from neural_ik.layers import Sum, WeightedSum
from neural_ik.models.common import theta_iters_dist, dnn_block, linear_identity, decorate_model_in_out
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad
from tf_kinematics.layers.iso_layers import IsometryCompact, CompactMSE, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics


class AttentionGradBoost(tf.keras.Model):
    def __init__(self, kin_model_name: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dof = load_kin(kin_model_name, batch_size).dof
        activation = tf.nn.relu

        self.n_iters = n_iters
        self.iso_compact = IsometryCompact()
        self.grad = SolveCompactIterGrad("mse", kin_model_name, batch_size)
        self.grad_and_seed = layers.Concatenate()
        self.lr_corrector = [
            layers.Dense(32, activation=activation, name='lr_corrector0'),
            layers.Dense(64, activation=activation, name='lr_corrector1'),
            layers.Dense(128, activation=activation, name='lr_corrector2'),
            layers.Dense(65, activation=activation, name='lr_corrector3'),
            layers.Dense(32, activation=activation, name='lr_corrector4'),
            layers.Dense(dof, activation=activation, name='lr_corrector5')
        ]

        self.theta_out = WeightedSum(name="final_ik")
        self.fk_iso = ForwardKinematics(kin_model_name, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

    def call(self, inputs, training=None, mask=None):
        theta_seed, iso_goal = inputs
        iso_goal_compact = self.iso_compact(iso_goal)
        for _ in range(self.n_iters):
            grad = self.grad([iso_goal_compact, theta_seed])
            lr = self.grad_and_seed([grad, theta_seed])
            for corr in self.lr_corrector:
                lr = corr(lr)
            theta_seed = self.theta_out([lr, grad, theta_seed])
        fk = self.fk_iso(theta_seed)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, iso_goal_compact])
        return ft_diff


def newton_rnn_grad_boost(kin_model_name: str, batch_size: int, n_iters: int) -> (Model, Model):
    assert n_iters > 0
    model = AttentionGradBoost(kin_model_name, batch_size, n_iters, name='newton_rnn_grad_boost')

    dof = load_kin(kin_model_name, batch_size).dof
    model.build([(batch_size, dof), (batch_size, 4, 4)])
    return model
