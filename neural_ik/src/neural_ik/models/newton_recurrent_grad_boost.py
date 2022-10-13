import keras
import tensorflow as tf
from keras import Model
from keras import layers
from keras import regularizers

from neural_ik.layers import WeightedSum, Sum, GradOpt
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad


class GDRecurrentGradBoost(tf.keras.Model):
    def __init__(self, kin_model_name: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dof = load_kin(kin_model_name, batch_size).dof
        activation = tf.nn.relu

        self.n_iters = n_iters
        self.iso_compact = IsometryCompact()
        self.grad = SolveCompactIterGrad("mse", kin_model_name, batch_size)
        self.grad_gamma_and_seed = layers.Concatenate()

        self.lr_corrector = keras.Sequential([
            layers.Dense(32, activation=activation),
            layers.Dense(32, activation=activation),
            layers.Dense(dof, activation=activation)
        ], name='gradient_boost')

        self.inner_optimizer = GradOpt(name="final_ik")

        self.fk_iso = ForwardKinematics(kin_model_name, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

    def call(self, inputs: (tf.Tensor, tf.Tensor), training=None, mask=None):
        theta_seed, iso_goal = inputs
        gamma = self.iso_compact(iso_goal)
        for _ in range(self.n_iters):
            grad = self.grad([gamma, theta_seed])
            grad_gamma_and_seed = self.grad_gamma_and_seed([grad, gamma, theta_seed])
            lr = self.lr_corrector(grad_gamma_and_seed)
            theta_seed = self.inner_optimizer([grad, lr, theta_seed])

        fk = self.fk_iso(theta_seed)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, gamma])
        return ft_diff


def newton_recurrent_grad_boost(kin_model_name: str, batch_size: int, n_iters: int) -> (Model, Model):
    assert n_iters > 0
    model = GDRecurrentGradBoost(kin_model_name, batch_size, n_iters, name='newton_rnn_grad_boost')

    dof = load_kin(kin_model_name, batch_size).dof
    model.build([(batch_size, dof), (batch_size, 4, 4)])
    return model
