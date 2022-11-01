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
from tf_kinematics.sys_solve import iter_grad_hess
from tf_kinematics.tf_transformations import tf_compact


class NewtonRecurrentBoost(tf.keras.Model):
    def __init__(self, kin_model_name: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel = load_kin(kin_model_name, batch_size)

        self.n_iters = n_iters
        self.iso_compact = IsometryCompact()

        self.fk_iso = ForwardKinematics(kin_model_name, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

        self.target_fn = tf.function(lambda theta: tf_compact(kernel.forward(tf.reshape(theta, [-1]))))
        self.loss_fn = tf.losses.mean_squared_error

    def call(self, inputs: (tf.Tensor, tf.Tensor), training=None, mask=None):
        theta_seed, iso_goal = inputs
        gamma_goal = self.iso_compact(iso_goal)
        for _ in range(self.n_iters):
            grad, hess = iter_grad_hess(gamma_goal, theta_seed, self.target_fn, self.loss_fn)
            delta = tf.matmul(tf.linalg.inv(hess), tf.expand_dims(grad, axis=2))
            theta_seed = theta_seed - tf.squeeze(delta)

        fk = self.fk_iso(theta_seed)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, gamma_goal])
        return ft_diff


def newton_recurrent_boost(kin_model_name: str, batch_size: int, n_iters: int) -> Model:
    assert n_iters > 0
    model = NewtonRecurrentBoost(kin_model_name, batch_size, n_iters, name='newton_recurrent_boost')

    dof = load_kin(kin_model_name, batch_size).dof
    model.build([(batch_size, dof), (batch_size, 4, 4)])
    return model
