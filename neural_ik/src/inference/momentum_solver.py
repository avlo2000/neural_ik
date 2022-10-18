import keras
import tensorflow as tf
from keras import Model
from keras import layers

from neural_ik.layers import MomentumOpt
from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad


class MomentumModel(tf.keras.Model):
    def __init__(self, kin_model_name: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_iters = n_iters
        self.iso_compact = IsometryCompact()
        self.grad = SolveCompactIterGrad(tf.losses.mean_squared_error, kin_model_name, batch_size)
        self.grad_gamma_and_seed = layers.Concatenate()

        self.fk_iso = ForwardKinematics(kin_model_name, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

    def call(self, inputs: (tf.Tensor, tf.Tensor), training=None, mask=None):
        theta_seed, iso_goal = inputs
        gamma = self.iso_compact(iso_goal)

        beta = 0.9
        lr = 0.01
        momentum = tf.zeros_like(theta_seed)
        for _ in range(self.n_iters):
            grad = self.grad([gamma, theta_seed])
            momentum = (1.0 - beta) * momentum + beta * grad

            theta_seed = theta_seed - lr * momentum

        fk = self.fk_iso(theta_seed)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, gamma])
        return ft_diff

