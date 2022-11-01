from typing import Optional, Mapping

import keras
import numpy as np
import tensorflow as tf
import keras.losses as losses
from tf_kinematics.kinematic_models_io import load as load_kin
from keras.optimizers import Adam

from inference.tf_solver import TFSolver
from neural_ik.layers import AdamOpt
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad
from tf_kinematics.sys_solve import solve


class RMSPropModel(tf.keras.Model):
    def __init__(self, kin_model_name: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iters = n_iters
        self.iso_compact = IsometryCompact()
        self.grad = SolveCompactIterGrad('mse', kin_model_name, batch_size)

        self.fk_iso = ForwardKinematics(kin_model_name, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

        self.eps = 0.001

    def call(self, inputs: (tf.Tensor, tf.Tensor), training=None, mask=None):
        theta_seed, iso_goal = inputs
        gamma = self.iso_compact(iso_goal)

        ts = 1.0
        lr = 0.01
        beta1 = 0.9

        vel = tf.zeros_like(theta_seed)
        for _ in range(self.n_iters):
            grad = self.grad([gamma, theta_seed])
            vel = (1.0 - beta1) * vel + beta1 * tf.square(grad)
            theta_seed = theta_seed - tf.math.divide_no_nan(lr * grad, tf.sqrt(vel + self.eps))
            ts += 1

        fk = self.fk_iso(theta_seed)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, gamma])
        return ft_diff


class AdamSolver(TFSolver):
    def __init__(self, n_iters: int, loss_confidence: float, kin_model_ident: str, *args, **kwargs):
        super().__init__(kin_model_ident, *args, **kwargs)
        self.loss_confidence = loss_confidence
        self.n_iters = n_iters

        self.__opt = Adam()
        self.__loss_fn = losses.MeanSquaredError()
        self.__sys_fn = tf.function(lambda theta: self._kernel.forward(tf.reshape(theta, [-1])))

    def solve_ik(self, iso_goal: tf.Tensor, theta_seed: tf.Tensor) -> (Optional[np.ndarray], Mapping[str, float]):
        theta = solve(iso_goal, theta_seed, self.__sys_fn, self.n_iters, self.__loss_fn, self.__opt)
        theta_np = theta.numpy()

        fk_frame = self.solve_fk(theta_np)

        metric_vals = dict()
        for name, metric in self._metrics_map:
            metric_vals[name] = metric(tf.squeeze(tf.convert_to_tensor(fk_frame.t_4_4)), iso_goal)

        return theta_np, metric_vals
