from typing import Optional, Mapping

import keras
import numpy as np
import tensorflow as tf
import keras.losses as losses
from keras.optimizers import SGD

from inference.tf_solver import TFSolver
from neural_ik.layers import GradOpt
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.layers.solve_layers import SolveCompactIterGrad
from tf_kinematics.sys_solve import solve


class GDModel(keras.Model):
    def __init__(self, kin_model_ident: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iters = n_iters
        self.kin_model_ident = kin_model_ident

        self.iso_compact = IsometryCompact()
        self.grad = SolveCompactIterGrad("mse", kin_model_ident, batch_size)
        self.inner_optimizer = GradOpt(name="final_ik")

        self.fk_iso = ForwardKinematics(kin_model_ident, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

    def call(self, inputs, training=None, mask=None):
        theta_seed, iso_goal = inputs
        gamma = self.iso_compact(iso_goal)
        for _ in range(self.n_iters):
            grad = self.grad([gamma, theta_seed])
            theta_seed = self.inner_optimizer([grad, 0.01, theta_seed])

        fk = self.fk_iso(theta_seed)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, gamma])
        return ft_diff


class NewtonSolver(TFSolver):
    def __init__(self, n_iters: int, loss_confidence: float, kin_model_ident: str, *args, **kwargs):
        super().__init__(kin_model_ident, *args, **kwargs)
        self.loss_confidence = loss_confidence
        self.n_iters = n_iters

        self.__opt = SGD()
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
