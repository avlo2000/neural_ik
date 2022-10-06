from typing import Optional

import numpy as np
import tensorflow as tf
import keras.losses as losses
from keras.optimizers import Adam

from inference.fk_solver import FKSolver
from tf_kinematics.sys_solve import solve


class AdamSolver(FKSolver):
    def __init__(self, n_iters: int, loss_confidence: float, kin_model_ident: str):
        super().__init__(kin_model_ident)
        self.loss_confidence = loss_confidence
        self.n_iters = n_iters

        self.__opt = Adam()
        self.__loss_fn = losses.MeanSquaredError()
        self.__sys_fn = tf.function(lambda theta: self._kernel.forward(tf.reshape(theta, [-1])))

    def solve_ik(self, iso_goal: tf.Tensor, theta_seed: tf.Tensor) -> Optional[np.ndarray]:
        theta = solve(iso_goal, theta_seed, self.__sys_fn, self.n_iters, self.__loss_fn, self.__opt)
        theta_np = theta.numpy()

        fk_frame = self.solve_fk(theta_np)

        loss = self.__loss_fn(tf.squeeze(tf.convert_to_tensor(fk_frame.t_4_4)), iso_goal)
        if loss > self.loss_confidence:
            return None
        return theta_np
