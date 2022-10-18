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
    def __init__(self, n_iters: int, kin_model_ident: str, batch_size: int, *args, **kwargs):
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
        tfp.math.secant_root