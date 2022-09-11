from abc import ABCMeta
from abc import abstractmethod
from visual_kinematics.RobotSerial import RobotSerial
from visual_kinematics.RobotTrajectory import Frame

import tensorflow as tf
import numpy as np


class FKGenerator(tf.keras.utils.Sequence):
    __metaclass__ = ABCMeta

    def __init__(self, batch_size: int, n: int, robot: RobotSerial, ws_lim=None, shuffle=True):
        self.batch_size = batch_size
        self._n = n
        self._dof = len(robot.dh_params)
        self._robot = robot
        self._x: np.ndarray = None
        self._y: np.ndarray = None
        self._ws_lim = ws_lim if ws_lim is not None else self._robot.ws_lim
        self._generate()
        self._shuffle = shuffle
        self.__key_array = np.arange(self._n, dtype=np.int32)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self._shuffle:
            self.__key_array = np.random.permutation(self.__key_array)

    def __getitem__(self, index):
        keys = self.__key_array[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.asarray(self._x[keys], dtype=np.float32)
        y = np.asarray(self._y[keys], dtype=np.float32)
        return x, y

    def __len__(self):
        return self._n // self.batch_size

    @abstractmethod
    def _generate(self):
        pass

    @staticmethod
    def _frame_to_vec(cart_iso: Frame) -> np.array:
        r = cart_iso.r_3
        tr = cart_iso.t_3_1
        return np.append(tr, r)

    @staticmethod
    def _vec_to_frame(vec: np.array) -> Frame:
        return Frame.from_r_3(vec[:3], np.array([[t] for t in vec[3:]]))

    @property
    def frames(self) -> list:
        return [self._vec_to_frame(vec) for vec in self._x]

    @property
    def input_dim(self) -> int:
        return self._x.shape[1]

    @property
    def output_dim(self) -> int:
        return self._y.shape[1]

    @property
    def dof(self):
        return self._dof
