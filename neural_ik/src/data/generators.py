import numpy as np
from tqdm import tqdm

from data.abstract_generator import FKGenerator
from data.dataset import frame_to_vec


class RandomGen(FKGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate(self):
        self._x = []
        self._y = []

        mn = self._ws_lim[:, 0]
        mx = self._ws_lim[:, 1]
        for _ in tqdm(range(self._n)):
            joint = np.random.rand(self._dof) * (mx - mn) + mn
            cart_iso = self._robot.forward(joint)
            cart = frame_to_vec(cart_iso)

            self._x.append(cart)
            self._y.append(joint)
        self._x = np.stack(self._x)
        self._y = np.stack(self._y)


class JakGen(FKGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate(self):
        self._x = []
        self._y = []

        mn = self._ws_lim[:, 0]
        mx = self._ws_lim[:, 1]
        priors = self._calc_joints_priorities()
        for _ in tqdm(range(self._n)):
            joint = np.random.rand(self._dof) * priors * (mx - mn) + mn
            cart_iso = self._robot.forward(joint)
            flat_cart = frame_to_vec(cart_iso)

            self._x.append(self._append_jac(flat_cart))
            self._y.append(joint)
        self._x = np.stack(self._x)
        self._y = np.stack(self._y)

    def _calc_joints_priorities(self):
        j_dists = self._robot.dh_params[:, 0]
        priors = np.cumsum(j_dists[::-1])
        return priors / np.linalg.norm(priors)

    def _append_jac(self, flat_cart):
        flat_jac = self._robot.jacobian.flatten()
        return np.append(flat_cart, flat_jac)


class TrjGen(FKGenerator):

    def __init__(self, trj_count: int, trj_size: int, step: float, *args, **kwargs):
        self.step = step
        self.trj_size = trj_size
        self.trj_count = trj_count
        self.__x_prev = []
        super().__init__(n=trj_count * trj_size, batch_size=trj_size, *args, **kwargs)

    def _generate(self):
        self.shuffle = False

        self._x = []
        self.__x_prev = []
        self._y = []
        self.batch_size = self.trj_size

        mn = self._ws_lim[:, 0]
        mx = self._ws_lim[:, 1]
        for _ in tqdm(range(self.trj_count)):
            from_j = np.random.rand(self._dof) * (mx - mn) + mn
            to_j = np.random.rand(self._dof) * (mx - mn) + mn
            x, x_prev, y = self._interpolate(from_j, to_j, self.trj_size)
            self._x.extend(x)
            self.__x_prev.extend(x_prev)
            self._y.extend(y)

        self._x = np.stack(self._x)
        self.__x_prev = np.stack(self.__x_prev)
        self._y = np.stack(self._y)

    def _interpolate(self, from_j: np.ndarray, to_j: np.ndarray, size: int):
        dir_j = to_j - from_j
        dir_j = dir_j * self.step / np.linalg.norm(dir_j)
        j_inter = np.linspace(from_j, from_j + dir_j, size)

        x = []
        x_prev = []
        y = []
        prev = from_j - np.random.uniform(low=-self.step, high=self.step, size=self._dof)
        for joint in j_inter:
            cart_iso = self._robot.forward(joint)
            flat_cart = frame_to_vec(cart_iso)

            x.append(flat_cart)
            x_prev.append(prev)
            y.append(joint)
            prev = joint

        return x, x_prev, y

    def __getitem__(self, index):
        i = self.batch_size * index
        x = self._x[i:i + self.batch_size, :]
        x_prev = self.__x_prev[i:i + self.batch_size, :]
        y = self._y[i:i + self.batch_size, :]
        assert len(y) == len(x_prev) == len(x) == self.batch_size
        return [x, x_prev], y
