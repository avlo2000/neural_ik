from unittest import TestCase
from neural_ik.ws_space import WsSpace

import numpy as np


class TestWsSpace(TestCase):
    def __init__(self, *args, **kwargs):
        self.dof = 8
        self.ws_lim = np.zeros((self.dof, 2), dtype=np.float32)
        self.ws_lim[:, 1] = [np.pi] * self.dof
        self.ws_lim[:, 0] = [-np.pi] * self.dof
        super().__init__(*args, **kwargs)

    def test_lin_split_returns_correct_shape(self):
        ws = WsSpace(self.ws_lim)
        n = 50
        ws_pars = ws.lin_split(n)
        self.assertEqual(ws_pars.shape, (n, self.dof, 2))
