from unittest import TestCase

import numpy as np
from visual_kinematics import Frame

from data.data_io import frame_to_vec, vec_to_frame


class TestEvaluate(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 0.0001

    def test_frame_to_vec_transitivity(self):
        frame = Frame.from_r_3(np.random.rand(3), np.random.rand(3, 1))
        vec = frame_to_vec(frame)
        frame_back = vec_to_frame(vec)

        self.assertTrue(frame.distance_to(frame_back) <= self.eps)

    def test_vec_to_frame_transitivity(self):
        vec = np.random.rand(7)
        frame = vec_to_frame(vec)
        vec_back = frame_to_vec(frame)

        self.assertTrue(np.max(vec - vec_back) <= self.eps)
