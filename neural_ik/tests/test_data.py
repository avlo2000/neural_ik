from unittest import TestCase

import numpy as np
from visual_kinematics import Frame

from data.data_io import frame_to_vec, vec_to_frame


class TestEvaluate(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 0.0001

    def test_frame_to_vec_transitivity(self):
        frame = Frame.from_r_3([0.30369251, 0.68088033, 0.98290186], [[0.56932648],
                                                                      [0.39061527],
                                                                      [0.0554877]])
        vec = frame_to_vec(frame)
        frame_back = vec_to_frame(vec)

        self.assertTrue(frame.distance_to(frame_back) <= self.eps)

    def test_vec_to_frame_transitivity(self):
        vec = np.asarray([0.15180924, 0.56683175, 0.34126947, 0.00432348, 0.6785767,  0.55012793, 0.76428717])
        frame = vec_to_frame(vec)
        vec_back = frame_to_vec(frame)

        self.assertTrue(np.max(vec - vec_back) <= self.eps)
