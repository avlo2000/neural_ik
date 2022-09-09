import numpy as np


class WsSpace:
    def __init__(self, ws_lim: np.ndarray):
        self.ws_lim = ws_lim
        self._dof = self.ws_lim.shape[0]

    def lin_split(self, cnt: int) -> np.ndarray:
        res = []
        for lim in self.ws_lim:
            lin_space = np.linspace(lim[0], lim[1], cnt + 1)
            segments = self._lin_space_to_segments(lin_space)
            res.append(segments)
        res = np.moveaxis(np.stack(res), [0, 1, 2], [1, 0, 2])
        return res

    @staticmethod
    def _lin_space_to_segments(lin_space: np.ndarray) -> np.ndarray:
        prev = lin_space[0]
        segments = []
        for x in lin_space[1:]:
            segments.append([prev, x])
            prev = x
        return np.array(segments)
