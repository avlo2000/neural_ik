import matplotlib.pyplot as plt
import numpy as np

N = 50


def create_gravity(m, m_pos: np.ndarray):
    def gravity(pos: np.ndarray):
        dir = m_pos - pos
        n = np.linalg.norm(dir)
        if n == 0:
            return 0, 0
        magn = m / n ** 2
        return magn * np.flip(dir) / n
    return gravity


def add_fields(*fields):
    def res_field(pos: np.ndarray):
        return sum([f(pos) for f in fields])
    return res_field


def traj(field, steps, initial, eps=0.01):
    x = np.empty(steps, dtype=float)
    y = np.empty(steps, dtype=float)
    for i in range(steps):
        x[i] = initial[0]
        y[i] = initial[1]
        initial += field(initial) * eps
    return x, y


def main():
    G1 = create_gravity(0.1, np.asarray([1.0, 1.0]))
    G2 = create_gravity(0.3, np.asarray([0.0, 1.0]))
    G3 = create_gravity(1, np.asarray([-1.0, -1.0]))
    sys = add_fields(G1, G2, G3)

    border = 2
    X, Y = np.mgrid[-border:border:20j, -border:border:20j]

    st = np.stack([X, Y])
    U, V = np.apply_along_axis(sys, arr=st, axis=0)
    plt.quiver(U, V)

    # x, y = traj(sys, 3000, np.asarray([1, 1], dtype=float), 0.01)
    # plt.plot(x, y, color='blue')
    plt.show()


if __name__ == '__main__':
    main()
