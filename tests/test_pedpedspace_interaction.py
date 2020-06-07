import numpy as np

import pysocialforce as psf


def test_r_aB():
    state = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],])
    obstacles = [np.array([[0.0, 100.0], [0.0, 0.5]])]
    r_aB = psf.PedSpacePotential(obstacles).r_aB(state)
    assert r_aB.tolist() == [
        [[0.0, -0.5]],
        [[1.0, -0.5]],
    ]
