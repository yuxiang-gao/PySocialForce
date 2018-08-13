import numpy as np
import socialforce


def test_raB():
    state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    space = [
        np.array([[0.0, 100.0], [0.0, 0.5]])
    ]
    raB = socialforce.potentials.PedSpacePotential(space).raB(state)
    assert raB.tolist() == [
        [[0.0, -0.5]],
        [[1.0, -0.5]],
    ]
