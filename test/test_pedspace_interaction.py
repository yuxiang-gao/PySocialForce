import numpy as np
import pytest
import socialforce


def test_raB():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    space = [
        np.array([[0.0, 100.0], [0.0, 0.5]])
    ]
    s = socialforce.Simulator(initial_state, space)
    assert s.raB().tolist() == [
        [[0.0, -0.5]],
        [[1.0, -0.5]],
    ]
