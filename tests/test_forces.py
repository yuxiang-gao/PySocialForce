import numpy as np
import pytest

import pysocialforce as psf


@pytest.fixture()
def generate_state():
    state = np.zeros((5, 7))
    state[:, :2] = np.array([[1, 1], [1, 1.1], [3, 3], [3, 3.01], [3, 4]])
    return state


def test_group_rep_force(generate_state):
    state = generate_state
    groups = [[1, 0], [3, 2]]
    f = psf.GroupRepulsiveForce()
    f.config = {"factor": 1.0, "threshold": 0.5}
    f.set_state(state, groups=groups)
    print(f.get_force())
    assert f.get_force() == pytest.approx(
        -np.array([[0.0, -0.1], [0.0, 0.1], [0.0, -0.01], [0.0, 0.01], [0.0, 0.0]])
    )


def test_group_coherence_force(generate_state):
    state = generate_state
    groups = [[0, 1, 3], [2, 4]]
    f = psf.GroupCoherenceForce()
    f.config = {"factor": 1.0}
    f.set_state(state, groups=groups)
    assert f.get_force() == pytest.approx(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-0.71421284, -0.69992858],
                [0.0, -1.0],
            ]
        )
    )


def test_group_gaze_force(generate_state):
    state = generate_state
    groups = [[0, 1, 3], [2, 4]]
    f = psf.GroupGazeForce()
    f.config = {"factor": 1.0}
    f.set_state(state, groups=groups)
    assert f.get_force() == pytest.approx(
        np.array(
            [
                [0.96838684, 0.96838684],
                [0.87370295, 0.96107324],
                [0.43194695, 0.43194695],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
    )
