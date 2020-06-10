import numpy as np
import pytest

from pysocialforce import forces
from pysocialforce.scene import Scene, PedState
from pysocialforce.utils import DefaultConfig


@pytest.fixture()
def generate_scene():
    state = np.zeros((5, 7))
    state[:, :2] = np.array([[1, 1], [1, 1.1], [3, 3], [3, 3.01], [3, 4]])
    config = DefaultConfig()
    scene = Scene(state, config=config)
    return scene, config


def test_group_rep_force(generate_scene):
    scene, config = generate_scene
    scene.peds.groups = [[1, 0], [3, 2]]
    f = forces.GroupRepulsiveForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force() == pytest.approx(
        -np.array([[0.0, -0.1], [0.0, 0.1], [0.0, -0.01], [0.0, 0.01], [0.0, 0.0]])
    )


def test_group_coherence_force(generate_scene):
    scene, config = generate_scene
    scene.peds.groups = [[0, 1, 3], [2, 4]]
    f = forces.GroupCoherenceForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force() == pytest.approx(
        np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-0.71421284, -0.69992858], [0.0, -1.0],])
    )


def test_group_gaze_force(generate_scene):
    scene, config = generate_scene
    scene.peds.groups = [[0, 1, 3], [2, 4]]
    f = forces.GroupGazeForce()
    f.init(scene, config)
    f.factor = 1.0
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
