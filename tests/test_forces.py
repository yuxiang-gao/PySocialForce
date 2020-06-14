import numpy as np
import pytest

from pysocialforce import forces
from pysocialforce.scene import Scene, PedState
from pysocialforce.utils import DefaultConfig


@pytest.fixture()
def generate_scene():
    state = np.zeros((5, 7))
    state[:, :4] = np.array(
        [[1, 1, 1, 0], [1, 1.1, 0, 1], [3, 3, 1, 1], [3, 3.01, 1, 2], [3, 4, 3, 1]]
    )
    config = DefaultConfig()
    scene = Scene(state, config=config)
    return scene, config


def test_desired_force(generate_scene):
    scene, config = generate_scene
    f = forces.DesiredForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force(debug=True) == pytest.approx(
        np.array(
            [
                [-3.83847763, -1.83847763],
                [-1.74894926, -3.92384419],
                [-4.6, -4.6],
                [-6.10411508, -8.11779546],
                [-10.93315315, -8.57753753],
            ]
        )
    )


def test_social_force(generate_scene):
    scene, config = generate_scene
    f = forces.SocialForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force(debug=True) == pytest.approx(
        np.array(
            [
                [1.09406676e-04, 8.11547148e-05],
                [5.20013105e-05, 9.86718725e-05],
                [7.66236804e-03, -1.16738960e-04],
                [7.72339025e-03, 3.03769792e-05],
                [-9.12767677e-06, -1.23117582e-05],
            ]
        )
    )


def test_group_rep_force(generate_scene):
    scene, config = generate_scene
    scene.peds.groups = [[1, 0], [3, 2]]
    f = forces.GroupRepulsiveForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force(debug=True) == pytest.approx(
        np.array([[0.0, -0.1], [0.0, 0.1], [0.0, -0.01], [0.0, 0.01], [0.0, 0.0]])
    )


def test_group_coherence_force(generate_scene):
    scene, config = generate_scene
    scene.peds.groups = [[0, 1, 3], [2, 4]]
    f = forces.GroupCoherenceForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force(debug=True) == pytest.approx(
        np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-0.71421284, -0.69992858], [0.0, -1.0],])
    )


def test_group_gaze_force(generate_scene):
    scene, config = generate_scene
    scene.peds.groups = [[0, 1, 3], [2, 4]]
    f = forces.GroupGazeForce()
    f.init(scene, config)
    f.factor = 1.0
    assert f.get_force(debug=True) == pytest.approx(
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
