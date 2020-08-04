import pytest
import numpy as np
import pysocialforce as psf
from pysocialforce.utils.plot import SceneVisualizer

OUTPUT_DIR = "images/"


def test_separator():
    initial_state = np.array([[-10.0, -0.0, 1.0, 0.0, 10.0, 0.0],])
    obstacles = [(-1, 4, -1, 4)]
    s = psf.Simulator(initial_state, obstacles=obstacles)
    s.step(80)

    with SceneVisualizer(s, OUTPUT_DIR + "separator") as sv:
        sv.animate()


def test_gate():
    initial_state = np.array(
        [
            [-9.0, -0.0, 1.0, 0.0, 10.0, 0.0],
            [-10.0, -1.5, 1.0, 0.0, 10.0, 0.0],
            [-10.0, -2.0, 1.0, 0.0, 10.0, 0.0],
            [-10.0, -2.5, 1.0, 0.0, 10.0, 0.0],
            [-10.0, -3.0, 1.0, 0.0, 10.0, 0.0],
            [10.0, 1.0, -1.0, 0.0, -10.0, 0.0],
            [10.0, 2.0, -1.0, 0.0, -10.0, 0.0],
            [10.0, 3.0, -1.0, 0.0, -10.0, 0.0],
            [10.0, 4.0, -1.0, 0.0, -10.0, 0.0],
            [10.0, 5.0, -1.0, 0.0, -10.0, 0.0],
        ]
    )
    obstacles = [(0, 0, -10, -1.0), (0, 0, 1.0, 10)]
    s = psf.Simulator(initial_state, obstacles=obstacles)
    s.step(100)
    with SceneVisualizer(s, OUTPUT_DIR + "gate") as sv:
        sv.animate()


@pytest.mark.parametrize("n", [30, 60])
def test_walkway(n):
    pos_left = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])
    pos_right = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])

    x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
    x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
    x_destination_left = 100.0 * np.ones((n, 1))
    x_destination_right = -100.0 * np.ones((n, 1))

    zeros = np.zeros((n, 1))

    state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
    state_right = np.concatenate(
        (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1
    )
    initial_state = np.concatenate((state_left, state_right))

    obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)]
    agent_colors = [(1, 0, 0)] * n + [(0, 0, 1)] * n
    s = psf.Simulator(initial_state, obstacles=obstacles)
    s.step(150)
    with SceneVisualizer(s, OUTPUT_DIR + f"walkway_{n}", agent_colors=agent_colors) as sv:
        sv.ax.set_xlim(-30, 30)
        sv.ax.set_ylim(-20, 20)
        sv.animate()
