import numpy as np
import pysocialforce as psf
from pysocialforce.utils.plot import SceneVisualizer

OUTPUT_DIR = "images/"


def test_crossing():
    initial_state = np.array([[0.0, 0.0, 0.5, 0.5, 10.0, 10.0], [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],])
    s = psf.Simulator(initial_state)
    s.step(50)

    with SceneVisualizer(s.scene, OUTPUT_DIR + "crossing") as sv:
        sv.plot()


def test_narrow_crossing():
    initial_state = np.array([[0.0, 0.0, 0.5, 0.5, 2.0, 10.0], [2.0, 0.3, -0.5, 0.5, 0.0, 10.0],])
    s = psf.Simulator(initial_state)
    s.step(40)
    with SceneVisualizer(s.scene, OUTPUT_DIR + "narrow_crossing") as sv:
        sv.plot()


def test_opposing():
    initial_state = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 10.0], [-0.3, 10.0, -1.0, 0.0, -0.3, 0.0],])
    s = psf.Simulator(initial_state)
    s.step(21)
    with SceneVisualizer(s.scene, OUTPUT_DIR + "opposing") as sv:
        sv.plot()


def test_2opposing():
    initial_state = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 0.0, 10.0],
            [0.6, 10.0, -0.5, 0.0, 0.6, 0.0],
            [2.0, 10.0, -0.5, 0.0, 2.0, 0.0],
        ]
    )
    s = psf.Simulator(initial_state)
    s.step(40)
    with SceneVisualizer(s.scene, OUTPUT_DIR + "2opposing") as sv:
        sv.plot()
