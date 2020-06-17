import numpy as np
import pysocialforce as psf


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian: [px, py, vx, vy, gx, gy]
    initial_state = np.array(
        [
            # [0.0, 0.0, 0.5, 0.5, 10.0, 10.0],
            # [0.0, 0.5, 0.5, 0.5, 10.0, 10.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 30.0],
            [1.0, 0.0, 0.0, 0.5, 2.0, 30.0],
            [2.0, 0.0, 0.0, 0.5, 3.0, 30.0],
            [3.0, 0.0, 0.0, 0.5, 4.0, 30.0],
        ]
    )
    # group informoation is represented as listes of indices of the members
    groups = [[0, 1, 2, 3]]
    # initiate the simulator,
    s = psf.Simulator(initial_state, groups=groups, obstacles=None)
    # update 80 steps
    s.step(150)

    with psf.plot.SceneVisualizer(s, "images/exmaple") as sv:
        sv.animate()
