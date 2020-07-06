import numpy as np
import pysocialforce as psf


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of (px, py, vx, vy, gx, gy)
    initial_state = np.array(
        [
            [0.0, 10, -0.5, -0.5, 0.0, 0.0],
            [0.5, 10, -0.5, -0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
            # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            # [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
    )
    # social groups informoation is represented as lists of indices of the state array
    groups = [[1, 0], [2]]
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    obs = [[1, 2, 7, 8]]
    # obs = None
    # initiate the simulator,
    s = psf.Simulator(initial_state, groups=groups, obstacles=obs)
    # update 80 steps
    s.step(50)

    with psf.plot.SceneVisualizer(s, "images/exmaple") as sv:
        sv.animate()
        # sv.plot()
