import pysocialforce as psf
import numpy as np


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian: [px, py, vx, vy, gx, gy]
    initial_state = np.array(
        [
            [0.0, 0.0, 0.5, 0.5, 10.0, 10.0],
            [0.0, 1.0, 0.5, 0.5, 10.0, 10.0],
            [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],
            [11.0, 0.3, -0.5, 0.5, 0.0, 10.0],
            [12.0, 0.3, -0.5, 0.5, 0.0, 10.0],
        ]
    )
    # group informoation is represented as listes of indices of the members
    groups = [[0, 1], [2, 3, 4]]
    # initiate the simulator,
    s = psf.Simulator(
        initial_state, groups=groups, space=None, config_file="config.toml"
    )
    # update 80 steps
    states = np.stack([s.step().state.copy() for _ in range(80)])

    # plot
    with psf.show.canvas("images/exmaple.png") as ax:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        for i, group in enumerate(groups):
            for ped in group:
                x = states[:, ped, 0]
                y = states[:, ped, 1]
                ax.plot(x, y, "-o", label="ped {}".format(ped), markersize=2.5)
        ax.legend()
