import numpy as np
import socialforce


def test_group_crossing():
    import matplotlib.cm as cm

    initial_state = np.array(
        [
            [0.0, 0.0, 0.5, 0.5, 10.0, 10.0],
            [0.0, 1.0, 0.5, 0.5, 10.0, 10.0],
            [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],
            [11.0, 0.3, -0.5, 0.5, 0.0, 10.0],
            [12.0, 0.3, -0.5, 0.5, 0.0, 10.0],
        ]
    )
    groups = [[0, 1], [2, 3, 4]]
    colors = cm.rainbow(np.linspace(0, 1, len(groups)))
    s = socialforce.Simulator(initial_state, groups=groups)
    states = np.stack([s.step().state.copy() for _ in range(80)])

    with socialforce.show.canvas("docs/group_crossing.png") as ax:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        for i, group in enumerate(groups):
            color = colors[i]
            for ped in group:
                x = states[:, ped, 0]
                y = states[:, ped, 1]
                ax.plot(
                    x, y, "-o", label="ped {}".format(ped), markersize=2.5, color=color
                )
        ax.legend()
