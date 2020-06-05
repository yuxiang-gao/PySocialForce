import numpy as np
import pysocialforce
from contextlib import contextmanager


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
    s = pysocialforce.Simulator(initial_state, groups=groups)
    states = np.stack([s.step().state.copy() for _ in range(80)])

    with pysocialforce.show.canvas("docs/group_crossing.png") as ax:
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


@contextmanager
def visualize(states, space, output_filename):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print("")
    with pysocialforce.show.animation(
        len(states), output_filename, writer="imagemagick"
    ) as context:
        ax = context["ax"]
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        yield ax

        for s in space:
            ax.plot(s[:, 0], s[:, 1], "-o", color="black", markersize=2.5)

        actors = []
        for ped in range(states.shape[1]):
            speed = np.linalg.norm(states[0, ped, 2:4])
            radius = 0.2 + speed / 2.0 * 0.3
            p = plt.Circle(
                states[0, ped, 0:2],
                radius=radius,
                facecolor="black" if states[0, ped, 4] > 0 else "white",
                edgecolor="black",
            )
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                p.center = states[i, ped, 0:2]
                speed = np.linalg.norm(states[i, ped, 2:4])
                p.set_radius(0.2 + speed / 2.0 * 0.3)

        context["update_function"] = update
