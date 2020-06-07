from contextlib import contextmanager
import numpy as np
import pytest
import pysocialforce as psf

OUTPUT_DIR = "images/"


@contextmanager
def visualize(states, space, output_filename):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print("")
    with psf.show.animation(len(states), output_filename, writer="imagemagick") as context:
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


@pytest.mark.plot
def test_separator():
    initial_state = np.array([[-10.0, -0.0, 1.0, 0.0, 10.0, 0.0],])
    obstacles = [(-1, 4, -1, 4)]
    s = psf.Simulator(initial_state, obstacles=obstacles)
    s.step(80)
    states = s.get_states()

    # visualize
    with visualize(states, s.scene.obstacles, OUTPUT_DIR + "separator.gif") as ax:
        ax.set_xlim(-10, 10)


@pytest.mark.plot
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
    obstacles = [(0, 0, -10, -0.7), (0, 0, 0.7, 10)]
    s = psf.Simulator(initial_state, obstacles=obstacles)
    s.step(150)
    states = s.get_states()

    with visualize(states, s.scene.obstacles, OUTPUT_DIR + "gate.gif") as _:
        pass


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
    s = psf.Simulator(initial_state, obstacles=obstacles)
    states = []
    s.step(250)
    states = s.get_states()
    # for _ in range(250):
    #     state = s.step().peds.state
    #     # periodic boundary conditions
    #     # state[state[:, 0] > 25, 0] -= 50
    #     # state[state[:, 0] < -25, 0] += 50

    #     states.append(state.copy())
    # states = np.stack(states)

    with visualize(states, s.scene.obstacles, OUTPUT_DIR + "walkway_{}.gif".format(n)) as _:
        pass
