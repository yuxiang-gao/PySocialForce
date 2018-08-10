import numpy as np
import socialforce


def test_gate():
    initial_state = np.array([
        [-9.0, -0.0, 1.0, 0.0],
        [-10.0, -1.5, 1.0, 0.0],
        [-10.0, -2.0, 1.0, 0.0],
        [-10.0, -2.5, 1.0, 0.0],
        [-10.0, -3.0, 1.0, 0.0],
        [10.0, 1.0, -1.0, 0.0],
        [10.0, 2.0, -1.0, 0.0],
        [10.0, 3.0, -1.0, 0.0],
        [10.0, 4.0, -1.0, 0.0],
        [10.0, 5.0, -1.0, 0.0],
    ])
    destinations = np.array([
        [10.0, 0.0],
        [10.0, 0.0],
        [10.0, 0.0],
        [10.0, 0.0],
        [10.0, 0.0],
        [-10.0, 0.0],
        [-10.0, 0.0],
        [-10.0, 0.0],
        [-10.0, 0.0],
        [-10.0, 0.0],
    ])
    space = [
        np.array([(0.0, y) for y in np.linspace(-10, -0.7)]),
        np.array([(0.0, y) for y in np.linspace(0.7, 10)]),
    ]
    s = socialforce.Simulator(initial_state, destinations, space)
    states = np.stack([s.step().state.copy() for _ in range(150)])

    # visualize
    print('')
    with socialforce.show.canvas('docs/gate.png') as ax:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for ped in range(initial_state.shape[0]):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)

        for s in space:
            ax.plot(s[:, 0], s[:, 1], color='black')

        ax.legend()

    with socialforce.show.animation(
            len(states) - 5,
            'docs/gate.gif',
            writer='imagemagick') as (ax, update_functions):
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for s in space:
            ax.plot(s[:, 0], s[:, 1], color='black')

        actors = []
        for ped in range(initial_state.shape[0]):
            p, = ax.plot(states[0:5, ped, 0], states[0:5, ped, 1],
                         '-o', label='ped {}'.format(ped), markersize=2.5)
            actors.append(p)

        def update(i):
            for ped, p in enumerate(actors):
                p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])

        update_functions.append(update)
