import matplotlib.pyplot as plt
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
    with socialforce.show.animation(
            len(states),
            'docs/gate.gif',
            writer='imagemagick') as context:
        ax = context['ax']
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        for s in space:
            ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

        actors = []
        for ped in range(initial_state.shape[0]):
            # p, = ax.plot(states[0:5, ped, 0], states[0:5, ped, 1],
            #              '-o', label='ped {}'.format(ped), markersize=2.5)
            speed = np.linalg.norm(states[0, ped, 2:4])
            radius = 0.2 + speed / 2.0 * 0.3
            p = plt.Circle(states[0, ped, 0:2], radius=radius,
                           facecolor='black' if states[0, ped, 0] < 0 else 'white',
                           edgecolor='black')
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                p.center = states[i, ped, 0:2]
                speed = np.linalg.norm(states[i, ped, 2:4])
                p.set_radius(0.2 + speed / 2.0 * 0.3)

        context['update_function'] = update
