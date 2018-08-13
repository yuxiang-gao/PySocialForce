"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np

from .potentials import PedPedPotential, PedSpacePotential
from .fieldofview import FieldOfView
from . import stateutils

MEAN_VELOCITY = 1.34  # m/s
SIGMA_VEL = 0.26  # std dev in m/s

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed


class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (x, y, v_x, v_y, d_x, d_y, [tau]).
    tau is optional in this vector.

    space is a list of numpy arrays containing points of boundaries.

    delta_t in seconds.
    tau in seconds: either float or numpy array of shape[n_ped].
    """
    def __init__(self, initial_state, space=None, delta_t=0.4, tau=0.5):
        self.state = initial_state
        self.initial_speeds = stateutils.speeds(initial_state)
        self.max_speeds = MAX_SPEED_MULTIPLIER * self.initial_speeds

        self.delta_t = delta_t

        if self.state.shape[1] < 7:
            if not hasattr(tau, 'shape'):
                tau = tau * np.ones(self.state.shape[0])
            self.state = np.concatenate((self.state, np.expand_dims(tau, -1)), axis=-1)

        # potentials
        self.V = PedPedPotential(self.delta_t)
        self.U = PedSpacePotential(space)

        # field of view
        self.w = FieldOfView()

    def fab(self):
        """Compute f_ab."""
        return -1.0 * self.V.grad_rab(self.state)

    def faB(self):
        """Compute f_aB."""
        return -1.0 * self.U.grad_raB(self.state)

    def capped_velocity(self, desired_velocity):
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        # accelerate to desired velocity
        e = stateutils.desired_directions(self.state)
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        F0 = 1.0 / tau * (np.expand_dims(self.initial_speeds, -1) * e - vel)

        # repulsive terms between pedestrians
        fab = self.fab()
        w = np.expand_dims(self.w(e, -fab), -1)
        Fab = w * fab

        # repulsive terms between pedestrians and boundaries
        FaB = self.faB()

        # social force
        F = F0 + np.sum(Fab, axis=1) + np.sum(FaB, axis=1)
        # desired velocity
        w = self.state[:, 2:4] + self.delta_t * F
        # velocity
        v = self.capped_velocity(w)

        # update state
        self.state[:, 0:2] += v * self.delta_t
        self.state[:, 2:4] = v

        return self
