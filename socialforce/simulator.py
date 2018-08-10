"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np

from .potentials import PedPedPotential, PedSpacePotential
from .fieldofview import FieldOfView

MEAN_VELOCITY = 1.34  # m/s
SIGMA_VEL = 0.26  # std dev in m/s

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed


class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector in phase space (x, y, v_x, v_y).

    space is a list of numpy arrays containing points of boundaries.

    delta_t in seconds.
    tau in seconds.
    """
    def __init__(self, initial_state, destinations, space=None, delta_t=0.4, tau=0.5):
        self.state = initial_state
        self.destinations = destinations
        self.space = space or []
        self.delta_t = delta_t
        self.tau = tau

        self.initial_speeds = self.speeds()
        self.max_speeds = MAX_SPEED_MULTIPLIER * self.initial_speeds

        # potentials
        self.V = PedPedPotential(self.delta_t)
        self.U = PedSpacePotential()

        # field of view
        self.w = FieldOfView()

    def speeds(self):
        """Calculate the speeds of all pedestrians."""
        velocities = self.state[:, 2:4]
        return np.linalg.norm(velocities, axis=1)

    def desired_directions(self):
        """Given the current state and destination, compute desired direction."""
        destination_vectors = self.destinations - self.state[:, 0:2]
        norm_factors = np.linalg.norm(destination_vectors, axis=-1)
        return destination_vectors / np.expand_dims(norm_factors, -1)

    def rab(self):
        """r_ab"""
        r = self.state[:, 0:2]
        r_a = np.expand_dims(r, 1)
        r_b = np.expand_dims(r, 0)
        return r_a - r_b

    def fab(self):
        """Compute f_ab using finite difference differentiation."""
        return -1.0 * self.V.grad_rab(self.rab(), self.speeds(), self.desired_directions())

    def raB(self):
        """r_aB"""
        if not self.space:
            return np.zeros((self.state.shape[0], 0, 2))

        r_a = np.expand_dims(self.state[:, 0:2], 1)
        closest_i = [
            np.argmin(np.linalg.norm(r_a - np.expand_dims(B, 0), axis=-1), axis=1)
            for B in self.space
        ]
        closest_points = np.swapaxes(
            np.stack([B[i] for B, i in zip(self.space, closest_i)]),
            0, 1)  # index order: pedestrian, boundary, coordinates
        return r_a - closest_points

    def faB(self):
        return -1.0 * self.U.grad_raB(self.raB())

    def capped_velocity(self, desired_velocity):
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        # accelerate to desired velocity
        e = self.desired_directions()
        vel = self.state[:, 2:4]
        F0 = 1.0 / self.tau * (np.expand_dims(self.initial_speeds, -1) * e - vel)

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
