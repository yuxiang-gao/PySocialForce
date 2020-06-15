"""Interaction potentials."""

import numpy as np

from pysocialforce.utils import stateutils


class PedPedPotential(object):
    """Ped-ped interaction potential.

    v0 is in m^2 / s^2.
    sigma is in m.
    """

    def __init__(self, delta_t, v0=None, sigma=None):
        self.delta_t = delta_t
        self.v0 = v0 or 2.1
        self.sigma = sigma or 0.3

    def b(self, r_ab, speeds, desired_directions):
        """Calculate b.
        b denotes the semi-minor axis of the ellipse and is given by
        e: desired direction
        2b=sqrt((r_ab+(r_ab-v*delta_t*e_b))
        """
        speeds_b = np.expand_dims(speeds, axis=0)
        speeds_b_abc = np.expand_dims(speeds_b, axis=2)  # abc = alpha, beta, coordinates
        e_b = np.expand_dims(desired_directions, axis=0)

        in_sqrt = (
            np.linalg.norm(r_ab, axis=-1)
            + np.linalg.norm(r_ab - self.delta_t * speeds_b_abc * e_b, axis=-1)
        ) ** 2 - (self.delta_t * speeds_b) ** 2
        np.fill_diagonal(in_sqrt, 0.0)

        return 0.5 * np.sqrt(in_sqrt)

    def value_r_ab(self, r_ab, speeds, desired_directions):
        """Value of potential explicitly parametrized with r_ab."""
        return self.v0 * np.exp(-self.b(r_ab, speeds, desired_directions) / self.sigma)

    @staticmethod
    def r_ab(state):
        """r_ab
        r_ab := r_a − r_b.
        """
        return stateutils.vec_diff(state[:, :2])

    def __call__(self, state):
        speeds = stateutils.speeds(state)
        return self.value_r_ab(self.r_ab(state), speeds, stateutils.desired_directions(state))

    def grad_r_ab(self, state, delta=1e-3):
        """Compute gradient wrt r_ab using finite difference differentiation."""
        r_ab = self.r_ab(state)
        speeds = stateutils.speeds(state)
        desired_directions = stateutils.desired_directions(state)

        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self.value_r_ab(r_ab, speeds, desired_directions)
        dvdx = (self.value_r_ab(r_ab + dx, speeds, desired_directions) - v) / delta
        dvdy = (self.value_r_ab(r_ab + dy, speeds, desired_directions) - v) / delta

        # remove gradients from self-intereactions
        np.fill_diagonal(dvdx, 0.0)
        np.fill_diagonal(dvdy, 0.0)

        return np.stack((dvdx, dvdy), axis=-1)


class PedSpacePotential(object):
    """Pedestrian-obstacles interaction potential.

    obstacles is a list of numpy arrays containing points of boundaries.

    u0 is in m^2 / s^2.
    r is in m
    """

    def __init__(self, obstacles, u0=None, r=None):
        self.obstacles = obstacles or []
        self.u0 = u0 or 10
        self.r = r or 0.2

    def value_r_aB(self, r_aB):
        """Compute value parametrized with r_aB."""
        return self.u0 * np.exp(-1.0 * np.linalg.norm(r_aB, axis=-1) / self.r)

    def r_aB(self, state):
        """r_aB"""
        if not self.obstacles:
            return np.zeros((state.shape[0], 0, 2))

        r_a = np.expand_dims(state[:, 0:2], 1)
        closest_i = [
            np.argmin(np.linalg.norm(r_a - np.expand_dims(B, 0), axis=-1), axis=1)
            for B in self.obstacles
        ]
        closest_points = np.swapaxes(
            np.stack([B[i] for B, i in zip(self.obstacles, closest_i)]), 0, 1
        )  # index order: pedestrian, boundary, coordinates
        return r_a - closest_points

    def __call__(self, state):
        return self.value_r_aB(self.r_aB(state))

    def grad_r_aB(self, state, delta=1e-3):
        """Compute gradient wrt r_aB using finite difference differentiation."""
        r_aB = self.r_aB(state)

        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self.value_r_aB(r_aB)
        dvdx = (self.value_r_aB(r_aB + dx) - v) / delta
        dvdy = (self.value_r_aB(r_aB + dy) - v) / delta

        return np.stack((dvdx, dvdy), axis=-1)
