"""Interaction potentials."""

import numpy as np


class PedPedPotential(object):
    """Ped-ped interaction potential.

    v0 is in m^2 / s^2.
    sigma is in m.
    """

    def __init__(self, delta_t, v0=2.1, sigma=0.3):
        self.delta_t = delta_t
        self.v0 = v0
        self.sigma = sigma

    def b(self, rab, speeds, desired_directions):
        speeds_b = np.expand_dims(speeds, axis=0)
        speeds_b_abc = np.expand_dims(speeds_b, axis=2)  # abc = alpha, beta, coordinates
        e_b = np.expand_dims(desired_directions, axis=0)

        in_sqrt = (
            np.linalg.norm(rab, axis=-1) +
            np.linalg.norm(rab - self.delta_t * speeds_b_abc * e_b, axis=-1)
        )**2 - (self.delta_t * speeds_b)**2
        np.fill_diagonal(in_sqrt, 0.0)

        return 0.5 * np.sqrt(in_sqrt)

    def __call__(self, rab, speeds, desired_directions):
        return self.v0 * np.exp(-self.b(rab, speeds, desired_directions) / self.sigma)

    def grad_rab(self, rab, speeds, desired_directions, delta=1e-3):
        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self(rab, speeds, desired_directions)
        dvdx = (self(rab + dx, speeds, desired_directions) - v) / delta
        dvdy = (self(rab + dy, speeds, desired_directions) - v) / delta

        # remove gradients from self-intereactions
        np.fill_diagonal(dvdx, 0.0)
        np.fill_diagonal(dvdy, 0.0)

        return np.stack((dvdx, dvdy), axis=-1)


class PedSpacePotential(object):
    """Pedestrian-space interaction potential.

    u0 is in m^2 / s^2.
    r is in m
    """

    def __init__(self, u0=10, r=0.2):
        self.u0 = u0
        self.r = r

    def __call__(self, raB):
        return self.u0 * np.exp(-1.0 * np.linalg.norm(raB, axis=-1) / self.r)

    def grad_raB(self, raB, delta=1e-3):
        dx = np.array([[[delta, 0.0]]])
        dy = np.array([[[0.0, delta]]])

        v = self(raB)
        dvdx = (self(raB + dx) - v) / delta
        dvdy = (self(raB + dy) - v) / delta

        return np.stack((dvdx, dvdy), axis=-1)
