"""Utility functions to process state."""

import numpy as np
from numba import njit


@njit
def desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, norm_factors = normalize(destination_vectors)
    # directions[norm_factors == 0] = [0, 0] not supported
    for i in range(norm_factors.shape[0]):
        if norm_factors[i] == 0:
            directions[i] = [0, 0]
    return directions


@njit
def normalize(mat):
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = []
    for line in mat:
        norm_factors.append(np.linalg.norm(line))
    norm_factors = np.array(norm_factors)
    normalized = mat / np.expand_dims(norm_factors, -1)
    return normalized, norm_factors


@njit
def vec_diff(state):
    """r_ab
    r_ab := r_a − r_b.
    """
    r = state[:, 0:2]
    r_a = np.expand_dims(r, 1)
    r_b = np.expand_dims(r, 0)
    return r_a - r_b


@njit
def speeds(state):
    """Return the speeds corresponding to a given state."""
    #     return np.linalg.norm(state[:, 2:4], axis=-1)
    speed_vecs = state[:, 2:4]
    speeds_array = np.array([np.linalg.norm(s) for s in speed_vecs])
    return speeds_array


@njit
def group_center(state):
    """Center-of-mass of a given group"""
    return np.sum(state[:, 0:2], axis=0) / state.shape[0]
