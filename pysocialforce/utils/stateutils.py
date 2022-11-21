"""Utility functions to process state."""
from typing import Tuple

import numpy as np
from numba import njit

import logging
logging.getLogger('numba').setLevel(logging.WARNING)


# @jit
# def normalize(array_in):
#     """nx2 or mxnx2"""
#     if len(array_in.shape) == 2:
#         vec, fac = normalize_array(array_in)
#         return vec, fac
#     factors = []
#     vectors = []
#     for m in array_in:
#         vec, fac = normalize_array(m)
#         vectors.append(vec)
#         factors.append(fac)

#     return np.array(vectors), np.array(factors)


@njit
def vec_len_2d(vec_x: float, vec_y: float) -> float:
    return (vec_x**2 + vec_y**2)**0.5


@njit
def vector_angles(vecs: np.ndarray) -> np.ndarray:
    """Calculate angles for an array of vectors
    :param vecs: nx2 ndarray
    :return: nx1 ndarray
    """
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
    return ang


@njit
def left_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
    return vecs


@njit
def right_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([1.0, -1.0])
    return vecs


@njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    num_vecs = vecs.shape[0]
    vec_lengths = np.zeros((num_vecs))
    unit_vecs = np.zeros((num_vecs, 2))

    for i, (vec_x, vec_y) in enumerate(vecs):
        vec_len = vec_len_2d(vec_x, vec_y)
        vec_lengths[i] = vec_len
        if vec_len > 0:
            unit_vecs[i] = vecs[i] / vec_len

    return unit_vecs, vec_lengths


@njit
def desired_directions(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a − r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
    diff = diff[
        ~np.eye(diff.shape[0], dtype=bool), :
    ]  # get rif of diagonal elements in the diff matrix
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

    return diff


# @njit
def speeds(state: np.ndarray) -> np.ndarray:
    """Return the speeds corresponding to a given state."""
    return np.linalg.norm(state[:, 2:4], axis=1)
    # TODO: enable @njit if a newer version of numba supports the axis argument


@njit
def centroid(vecs: np.ndarray) -> Tuple[float, float]:
    """Center-of-mass of a given group as arithmetic mean."""
    num_datapoints = vecs.shape[0]
    centroid_x, centroid_y = 0, 0
    for x, y in vecs:
        centroid_x += x
        centroid_y += y
    centroid_x /= num_datapoints
    centroid_y /= num_datapoints
    return centroid_x, centroid_y


@njit
def minmax(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.min(vecs[:, 0])
    y_min = np.min(vecs[:, 1])
    x_max = np.max(vecs[:, 0])
    y_max = np.max(vecs[:, 1])
    return (x_min, y_min, x_max, y_max)
