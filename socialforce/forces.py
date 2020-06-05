import numpy as np
from . import stateutils
from .stateutils import normalize, vec_diff
from .potentials import PedPedPotential, PedSpacePotential
from .fieldofview import FieldOfView
from abc import ABC, abstractmethod
from itertools import combinations
from numba import jit, njit


def to_snake(camel_case_string):
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self):
        super().__init__()
        self.name = to_snake(type(self).__name__)
        self.state = None
        self.groups = None
        self.factor = 1.0
        self.time_step = 0.4
        self.config = {}

    def load_config(self, config_dict):
        self.config = config_dict.get(self.name)
        if self.config:
            self.factor = self.config.get("factor")
            self.time_step = config_dict.get("time_step")

    def set_state(self, state, groups=None, initial_speeds=None):
        self.state = np.array(state)
        self.groups = groups
        self.goal_vector = stateutils.desired_directions(self.state)  # e
        if initial_speeds is not None:
            self.initial_speeds = initial_speeds

    @abstractmethod
    def get_force(self):
        raise NotImplementedError


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def get_force(self):
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        F0 = (
            1.0
            / tau
            * (np.expand_dims(self.initial_speeds, -1) * self.goal_vector - vel)
        )
        return F0 * self.factor


class PedRepulsiveForce(Force):
    def get_force(self):
        potential_func = PedPedPotential(
            self.time_step, v0=self.config["v0"], sigma=self.config["sigma"]
        )
        f_ab = -1.0 * potential_func.grad_r_ab(self.state)

        fov = FieldOfView(
            phi=self.config.get("fov_phi"),
            out_of_view_factor=self.config.get("fov_factor"),
        )
        w = np.expand_dims(fov(self.goal_vector, -f_ab), -1)
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1) * self.factor


class SpaceRepulsiveForce(Force):
    def set_space(self, space):
        self.space = space

    def get_force(self):
        if self.space is None:
            F_aB = np.zeros((self.state.shape[0], 0, 2))
        else:
            potential_func = PedSpacePotential(
                self.space, u0=self.config.get("u0"), r=self.config.get("r")
            )
            F_aB = -potential_func.grad_r_aB(self.state)
        return np.sum(F_aB, axis=1) * self.factor


class GroupCoherenceForce(Force):
    def get_force(self):
        forces = np.zeros((self.state.shape[0], 2))
        if self.groups is not None:
            for group in self.groups:
                threshold = (len(group) - 1) / 2
                member_states = self.state[group, :]
                member_pos = member_states[:, 0:2]
                com = stateutils.group_center(member_states)
                force_vec = com - member_pos
                vectors, norms = normalize(force_vec)
                vectors[norms < threshold] = [0, 0]
                forces[group, :] += vectors
        return forces * self.factor


class GroupCoherenceForceAlt(Force):
    def get_force(self):
        forces = np.zeros((self.state.shape[0], 2))
        if self.groups is not None:
            for group in self.groups:
                threshold = (len(group) - 1) / 2
                member_states = self.state[group, :]
                member_pos = member_states[:, 0:2]
                com = stateutils.group_center(member_states)
                force_vec = com - member_pos
                vectors, norms = normalize(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.factor


class GroupRepulsiveForce(Force):
    def get_force(self):
        threshold = self.config.get("threshold") or 0.5
        forces = np.zeros((self.state.shape[0], 2))
        if self.groups is not None:
            for group in self.groups:
                member_pos = self.state[group][:, 0:2]
                for m in vec_diff(member_pos):
                    vectors, norms = normalize(m)
                    vectors = np.nan_to_num(vectors)
                    vectors[norms > threshold] = [0, 0]
                    forces[group, :] += m

        return forces * self.factor


class GroupGazeForce(Force):
    def get_force(self):
        forces = np.zeros((self.state.shape[0], 2))
        vision_angle = self.config.get("fov_phi") or 100.0
        if self.groups is not None:
            for group in self.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_states = self.state[group, :]
                member_pos = member_states[:, 0:2]
                member_directions = self.goal_vector[group, :]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.group_center(
                            member_pos[np.arange(group_size) != i, :]
                        )
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, _ = normalize(relative_com)
                # angle between walking direction and center of mass
                com_angles = np.degrees(
                    [
                        np.arccos(np.dot(d, c))
                        for d, c in zip(member_directions, com_directions)
                    ]
                )
                rotation = np.radians(
                    [a - vision_angle if a > vision_angle else 0.0 for a in com_angles]
                )
                force = -np.expand_dims(rotation, -1) * member_directions
                forces[group, :] += force
        return forces * self.factor
