import numpy as np
from . import stateutils
from .potentials import PedPedPotential, PedSpacePotential
from .fieldofview import FieldOfView
from abc import ABC, abstractmethod


def to_snake(camel_case_string):
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self):
        self.name = to_snake(type(self).__name__)
        self.state = None
        self.groups = None

    def load_config(self, config_dict):
        self.config = config_dict.get(self.name)

        self.factor = self.config.get("factor") or 1.0
        self.time_step = config_dict.get("time_step")

    def set_state(self, state, groups=None):
        self.state = state
        self.groups = groups
        self.goal_vector = stateutils.desired_directions(self.state)  # e

    @abstractmethod
    def get_force(self):
        pass


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def get_force(self):
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        initial_speeds = stateutils.speeds(self.state)
        F0 = 1.0 / tau * (np.expand_dims(initial_speeds, -1) * self.goal_vector - vel)
        return F0


class PedRepulsiveForce(Force):
    def get_force(self):
        potential_func = PedPedPotential(
            v0=self.config["v0"], sigma=self.config["sigma"]
        )
        f_ab = -1.0 * potential_func.grad_r_ab(self.state)

        fov = FieldOfView(
            phi=self.config.get("fov_phi"),
            out_of_view_factor=self.config.get("fov_factor"),
        )

        w = np.expand_dims(fov(self.goal_vector, -f_ab), -1)
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1)


class SpaceRepulsiveForce(Force):
    def set_space(self, space):
        self.space = space

    def get_force(self):
        if self.space is None:
            F_aB = np.zeros((self.state.shape[0], 0, 2))
        else:
            potential_func = PedSpacePotential(space)
            F_aB = -self.config.get("factor") * potential_func.grad_r_aB(self.state)
        return np.sum(F_aB, axis=1)


class GroupCoherenceForce(Force):
    def get_force(self):
        pass


class GroupRepulsiveForce(Force):
    def get_force(self):
        pass


class GroupGazeForce(Force):
    def get_force(self):
        pass
