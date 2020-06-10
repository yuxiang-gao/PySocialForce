"""Calculate forces for individuals and groups"""
import re
from abc import ABC, abstractmethod

import numpy as np

from pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pysocialforce.fieldofview import FieldOfView
from pysocialforce.utils import Config
from pysocialforce.utils import stateutils


def camel_to_snake(camel_case_string):
    """Convert CamelCase to snake_case"""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self):
        super().__init__()
        self.scene = None
        self.peds = None
        self.factor = 1.0
        self.config = Config()

    def init(self, scene, config):
        """Load config and scene"""
        # load the sub field corresponding to the force name from global confgi file
        self.config = config.sub_config(camel_to_snake(type(self).__name__))
        if self.config:
            self.factor = self.config("factor")

        self.scene = scene
        self.peds = self.scene.peds

    @abstractmethod
    def get_force(self):
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def get_force(self):
        F0 = (
            1.0
            / self.peds.tau()
            * (
                np.expand_dims(self.peds.initial_speeds, -1) * self.peds.desired_directions()
                - self.peds.vel()
            )
        )
        return F0 * self.factor


class PedRepulsiveForce(Force):
    """Ped to ped repulsive force"""

    def get_force(self):
        potential_func = PedPedPotential(
            self.peds.timestep, v0=self.config("v0"), sigma=self.config("sigma"),
        )
        f_ab = -1.0 * potential_func.grad_r_ab(self.peds.state)

        fov = FieldOfView(phi=self.config("fov_phi"), out_of_view_factor=self.config("fov_factor"),)
        w = np.expand_dims(fov(self.peds.desired_directions(), -f_ab), -1)
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1) * self.factor


class SpaceRepulsiveForce(Force):
    """obstacles to ped repulsive force"""

    def get_force(self):
        if self.scene.obstacles is None:
            F_aB = np.zeros((self.peds.size(), 0, 2))
        else:
            potential_func = PedSpacePotential(
                self.scene.obstacles, u0=self.config("u0"), r=self.config("r")
            )
            F_aB = -1.0 * potential_func.grad_r_aB(self.peds.state)
        return np.sum(F_aB, axis=1) * self.factor


class GroupCoherenceForce(Force):
    """Group coherence force, paper version"""

    def get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                vectors, norms = stateutils.normalize(force_vec)
                vectors[norms < threshold] = [0, 0]
                forces[group, :] += vectors
        return forces * self.factor


class GroupCoherenceForceAlt(Force):
    """ Alternative group coherence force as specified in pedsim_ros"""

    def get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                norms = stateutils.speeds(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.factor


class GroupRepulsiveForce(Force):
    """Group repulsive force"""

    def get_force(self):
        threshold = self.config("threshold") or 0.5
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                member_pos = self.peds.pos()[group, :]
                for m in stateutils.vec_diff(member_pos):
                    vectors, norms = stateutils.normalize(m)
                    vectors = np.nan_to_num(vectors)
                    vectors[norms > threshold] = [0, 0]
                    forces[group, :] += m

        return forces * self.factor


class GroupGazeForce(Force):
    """Group gaze force"""

    def get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        vision_angle = self.config("fov_phi") or 100.0
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = self.peds.desired_directions()[group, :]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, _ = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                com_angles = np.degrees(
                    [np.arccos(np.dot(d, c)) for d, c in zip(member_directions, com_directions)]
                )
                rotation = np.radians(
                    [a - vision_angle if a > vision_angle else 0.0 for a in com_angles]
                )
                force = -np.expand_dims(rotation, -1) * member_directions
                forces[group, :] += force
        return forces * self.factor
