"""Calculate forces for individuals and groups"""
import re
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numba import njit

from pysocialforce.scene import Line2D, Point2D
from pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pysocialforce.fieldofview import FieldOfView
from pysocialforce.utils import Config, stateutils, logger


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
            self.factor = self.config("factor", 1.0)

        self.scene = scene
        self.peds = self.scene.peds

    @abstractmethod
    def _get_force(self) -> np.ndarray:
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError

    def get_force(self, debug=False):
        force = self._get_force()
        if debug:
            logger.debug(f"{camel_to_snake(type(self).__name__)}:\n {repr(force)}")
        return force


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def _get_force(self):
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

    def _get_force(self):
        potential_func = PedPedPotential(
            self.peds.step_width, v0=self.config("v0"), sigma=self.config("sigma"),
        )
        f_ab = -1.0 * potential_func.grad_r_ab(self.peds.state)

        fov = FieldOfView(phi=self.config("fov_phi"), out_of_view_factor=self.config("fov_factor"),)
        w = np.expand_dims(fov(self.peds.desired_directions(), -f_ab), -1)
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1) * self.factor


class SpaceRepulsiveForce(Force):
    """obstacles to ped repulsive force"""

    def _get_force(self):
        if self.scene.get_obstacles() is None:
            F_aB = np.zeros((self.peds.size(), 0, 2))
        else:
            potential_func = PedSpacePotential(
                self.scene.get_obstacles(), u0=self.config("u0"), r=self.config("r")
            )
            F_aB = -1.0 * potential_func.grad_r_aB(self.peds.state)
        return np.sum(F_aB, axis=1) * self.factor


class GroupCoherenceForce(Force):
    """Group coherence force, paper version"""

    def _get_force(self):
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

    def _get_force(self):
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

    def _get_force(self):
        threshold = self.config("threshold", 0.5)
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                size = len(group)
                member_pos = self.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = stateutils.normalize(diff)
                diff[norms > threshold, :] = 0
                # forces[group, :] += np.sum(diff, axis=0)
                forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        return forces * self.factor


class GroupGazeForce(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        vision_angle = self.config("fov_phi", 100.0)
        directions, _ = stateutils.desired_directions(self.peds.state)
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
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
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                com_angles = np.degrees(np.arccos(element_prod))
                rotation = np.radians(
                    [a - vision_angle if a > vision_angle else 0.0 for a in com_angles]
                )
                force = -rotation.reshape(-1, 1) * member_directions
                forces[group, :] += force

        return forces * self.factor


class GroupGazeForceAlt(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        directions, dist = stateutils.desired_directions(self.peds.state)
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
                member_dist = dist[group]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, com_dist = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                force = (
                    com_dist.reshape(-1, 1)
                    * element_prod.reshape(-1, 1)
                    / member_dist.reshape(-1, 1)
                    * member_directions
                )
                forces[group, :] += force

        return forces * self.factor


class DesiredForce(Force):
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """

    def _get_force(self):
        relexation_time = self.config("relaxation_time", 0.5)
        goal_threshold = self.config("goal_threshold", 0.1)
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        direction, dist = stateutils.normalize(goal - pos)
        force = np.zeros((self.peds.size(), 2))
        force[dist > goal_threshold] = (
            direction * self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]
        force /= relexation_time
        return force * self.factor


class SocialForce(Force):
    """Calculates the social force between this agent and all the other agents
    belonging to the same scene.
    It iterates over all agents inside the scene, has therefore the complexity
    O(N^2). A better
    agent storing structure in Tscene would fix this. But for small (less than
    10000 agents) scenarios, this is just
    fine.
    :return:  nx2 ndarray the calculated force
    """

    def _get_force(self):
        lambda_importance = self.config("lambda_importance", 2.0)
        gamma = self.config("gamma", 0.35)
        n = self.config("n", 2)
        n_prime = self.config("n_prime", 3)

        pos_diff = stateutils.each_diff(self.peds.pos())  # n*(n-1)x2 other - self
        diff_direction, diff_length = stateutils.normalize(pos_diff)
        vel_diff = -1.0 * stateutils.each_diff(self.peds.vel())  # n*(n-1)x2 self - other

        # compute interaction direction t_ij
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # compute angle theta (between interaction and position difference vector)
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(
            diff_direction
        )
        # compute model parameter B = gamma * ||D||
        B = gamma * interaction_length

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(
            interaction_direction
        )

        force = force_velocity + force_angle  # n*(n-1) x 2
        force = np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1)
        return force * self.factor


# class ObstacleForce(Force):
#     """Calculates the force between this agent and the nearest obstacle in this
#     scene.
#     :return:  the calculated force
#     """

#     def _get_force(self):
#         sigma = self.config("sigma", 0.2)
#         threshold = self.config("threshold", 0.2) + self.peds.agent_radius
#         force = np.zeros((self.peds.size(), 2))
#         if len(self.scene.get_obstacles()) == 0:
#             return force
#         obstacles = np.vstack(self.scene.get_obstacles())
#         pos = self.peds.pos()

#         for i, p in enumerate(pos):
#             diff = p - obstacles
#             directions, dist = stateutils.normalize(diff)
#             dist = dist - self.peds.agent_radius
#             if np.all(dist >= threshold):
#                 continue
#             dist_mask = dist < threshold
#             directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
#             force[i] = np.sum(directions[dist_mask], axis=0)

#         return force * self.factor


class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this
    scene.
    :return:  the calculated force
    """

    def _get_force(self) -> np.ndarray:
        """Computes the obstacle forces per pedestrian,
        output shape (num_peds, 2), forces in x/y direction"""

        ped_positions = self.peds.pos()
        forces = np.zeros((ped_positions.shape[0], 2))
        obstacles = self.scene.get_raw_obstacles()
        if len(obstacles) == 0:
            return forces

        # TODO: figure out how to add those variables into the mix
        # sigma = self.config.sigma
        # threshold = self.config.threshold + self.peds.agent_radius

        radius = self.peds.agent_radius
        all_obstacle_forces(forces, ped_positions, obstacles, radius)
        return forces * self.factor


@njit(fastmath=True)
def all_obstacle_forces(out_forces: np.ndarray, ped_positions: np.ndarray,
                        obstacles: np.ndarray, ped_radius: float):
    obstacle_segments = obstacles[:, :4]
    ortho_vecs = obstacles[:, 4:]
    num_peds = ped_positions.shape[0]
    num_obstacles = obstacles.shape[0]
    for i in range(num_peds):
        ped_pos = ped_positions[i]
        for j in range(num_obstacles):
            force_x, force_y = obstacle_force(
                obstacle_segments[j], ortho_vecs[j], ped_pos, ped_radius)
            out_forces[i, 0] += force_x
            out_forces[i, 1] += force_y


@njit(fastmath=True)
def obstacle_force(obstacle: Line2D, ortho_vec: Point2D,
                   ped_pos: Point2D, ped_radius: float) -> Tuple[float, float]:
    """The obstacle force between a line segment (= obstacle) and
    a point (= pedestrian's position) is computed as follows:
    1) compute the distance between the line segment and the point
    2) compute the repulsive force, i.e. the partial derivative by x/y of the point
    regarding the virtual potential field denoted as 1 / (2 * dist(line_seg, point)^2)
    3) return the force as separate x/y components
    There are 3 cases to be considered for computing the distance:
    1) obstacle is just a point instead of a line segment
    2) orthogonal projection hits within the obstacle's line segment
    3) orthogonal projection doesn't hit within the obstacle's line segment"""

    x1, y1, x2, y2 = obstacle
    (x3, y3), (x4, y4) = ped_pos, (ped_pos[0] + ortho_vec[0], ped_pos[1] + ortho_vec[1])

    # handle edge case where the obstacle is just a point
    if (x1, y1) == (x2, y2):
        obst_dist = euclid_dist(ped_pos[0], ped_pos[1], x1, y1) - ped_radius
        dx_obst_dist, dy_obst_dist = der_euclid_dist(ped_pos, (x1, y1), obst_dist)
        return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)

    # info: there's always an intersection with the orthogonal vector
    num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = num / den
    ortho_hit = 0 <= t <= 1

    # orthogonal vector doesn't hit within segment bounds
    if not ortho_hit:
        d1 = euclid_dist(ped_pos[0], ped_pos[1], x1, y1)
        d2 = euclid_dist(ped_pos[0], ped_pos[1], x2, y2)
        obst_dist = min(d1, d2) - ped_radius
        closer_obst_bound = (x1, y1) if d1 < d2 else (x2, y2)
        dx_obst_dist, dy_obst_dist = der_euclid_dist(ped_pos, closer_obst_bound, obst_dist)
        return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)

    # orthogonal vector hits within segment bounds
    cross_x, cross_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    obst_dist = euclid_dist(ped_pos[0], ped_pos[1], cross_x, cross_y) - ped_radius
    dx3_cross_x = (y4 - y3) / den * (x2 - x1)
    dx3_cross_y = (y4 - y3) / den * (y2 - y1)
    dy3_cross_x = (x3 - x4) / den * (x2 - x1)
    dy3_cross_y = (x3 - x4) / den * (y2 - y1)
    dx_obst_dist = ((cross_x - ped_pos[0]) * (dx3_cross_x - 1) \
        + (cross_y - ped_pos[1]) * dx3_cross_y) / obst_dist
    dy_obst_dist = ((cross_x - ped_pos[0]) * dy3_cross_x \
        + (cross_y - ped_pos[1]) * (dy3_cross_y - 1)) / obst_dist
    return potential_field_force(obst_dist, dx_obst_dist, dy_obst_dist)


@njit(fastmath=True)
def potential_field_force(obst_dist: float, dx_obst_dist: float,
                          dy_obst_dist: float) -> Tuple[float, float]:
    der_potential = 1 / pow(obst_dist, 3)
    return der_potential * dx_obst_dist, der_potential * dy_obst_dist


@njit(fastmath=True)
def euclid_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), 0.5)


@njit(fastmath=True)
def der_euclid_dist(p1: Point2D, p2: Point2D, distance: float) -> Tuple[float, float]:
    # info: distance is an expensive operation and therefore pre-computed
    dx1_dist = (p1[0] - p2[0]) / distance
    dy1_dist = (p1[1] - p2[1]) / distance
    return dx1_dist, dy1_dist
