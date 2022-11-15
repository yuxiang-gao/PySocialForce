"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from math import cos, sin, atan2, pi
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np

from pysocialforce.utils import stateutils


Line2D = Tuple[float, float, float, float]
Point2D = Tuple[float, float]


class PedState:
    """Tracks the state of pedstrains and social groups"""

    def __init__(self, state, groups, config):
        self.default_tau = config("tau", 0.5)
        self.step_width = config("step_width", 0.4)
        self.agent_radius = config("agent_radius", 0.35)
        self.max_speed_multiplier = config("max_speed_multiplier", 1.3)

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []

        self.update(state, groups)

    def update(self, state, groups):
        self.state = state
        self.groups = groups

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        tau = self.default_tau * np.ones(state.shape[0])
        if state.shape[1] < 7:
            self._state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())

    def get_states(self):
        return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]

    def tau(self):
        return self.state[:, 6:7]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        """Move peds according to forces"""
        # desired velocity
        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # stop when arrived
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]

        # update state
        next_state = self.state
        next_state[:, 0:2] += desired_velocity * self.step_width
        next_state[:, 2:4] = desired_velocity
        next_groups = self.groups
        if groups is not None:
            next_groups = groups
        self.update(next_state, next_groups)

    # def initial_speeds(self):
    #     return stateutils.speeds(self.ped_states[0])

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    # def get_group_by_idx(self, index: int) -> np.ndarray:
    #     return self.state[self.groups[index], :]

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1


@dataclass
class EnvState:
    """State of the environment obstacles"""
    _orig_obstacles: List[Line2D]
    _resolution: int=10
    _obstacles_linspace: List[np.ndarray] = field(init=False)
    _obstacles_raw: np.ndarray = field(init=False)

    def __post_init__(self):
        self._obstacles_raw = self._update_obstacles_raw(self._orig_obstacles)
        self._obstacles_linspace = self._update_obstacles_linspace(self._orig_obstacles)

    @property
    def obstacles_raw(self) -> np.ndarray:
        """a 2D numpy array representing a list of 2D lines
        as (start_x, end_x, start_y, end_y) for array indices 0-3.
        Additionally, the array contains the orthogonal unit vector
        for each 2D line at indices 4-5."""
        return self._obstacles_raw

    @property
    def obstacles(self) -> List[np.ndarray]:
        """a list of np.ndarrays, each representing a uniform
        linspace of 0.1 steps between |p_start, p_end|"""
        return self._obstacles_linspace

    @obstacles.setter
    def obstacles(self, obstacles: List[Line2D]):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        self._orig_obstacles = obstacles
        self._obstacles_raw = self._update_obstacles_raw(obstacles)
        self._obstacles_linspace = self._update_obstacles_linspace(obstacles)

    def _update_obstacles_linspace(self, obs_lines: List[Line2D]) -> List[np.ndarray]:
        if obs_lines is None:
            obstacles = []
        else:
            obstacles = []
            for start_x, end_x, start_y, end_y in obs_lines:
                samples = int(np.linalg.norm((start_x - end_x, start_y - end_y)) * self._resolution)
                line = np.array(list(zip(
                    np.linspace(start_x, end_x, samples),
                    np.linspace(start_y, end_y, samples))))
                obstacles.append(line)
        return obstacles

    def _update_obstacles_raw(self, obs_lines: List[Line2D]) -> np.ndarray:
        def vec_dir_rad(vec: Point2D) -> float:
            return atan2(vec[1], vec[0])

        def unit_vec(orient: float) -> Point2D:
            return cos(orient), sin(orient)

        if obs_lines is None:
            return np.array([])

        ortho_left = [unit_vec(vec_dir_rad((end_x - start_x, end_y - start_y)) + pi/2)
                      for start_x, start_y, end_x, end_y in obs_lines]
        obstacles = np.zeros((len(obs_lines), 6))
        obstacles[:, :4] = [[start_x, start_y, end_x, end_y]
                            for start_x, start_y, end_x, end_y in obs_lines]
        obstacles[:, 4:] = ortho_left
        return obstacles
