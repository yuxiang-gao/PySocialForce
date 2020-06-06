# coding=utf-8

"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np

from .potentials import PedPedPotential
from .fieldofview import FieldOfView
from .forces import *
from . import stateutils
import toml


class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (x, y, v_x, v_y, d_x, d_y, [tau]).
    tau is optional in this vector.
    
    group_state 
    each group is represented by a list of indices
    """

    def __init__(self, state, groups=None, space=None, config_file="config.toml"):
        self.state = state
        self.groups = groups
        self.space = space
        self.config = toml.load(config_file)

        self.time_step = self.config.get("time_step") or 0.4

        self.initial_speeds = stateutils.speeds(self.state)
        self.max_speeds = self.config.get("max_speed_multiplier") * self.initial_speeds

        if self.state.shape[1] < 7:
            tau = self.config.get("tau") or 0.5
            if not hasattr(tau, "shape"):
                tau = tau * np.ones(self.state.shape[0])
            self.state = np.concatenate((self.state, np.expand_dims(tau, -1)), axis=-1)

        self.forces = [
            GoalAttractiveForce(),
            PedRepulsiveForce(),
            SpaceRepulsiveForce(),
        ]
        group_forces = [
            GroupCoherenceForce(),
            GroupRepulsiveForce(),
            GroupGazeForce(),
        ]
        if self.config.get("enable_group"):
            self.forces += group_forces

        # initiate forces
        for force in self.forces:
            force.load_config(self.config)
            force.set_state(
                self.state,
                groups=self.groups,
                space=self.space,
                initial_speeds=self.initial_speeds,
            )

    def capped_velocity(self, desired_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        """Step"""
        # social forces
        forces = sum(map(lambda x: x.get_force(), self.forces))
        # desired velocity
        desired_velocity = self.state[:, 2:4] + self.time_step * forces
        desired_velocity = self.capped_velocity(desired_velocity)

        # update state
        self.state[:, 0:2] += desired_velocity * self.time_step
        self.state[:, 2:4] = desired_velocity

        # Update states
        for force in self.forces:
            force.set_state(self.state, groups=self.groups, space=self.space)

        return self
