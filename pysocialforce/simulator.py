# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce import forces
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import Scene

# from pysocialforce.utils import timeit


class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
       Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, [tau])
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    step_width : Double
        Simulation time step, Default: 0.4
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """

    def __init__(self, state, groups=None, obstacles=None, config_file=None):
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file)

        self.scene = Scene(state, groups, obstacles, self.config)
        self.peds = self.scene.peds

        self.step_width = self.config("step_width") or 0.4

        self.forces = [
            forces.GoalAttractiveForce(),
            forces.PedRepulsiveForce(),
            forces.SpaceRepulsiveForce(),
        ]
        group_forces = [
            forces.GroupCoherenceForce(),
            forces.GroupRepulsiveForce(),
            forces.GroupGazeForce(),
        ]

        if self.config("enable_group"):
            self.forces += group_forces

        # initiate forces
        for force in self.forces:
            force.init(self.scene, self.config)

    def step_once(self):
        """Step."""
        # social forces
        sum_forces = sum(map(lambda x: x.get_force(), self.forces))

        # update state
        self.peds.step(sum_forces)

        return self

    def step(self, n=1):
        for _ in range(n):
            self.step_once()
        return self
