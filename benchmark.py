from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
import scalene.scalene_profiler as scalene_profiler
import pysocialforce as psf


@dataclass
class SimConfig:
    sim_steps: int
    initial_state: np.ndarray
    groups: List[List[int]]
    obstacles: List[Tuple[float, float, float, float]]
    # config: psf.utils.config.DefaultConfig = psf.utils.config.DefaultConfig()


@dataclass
class SimSettings:
    map_width: float
    map_height: float
    sim_steps: int
    num_peds: int
    num_groups: float
    group_cov: List[List[float]] # shape (2, 2)
    num_obstacles: int

    def sample(self) -> SimConfig:
        spawned_peds = 0
        group_centroids = self.rand_2d_coords(self.num_groups)
        group_goals = self.rand_2d_coords(self.num_groups)
        individual_dynamics = np.random.normal(size=(self.num_peds, 2))
        avg_peds_on_group = self.num_peds / self.num_groups

        individuals = np.zeros((self.num_peds, 6))
        groups = []

        for gid in range(self.num_groups):
            if spawned_peds > self.num_groups - gid:
                num_peds_on_group = 1
            elif gid == self.num_groups - 1:
                num_peds_on_group = self.num_peds - spawned_peds
            else:
                num_peds_on_group = round(np.random.normal(avg_peds_on_group))

            peds_pos_of_group = np.random.multivariate_normal(
                group_centroids[gid], self.group_cov, num_peds_on_group)
            groups.append(list(range(spawned_peds, spawned_peds + num_peds_on_group)))

            for i, ped_pos in enumerate(peds_pos_of_group):
                ped_id = spawned_peds + i
                ped_dyn = individual_dynamics[ped_id]
                goal = group_goals[gid]
                ped_features = [ped_pos[0], ped_pos[1], ped_dyn[0], ped_dyn[1], goal[0], goal[1]]
                individuals[ped_id] = ped_features
            spawned_peds += num_peds_on_group

        obs_start = self.rand_2d_coords(self.num_obstacles)
        obs_end = self.rand_2d_coords(self.num_obstacles)
        obstacles = [[s_x, s_y, e_x, e_y] for ((s_x, s_y), (e_x, e_y)) in zip(obs_start, obs_end)]

        config = SimConfig(self.sim_steps, individuals, groups, obstacles)
        return config

    def rand_2d_coords(self, num_coords: int) -> List[Tuple[float, float]]:
        x = np.random.uniform(0, self.map_width, num_coords)
        y = np.random.uniform(0, self.map_height, num_coords)
        return list(zip(x, y))


def simulate(config: SimConfig):
    s = psf.Simulator(config.initial_state, config.groups, config.obstacles)
    s.step(config.sim_steps)


def benchmark():
    num_simulations = 100
    settings = SimSettings(
        map_width=80,
        map_height=80,
        sim_steps=100,
        num_peds=25,
        num_groups=3,
        group_cov=[[1, 0], [0, 1]],
        num_obstacles=10)
    configs = [settings.sample() for _ in range(num_simulations)]

    scalene_profiler.start()
    for sim_id, config in enumerate(configs):
        print(f'simulating {sim_id+1} / {num_simulations}')
        simulate(config)
    scalene_profiler.stop()


if __name__ == '__main__':
    benchmark()
