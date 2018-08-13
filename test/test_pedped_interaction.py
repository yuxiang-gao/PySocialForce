import numpy as np
import pytest
import socialforce


def test_rab():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    s = socialforce.Simulator(initial_state)
    assert s.rab().tolist() == [[
        [0.0, 0.0],
        [-1.0, 0.0],
    ], [
        [1.0, 0.0],
        [0.0, 0.0],
    ]]


def test_fab():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    s = socialforce.Simulator(initial_state)
    force_at_unit_distance = 0.25  # TODO confirm
    assert s.fab() == pytest.approx(np.array([[
        [0.0, 0.0],
        [-force_at_unit_distance, 0.0],
    ], [
        [force_at_unit_distance, 0.0],
        [0.0, 0.0],
    ]]), abs=0.05)


def test_b_zero_vel():
    initial_state = np.array([
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    s = socialforce.Simulator(initial_state)
    assert s.V.b(s.rab(), s.speeds(), s.desired_directions()).tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]


def test_w():
    initial_state = np.array([
        [0.0, 0.0, 0.5, 0.5, 10.0, 10.0],
        [10.0, 0.3, -0.5, 0.5, 0.0, 10.0],
    ])
    s = socialforce.Simulator(initial_state)
    w = s.w(s.desired_directions(), -s.fab())
    assert w.tolist() == [
        [0, 1],
        [1, 0],
    ]
