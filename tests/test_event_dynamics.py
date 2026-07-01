# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for event-driven hard-sphere/disc dynamics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jdem
from jaxdem.utils import (
    EVENT_NONE,
    EVENT_PAIR,
    EVENT_WALL,
    event_corrected_rollout,
    event_corrected_step,
    event_rollout,
    event_step,
    validate_event_state,
)


def _sphere_system(pos, vel, rad=None, mass=None, *, domain_type="free", domain_kw=None):
    pos = jnp.asarray(pos, dtype=float)
    n = pos.shape[0]
    state = jdem.State.create(
        pos=pos,
        vel=jnp.asarray(vel, dtype=float),
        rad=jnp.ones(n, dtype=float) if rad is None else jnp.asarray(rad, dtype=float),
        mass=jnp.ones(n, dtype=float) if mass is None else jnp.asarray(mass, dtype=float),
    )
    system = jdem.System.create(
        state.shape,
        domain_type=domain_type,
        domain_kw=domain_kw,
        linear_integrator_type="",
        rotation_integrator_type="",
        collider_type="",
    )
    return state, system


def test_equal_mass_pair_collision_swaps_velocities():
    state, system = _sphere_system(
        [[0.0, 0.0], [3.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )
    validate_event_state(state, system)

    result = event_step(state, system)

    assert int(result.event.event_type) == EVENT_PAIR
    assert int(result.event.i) == 0
    assert int(result.event.j) == 1
    assert float(result.event.time) == pytest.approx(0.5)
    assert bool(jnp.allclose(result.state.pos, jnp.array([[0.5, 0.0], [2.5, 0.0]])))
    assert bool(jnp.allclose(result.state.vel, jnp.array([[-1.0, 0.0], [1.0, 0.0]])))


def test_unequal_mass_inelastic_pair_collision_matches_analytic_solution():
    state, system = _sphere_system(
        [[0.0, 0.0], [3.0, 0.0]],
        [[2.0, 0.0], [0.0, 0.0]],
        mass=[1.0, 3.0],
    )

    result = event_step(state, system, restitution=0.5)

    assert int(result.event.event_type) == EVENT_PAIR
    assert float(result.event.time) == pytest.approx(0.5)
    assert bool(jnp.allclose(result.state.vel[:, 0], jnp.array([-0.25, 0.75])))
    assert bool(jnp.allclose(result.state.vel[:, 1], jnp.zeros(2)))


def test_no_event_advances_by_max_dt():
    state, system = _sphere_system(
        [[0.0, 0.0], [5.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    result = event_step(state, system, max_dt=2.0)

    assert int(result.event.event_type) == EVENT_NONE
    assert not bool(result.event.hit)
    assert float(result.event.time) == pytest.approx(2.0)
    assert bool(jnp.allclose(result.state.pos, jnp.array([[2.0, 0.0], [7.0, 0.0]])))


def test_periodic_pair_collision_uses_image_displacement():
    state, system = _sphere_system(
        [[1.0, 0.0], [8.0, 0.0]],
        [[-1.0, 0.0], [1.0, 0.0]],
        domain_type="periodic",
        domain_kw={"box_size": jnp.array([10.0, 10.0])},
    )

    result = event_step(state, system)

    assert int(result.event.event_type) == EVENT_PAIR
    assert float(result.event.time) == pytest.approx(0.5)
    assert bool(jnp.allclose(result.state.vel, jnp.array([[1.0, 0.0], [-1.0, 0.0]])))


def test_reflective_wall_collision_uses_wall_restitution():
    state, system = _sphere_system(
        [[1.5, 5.0]],
        [[-1.0, 0.0]],
        domain_type="reflectsphere",
        domain_kw={
            "box_size": jnp.array([10.0, 10.0]),
            "restitution_coefficient": 0.25,
        },
    )

    result = event_step(state, system)

    assert int(result.event.event_type) == EVENT_WALL
    assert int(result.event.i) == 0
    assert int(result.event.axis) == 0
    assert int(result.event.j) == 0
    assert float(result.event.time) == pytest.approx(0.5)
    assert bool(jnp.allclose(result.state.vel, jnp.array([[0.25, 0.0]])))


def test_event_step_and_rollout_are_jittable():
    state, system = _sphere_system(
        [[0.0, 0.0], [3.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )

    result = jax.jit(event_step)(state, system)
    assert int(result.event.event_type) == EVENT_PAIR

    final_state, final_system, traj = event_rollout(
        state,
        system,
        n_events=2,
        max_dt_per_event=1.0,
    )
    assert final_state.pos.shape == state.pos.shape
    assert final_system.time.shape == system.time.shape
    assert traj[2].event_type.shape[0] == 2


def test_event_corrected_step_matches_normal_step_without_predicted_event():
    state, system = _sphere_system(
        [[0.0, 0.0], [5.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    system = jdem.System.create(
        state.shape,
        dt=0.25,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        collider_type="",
    )

    normal_state, normal_system = system.step(state, system)
    result = event_corrected_step(state, system)

    assert int(result.correction.n_substeps) == 1
    assert not bool(result.correction.hit)
    assert bool(jnp.allclose(result.state.pos, normal_state.pos))
    assert float(result.system.time) == pytest.approx(float(normal_system.time))


def test_event_corrected_step_subdivides_after_predicted_contact():
    state, system = _sphere_system(
        [[0.0, 0.0], [3.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    )
    system = jdem.System.create(
        state.shape,
        dt=1.0,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        collider_type="",
    )

    result = event_corrected_step(state, system, max_substeps=4)

    assert bool(result.correction.hit)
    assert int(result.correction.n_substeps) == 4
    assert float(result.correction.min_substep_dt) == pytest.approx(1.0 / 6.0)
    assert float(result.system.time) == pytest.approx(1.0)


def test_event_corrected_rollout_is_jittable_and_accepts_clumps():
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        vel=jnp.array([[0.1, 0.0], [0.1, 0.0]]),
        rad=jnp.array([0.25, 0.25]),
        clump_id=jnp.zeros(2, dtype=int),
    )
    system = jdem.System.create(
        state.shape,
        dt=0.1,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        collider_type="",
    )

    final_state, final_system, traj = event_corrected_rollout(
        state,
        system,
        n_steps=2,
    )

    assert final_state.pos.shape == state.pos.shape
    assert final_system.time.shape == system.time.shape
    assert traj[2].n_substeps.shape[0] == 2


def test_validation_rejects_clumps_bonds_facets_and_overlaps():
    state, system = _sphere_system(
        [[0.0, 0.0], [0.5, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="overlapping"):
        validate_event_state(state, system)

    clump_state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [3.0, 0.0]]),
        rad=jnp.ones(2),
        clump_id=jnp.zeros(2, dtype=int),
    )
    with pytest.raises(ValueError, match="independent"):
        validate_event_state(clump_state, system)

    bonded_state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [3.0, 0.0]]),
        rad=jnp.ones(2),
        bond_id=jnp.array([[1], [0]]),
    )
    with pytest.raises(ValueError, match="bonded"):
        validate_event_state(bonded_state, system)
