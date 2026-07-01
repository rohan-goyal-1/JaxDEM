# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Event-driven hard-sphere/disc dynamics utilities.

This module provides an explicit EDMD path for independent spherical
particles. It deliberately does not alter :meth:`jaxdem.System.step`, whose
contract is fixed-step, force-based DEM integration.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..colliders import valid_interaction_mask

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


EVENT_NONE = 0
EVENT_PAIR = 1
EVENT_WALL = 2


class Event(NamedTuple):
    """Description of the event selected by :func:`event_step`.

    ``event_type`` is one of ``EVENT_NONE``, ``EVENT_PAIR``, or
    ``EVENT_WALL``. For wall events, ``j`` stores the wall side
    (``0`` lower, ``1`` upper) and ``axis`` stores the wall axis.
    """

    time: jax.Array
    event_type: jax.Array
    i: jax.Array
    j: jax.Array
    axis: jax.Array
    hit: jax.Array


class EventStepResult(NamedTuple):
    """Single event-step result."""

    state: "State"
    system: "System"
    event: Event


def _normalize_type_name(x: Any) -> str:
    return str(x).replace(" ", "").replace("_", "").replace("-", "").lower()


def validate_event_state(
    state: State,
    system: System,
    *,
    overlap_tol: float = 1e-10,
) -> None:
    """Validate that ``state``/``system`` are compatible with v1 EDMD.

    V1 supports independent hard spheres/discs only. Rigid clumps,
    bonded/deformable bodies, facets, and initially overlapping particles are
    rejected with a descriptive ``ValueError``.
    """

    domain_key = _normalize_type_name(system.domain.type_name)
    if domain_key not in {"free", "periodic", "reflect", "reflectsphere"}:
        raise ValueError(
            "event dynamics supports only free, periodic, reflect, and "
            f"reflectsphere domains; got {system.domain.type_name!r}."
        )

    n = int(state.N)
    clump_id = np.asarray(state.clump_id)
    if np.unique(clump_id).size != n:
        raise ValueError("event dynamics v1 supports independent spheres only.")

    bond_id = np.asarray(state.bond_id)
    if bond_id.size and np.any(bond_id >= 0):
        raise ValueError("event dynamics v1 does not support bonded particles.")

    facet_vertices = np.asarray(state.facet_vertices)
    if facet_vertices.size and np.any(facet_vertices >= 0):
        raise ValueError("event dynamics v1 does not support facets.")

    pos_p = np.asarray(state.pos_p)
    if pos_p.size and not np.allclose(pos_p, 0.0, atol=overlap_tol, rtol=0.0):
        raise ValueError("event dynamics v1 supports only center-based spheres.")

    pos = jnp.asarray(state.pos)
    dr = pos[:, None, :] - pos[None, :, :]
    if system.domain.periodic:
        dr = dr - system.domain.box_size * jnp.round(dr / system.domain.box_size)
    dist_sq = jnp.sum(dr * dr, axis=-1)
    rad_sum = state.rad[:, None] + state.rad[None, :]
    iota = jnp.arange(n)
    pair_mask = iota[:, None] < iota[None, :]
    overlap_limit = jnp.maximum(rad_sum - overlap_tol, 0.0)
    overlaps = pair_mask & (dist_sq < overlap_limit * overlap_limit)
    if bool(np.asarray(jnp.any(overlaps))):
        raise ValueError("event dynamics initial state contains overlapping spheres.")


def _active_velocity(state: State) -> jax.Array:
    return jnp.where(state.fixed[..., None], 0.0, state.vel)


def _image_shifts(dim: int, periodic: bool) -> jax.Array:
    if not periodic:
        return jnp.zeros((1, dim), dtype=int)
    r = jnp.arange(-1, 2, dtype=int)
    mesh = jnp.meshgrid(*([r] * dim), indexing="ij")
    return jnp.stack([m.ravel() for m in mesh], axis=1)


@jax.jit(inline=True)
def _solve_collision_time(
    r: jax.Array,
    v: jax.Array,
    radius: jax.Array,
    min_dt: jax.Array,
    overlap_tol: jax.Array,
) -> jax.Array:
    a = jnp.sum(v * v, axis=-1)
    b = 2.0 * jnp.sum(r * v, axis=-1)
    c = jnp.sum(r * r, axis=-1) - radius * radius
    disc = b * b - 4.0 * a * c
    valid = (a > 0.0) & (b < 0.0) & (disc >= 0.0) & (c >= -overlap_tol)
    safe_disc = jnp.where(disc > 0.0, disc, 1.0)
    sqrt_disc = jnp.where(disc > 0.0, jnp.sqrt(safe_disc), 0.0)
    denom = jnp.where(a > 0.0, 2.0 * a, 1.0)
    t = (-b - sqrt_disc) / denom
    valid = valid & (t > min_dt)
    return jnp.where(valid, t, jnp.inf)


def _pair_event(state: State, system: System, min_dt: jax.Array, overlap_tol: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    pos = state.pos
    vel = _active_velocity(state)
    n = state.N

    shifts = _image_shifts(state.dim, bool(system.domain.periodic))
    dr = pos[:, None, :] - pos[None, :, :]
    if system.domain.periodic:
        dr = dr[None, :, :, :] - shifts[:, None, None, :] * system.domain.box_size
    else:
        dr = dr[None, :, :, :]
    dv = vel[:, None, :] - vel[None, :, :]
    radius = state.rad[:, None] + state.rad[None, :]

    times = _solve_collision_time(
        dr,
        dv[None, :, :, :],
        radius[None, :, :],
        min_dt,
        overlap_tol,
    )
    pair_times = jnp.min(times, axis=0)

    iota = jnp.arange(n, dtype=int)
    valid_by_i = jax.vmap(
        lambda ci, bi: valid_interaction_mask(
            ci, state.clump_id, bi, iota, system.interact_same_bond_id
        )
    )(state.clump_id, state.bond_id)
    valid_pairs = (
        (iota[:, None] < iota[None, :])
        & (valid_by_i > 0)
        & ~(state.fixed[:, None] & state.fixed[None, :])
    )
    pair_times = jnp.where(valid_pairs, pair_times, jnp.inf)

    flat_idx = jnp.argmin(pair_times.reshape(-1))
    pair_time = pair_times.reshape(-1)[flat_idx]
    hit = jnp.isfinite(pair_time)
    pair_i = jnp.where(hit, flat_idx // n, -1)
    pair_j = jnp.where(hit, flat_idx % n, -1)
    return pair_time, pair_i, pair_j


def _wall_event(
    state: State,
    system: System,
    min_dt: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if not hasattr(system.domain, "restitution_coefficient"):
        return (
            jnp.asarray(jnp.inf, dtype=state.pos.dtype),
            jnp.asarray(-1, dtype=int),
            jnp.asarray(-1, dtype=int),
            jnp.asarray(-1, dtype=int),
        )

    pos = state.pos
    vel = _active_velocity(state)
    dim = state.dim
    lo = system.domain.anchor + state.rad[:, None]
    hi = system.domain.anchor + system.domain.box_size - state.rad[:, None]

    t_lo = (lo - pos) / jnp.where(vel < 0.0, vel, -1.0)
    t_hi = (hi - pos) / jnp.where(vel > 0.0, vel, 1.0)
    free = ~state.fixed[:, None]
    t_lo = jnp.where((vel < 0.0) & free & (t_lo > min_dt), t_lo, jnp.inf)
    t_hi = jnp.where((vel > 0.0) & free & (t_hi > min_dt), t_hi, jnp.inf)

    wall_times = jnp.stack([t_lo, t_hi], axis=-1)
    flat_idx = jnp.argmin(wall_times.reshape(-1))
    wall_time = wall_times.reshape(-1)[flat_idx]
    hit = jnp.isfinite(wall_time)
    wall_i = flat_idx // (2 * dim)
    rem = flat_idx - wall_i * (2 * dim)
    wall_axis = rem // 2
    wall_side = rem % 2
    return (
        wall_time,
        jnp.where(hit, wall_i, -1),
        jnp.where(hit, wall_axis, -1),
        jnp.where(hit, wall_side, -1),
    )


def _select_pair_value(value: Any, state: State, system: System, i: jax.Array, j: jax.Array) -> jax.Array:
    if value is None:
        if hasattr(system.mat_table, "e_eff"):
            arr = system.mat_table.e_eff
        else:
            arr = jnp.asarray(1.0, dtype=state.pos.dtype)
    else:
        arr = jnp.asarray(value, dtype=state.pos.dtype)
    if arr.ndim == 0:
        return arr
    return arr[state.mat_id[i], state.mat_id[j]]


def _select_wall_value(value: Any, state: State, i: jax.Array) -> jax.Array:
    arr = jnp.asarray(value, dtype=state.pos.dtype)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        return arr[state.mat_id[i]]
    return arr[state.mat_id[i], state.mat_id[i]]


def _default_wall_restitution(restitution: Any, system: System) -> Any:
    if hasattr(system.domain, "restitution_coefficient"):
        return system.domain.restitution_coefficient
    return 1.0 if restitution is None else restitution


def _advance_ballistic(
    state: State, system: System, dt: jax.Array
) -> tuple[State, System]:
    state.pos_c += dt * _active_velocity(state)
    if system.domain.periodic:
        state, system = system.domain.shift(state, system)
    state.force *= 0.0
    state.torque *= 0.0
    return state, system


def _apply_pair_impulse(
    state: State,
    system: System,
    i: jax.Array,
    j: jax.Array,
    restitution: Any,
    active: jax.Array,
) -> tuple[State, System]:
    safe_i = jnp.maximum(i, 0)
    safe_j = jnp.maximum(j, 0)
    pos = state.pos
    vel = _active_velocity(state)
    rij = system.domain._displacement(pos[safe_i], pos[safe_j], system)
    dist_sq = jnp.sum(rij * rij)
    inv_dist = jnp.where(dist_sq > 0.0, jax.lax.rsqrt(dist_sq), 0.0)
    normal = rij * inv_dist

    inv_mass_i = jnp.where(state.fixed[safe_i], 0.0, 1.0 / state.mass[safe_i])
    inv_mass_j = jnp.where(state.fixed[safe_j], 0.0, 1.0 / state.mass[safe_j])
    inv_mass_sum = inv_mass_i + inv_mass_j
    v_rel_n = jnp.dot(vel[safe_i] - vel[safe_j], normal)
    e = _select_pair_value(restitution, state, system, safe_i, safe_j)
    impulse_mag = -(1.0 + e) * v_rel_n / jnp.where(inv_mass_sum > 0.0, inv_mass_sum, 1.0)
    impulse_mag = jnp.where(active & (inv_mass_sum > 0.0) & (v_rel_n < 0.0), impulse_mag, 0.0)
    impulse = impulse_mag * normal

    state.vel = state.vel.at[safe_i].add(inv_mass_i * impulse)
    state.vel = state.vel.at[safe_j].add(-inv_mass_j * impulse)
    return state, system


def _apply_wall_impulse(
    state: State,
    system: System,
    i: jax.Array,
    axis: jax.Array,
    wall_restitution: Any,
    active: jax.Array,
) -> tuple[State, System]:
    safe_i = jnp.maximum(i, 0)
    safe_axis = jnp.maximum(axis, 0)
    e = _select_wall_value(wall_restitution, state, safe_i)
    v_axis = state.vel[safe_i, safe_axis]
    dv = jnp.where(active & ~state.fixed[safe_i], -(1.0 + e) * v_axis, 0.0)
    state.vel = state.vel.at[safe_i, safe_axis].add(dv)
    return state, system


def event_step(
    state: State,
    system: System,
    *,
    restitution: Any = 1.0,
    wall_restitution: Any = None,
    max_dt: float | jax.Array = jnp.inf,
    min_dt: float | jax.Array = 1e-12,
    overlap_tol: float | jax.Array = 1e-10,
) -> EventStepResult:
    """Advance to the next hard-sphere event or by ``max_dt``.

    The function is JAX-transformable but does not call
    :func:`validate_event_state`; call validation once on the host before a
    rollout when accepting user-supplied states.
    """

    max_dt_arr = jnp.asarray(max_dt, dtype=state.pos.dtype)
    min_dt_arr = jnp.asarray(min_dt, dtype=state.pos.dtype)
    overlap_tol_arr = jnp.asarray(overlap_tol, dtype=state.pos.dtype)

    pair_time, pair_i, pair_j = _pair_event(state, system, min_dt_arr, overlap_tol_arr)
    wall_time, wall_i, wall_axis, wall_side = _wall_event(state, system, min_dt_arr)

    next_time = jnp.minimum(pair_time, wall_time)
    hit = jnp.isfinite(next_time) & (next_time <= max_dt_arr)
    dt = jnp.where(
        hit,
        next_time,
        jnp.where(jnp.isfinite(max_dt_arr), max_dt_arr, jnp.asarray(0.0, dtype=state.pos.dtype)),
    )
    is_pair = hit & (pair_time <= wall_time)
    is_wall = hit & ~is_pair

    state, system = _advance_ballistic(state, system, dt)
    state, system = _apply_pair_impulse(state, system, pair_i, pair_j, restitution, is_pair)
    wall_value = _default_wall_restitution(restitution, system) if wall_restitution is None else wall_restitution
    state, system = _apply_wall_impulse(state, system, wall_i, wall_axis, wall_value, is_wall)

    system = dataclasses.replace(
        system,
        time=system.time + dt,
        step_count=system.step_count + 1,
    )

    none_i = jnp.asarray(-1, dtype=int)
    event = Event(
        time=dt,
        event_type=jnp.where(hit, jnp.where(is_pair, EVENT_PAIR, EVENT_WALL), EVENT_NONE),
        i=jnp.where(hit, jnp.where(is_pair, pair_i, wall_i), none_i),
        j=jnp.where(hit, jnp.where(is_pair, pair_j, wall_side), none_i),
        axis=jnp.where(hit & is_wall, wall_axis, none_i),
        hit=hit,
    )
    return EventStepResult(state=state, system=system, event=event)


@jax.jit(static_argnames=("n_events", "unroll"))
def event_rollout(
    state: State,
    system: System,
    *,
    n_events: int,
    restitution: Any = 1.0,
    wall_restitution: Any = None,
    max_dt_per_event: float | jax.Array = jnp.inf,
    min_dt: float | jax.Array = 1e-12,
    overlap_tol: float | jax.Array = 1e-10,
    unroll: int = 2,
) -> tuple[State, System, tuple[State, System, Event]]:
    """Run ``n_events`` event-driven steps and collect the trajectory."""

    def body(
        carry: tuple[State, System], _: None
    ) -> tuple[tuple[State, System], tuple[State, System, Event]]:
        st, sys = carry
        result = event_step(
            st,
            sys,
            restitution=restitution,
            wall_restitution=wall_restitution,
            max_dt=max_dt_per_event,
            min_dt=min_dt,
            overlap_tol=overlap_tol,
        )
        return (result.state, result.system), (result.state, result.system, result.event)

    (state, system), traj = jax.lax.scan(
        body, (state, system), xs=None, length=n_events, unroll=unroll
    )
    return state, system, traj


__all__ = [
    "EVENT_NONE",
    "EVENT_PAIR",
    "EVENT_WALL",
    "Event",
    "EventStepResult",
    "event_rollout",
    "event_step",
    "validate_event_state",
]
