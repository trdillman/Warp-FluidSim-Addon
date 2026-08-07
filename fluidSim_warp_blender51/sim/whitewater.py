"""Whitewater (foam/spray/bubble) simulation kernels.

Ports of the fluid_whitewater group + fluidSimAdvectWhiteWater:
  fluidSimComputeChurn, fluidSimEmitWhiteWater, fluidSimWhiteWaterSetGridData,
  fluidSimAdvectWhiteWater
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .fields import (
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_center_world,
    world_to_voxel,
    sample_mac_velocity,
    sample_scalar_trilinear,
)
from .particles import k_clamp_count
from .registry import register_fn


WHITEWATER_POLICY_FIXED_CAPACITY = "FIXED_CAPACITY_DETERMINISTIC_DROPS"
WHITEWATER_STATUS_OK = "OK"
WHITEWATER_STATUS_CAPACITY_EXHAUSTED = "CAPACITY_EXHAUSTED"
WHITEWATER_STATUS_SDF_UNAVAILABLE = "SDF_DEPENDENCY_UNAVAILABLE"


def normalized_life_range(life_min, life_max):
    """Return a non-negative, ordered lifetime interval."""
    minimum = max(0.0, float(life_min))
    return minimum, max(minimum, float(life_max))


def deterministic_admission(candidate_flags, *, live_count, free_count, capacity):
    """Pure reference for the fixed-capacity, liquid-tid admission contract."""
    winners = tuple(index for index, flag in enumerate(candidate_flags) if flag)
    required = len(winners)
    free = max(0, min(int(free_count), int(capacity)))
    tail = max(0, int(capacity) - max(0, int(live_count)))
    available = min(int(capacity), free + tail)
    admitted = min(required, available)
    return {
        "policy": WHITEWATER_POLICY_FIXED_CAPACITY,
        "required": required,
        "available": available,
        "admitted": admitted,
        "dropped": required - admitted,
        "winner_tids": winners[:admitted],
    }


# ---------------------------------------------------------------------------
# Churn: measure of local turbulence used to drive whitewater emission.
# ---------------------------------------------------------------------------


@wp.kernel
def k_compute_churn(
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    dt: float,
    decay: float,
    gain: float,
    churn: wp.array(dtype=float),
):
    """Churn accumulates from local velocity variance (curl magnitude +
    divergence of direction) and decays over time.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    old = churn[tid] * wp.exp(-decay * dt)
    if (vt & C.VOXEL_FLUID) == 0 or (vt & C.VOXEL_COLLISION) != 0:
        churn[tid] = old
        return

    ijk = voxel_coord_of_flat(leaf_coords, tid)
    h = voxel_size

    # Central differences of the cell-averaged velocity field.
    c_vel = wp.vec3(u[tid], v[tid], w[tid])
    dudy = float(0.0)
    dudz = float(0.0)
    dvdx = float(0.0)
    dvdz = float(0.0)
    dwdx = float(0.0)
    dwdy = float(0.0)
    variance = float(0.0)
    n_count = float(0.0)

    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            flat_n = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
            if flat_n < 0:
                continue
            if (voxel_type[flat_n] & C.VOXEL_FLUID) == 0:
                continue
            n_vel = wp.vec3(u[flat_n], v[flat_n], w[flat_n])
            d = n_vel - c_vel
            variance += wp.length_sq(d)
            n_count += 1.0
            fs = float(step)
            if axis == 0:
                dvdx += d[1] * fs / (2.0 * h)
                dwdx += d[2] * fs / (2.0 * h)
            elif axis == 1:
                dudy += d[0] * fs / (2.0 * h)
                dwdy += d[2] * fs / (2.0 * h)
            else:
                dudz += d[0] * fs / (2.0 * h)
                dvdz += d[1] * fs / (2.0 * h)

    curl = wp.vec3(dwdy - dvdz, dudz - dwdx, dvdx - dudy)
    turb = wp.length(curl) * h
    if n_count > 0.0:
        turb += wp.sqrt(variance / n_count)
    churn[tid] = old + gain * turb * dt


register_fn(k_compute_churn, "fluidSimComputeChurn",
            notes="Curl + neighbor velocity variance, exponential decay.")


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


@wp.kernel
def k_mark_whitewater_candidates(
    liquid_positions: wp.array(dtype=wp.vec3),
    liquid_velocities: wp.array(dtype=wp.vec3),
    liquid_count: wp.array(dtype=wp.int32),
    churn: wp.array(dtype=float),
    surface_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    churn_threshold: float,
    emission_rate: float,
    surface_band: float,
    min_speed: float,
    seed: int,
    candidates: wp.array(dtype=wp.int32),
):
    """Mark probability-passing candidates without mutating the output pool."""
    tid = wp.tid()
    candidates[tid] = 0
    if tid >= liquid_count[0]:
        return
    p = liquid_positions[tid]
    vel = liquid_velocities[tid]
    if wp.length(vel) < min_speed:
        return
    vp = world_to_voxel(p, origin, voxel_size)
    flat = voxel_flat_index(
        leaf_table, voxels_dim, leaves_dim,
        wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2]))),
    )
    if flat < 0:
        return
    c = churn[flat]
    if c < churn_threshold:
        return
    sdf = sample_scalar_trilinear(
        surface_sdf, leaf_table, voxels_dim, leaves_dim, vp, 1.0e6
    )
    if wp.abs(sdf) > surface_band:
        return
    state = wp.rand_init(seed, tid)
    prob = wp.clamp((c - churn_threshold) * emission_rate * dt, 0.0, 1.0)
    if wp.randf(state) <= prob:
        candidates[tid] = 1


@wp.kernel
def k_prepare_whitewater_admission(
    candidates: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    candidate_count: int,
    ww_count: wp.array(dtype=wp.int32),
    ww_free_count: wp.array(dtype=wp.int32),
    ww_capacity: int,
    plan: wp.array(dtype=wp.int32),
    diagnostics: wp.array(dtype=wp.int32),
):
    """Count candidates and publish a device-only deterministic admission plan."""
    required = offsets[candidate_count - 1] + candidates[candidate_count - 1]
    live_before = ww_count[0]
    if live_before < 0:
        live_before = 0
    if live_before > ww_capacity:
        live_before = ww_capacity
    free_before = ww_free_count[0]
    if free_before < 0:
        free_before = 0
    if free_before > ww_capacity:
        free_before = ww_capacity
    available = free_before + (ww_capacity - live_before)
    if available > ww_capacity:
        available = ww_capacity
    admitted = required
    if admitted > available:
        admitted = available
    plan[0] = required
    plan[1] = admitted
    plan[2] = free_before
    plan[3] = live_before
    diagnostics[0] = diagnostics[0] + required
    diagnostics[1] = diagnostics[1] + admitted
    diagnostics[2] = diagnostics[2] + (required - admitted)
    diagnostics[3] = diagnostics[3] + available


@wp.kernel
def k_commit_whitewater_candidates(
    liquid_positions: wp.array(dtype=wp.vec3),
    liquid_velocities: wp.array(dtype=wp.vec3),
    candidates: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    voxel_size: float,
    life_min: float,
    life_max: float,
    seed: int,
    ww_free_slots: wp.array(dtype=wp.int32),
    plan: wp.array(dtype=wp.int32),
    ww_positions: wp.array(dtype=wp.vec3),
    ww_velocities: wp.array(dtype=wp.vec3),
    ww_age: wp.array(dtype=float),
    ww_life: wp.array(dtype=float),
    ww_alive: wp.array(dtype=wp.int32),
):
    """Commit ascending-liquid-tid winners to free slots, then the pool tail."""
    tid = wp.tid()
    if candidates[tid] == 0:
        return
    rank = offsets[tid]
    if rank >= plan[1]:
        return
    free_before = plan[2]
    slot = plan[3] + rank - free_before
    if rank < free_before:
        slot = ww_free_slots[free_before - 1 - rank]

    # Reconsume the predicate draw so jitter/lifetime retain the prior stream.
    state = wp.rand_init(seed, tid)
    wp.randf(state)
    jitter = wp.vec3(
        (wp.randf(state) - 0.5) * voxel_size,
        (wp.randf(state) - 0.5) * voxel_size,
        (wp.randf(state) - 0.5) * voxel_size,
    )
    ww_positions[slot] = liquid_positions[tid] + jitter
    ww_velocities[slot] = liquid_velocities[tid]
    ww_age[slot] = 0.0
    ww_life[slot] = life_min + wp.randf(state) * (life_max - life_min)
    ww_alive[slot] = 1


@wp.kernel
def k_finalize_whitewater_admission(
    plan: wp.array(dtype=wp.int32),
    ww_count: wp.array(dtype=wp.int32),
    ww_free_count: wp.array(dtype=wp.int32),
):
    admitted = plan[1]
    free_used = admitted
    if free_used > plan[2]:
        free_used = plan[2]
    ww_free_count[0] = plan[2] - free_used
    ww_count[0] = plan[3] + admitted - free_used


register_fn(k_mark_whitewater_candidates, "fluidSimMarkWhiteWaterCandidates",
            notes="Deterministic probability candidate marking in liquid tid order.")
register_fn(k_commit_whitewater_candidates, "fluidSimEmitWhiteWater",
            notes="Fixed-capacity deterministic admission with explicit drops.")


# ---------------------------------------------------------------------------
# Advection: foam (on surface), spray (ballistic), bubbles (drag toward fluid)
# ---------------------------------------------------------------------------


@wp.kernel
def k_advect_whitewater(
    ww_positions: wp.array(dtype=wp.vec3),
    ww_velocities: wp.array(dtype=wp.vec3),
    ww_age: wp.array(dtype=float),
    ww_life: wp.array(dtype=float),
    ww_alive: wp.array(dtype=wp.int32),
    ww_count: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww_weight: wp.array(dtype=float),
    surface_sdf: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    gravity: wp.vec3,
    foam_band: float,
    bubble_drag: float,
    spray_drag: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    ww_capacity: int,
    ww_free_slots: wp.array(dtype=wp.int32),
    ww_free_count: wp.array(dtype=wp.int32),
):
    """Advect whitewater by regime:
      - inside liquid (sdf < -band): bubble -> dragged to fluid velocity + buoyancy
      - near surface (|sdf| <= band): foam -> follows fluid velocity
      - outside (sdf > band): spray -> ballistic with mild drag
    Ages particles and kills expired/out-of-domain ones."""
    tid = wp.tid()
    if tid >= ww_count[0]:
        return
    if ww_alive[tid] == 0:
        return
    p = ww_positions[tid]
    vel = ww_velocities[tid]

    vp = world_to_voxel(p, origin, voxel_size)
    fluid_vel = sample_mac_velocity(
        u, v, w, wu, wv, ww_weight, leaf_table, voxels_dim, leaves_dim, vp
    )
    sdf = sample_scalar_trilinear(
        surface_sdf, leaf_table, voxels_dim, leaves_dim, vp, 1.0e6
    )

    if sdf < -foam_band:
        # Bubble: strong drag toward fluid velocity + buoyancy against gravity.
        t = wp.clamp(bubble_drag * dt, 0.0, 1.0)
        vel = vel + (fluid_vel - vel) * t - gravity * dt
    elif sdf <= foam_band:
        # Foam: ride the surface flow.
        vel = fluid_vel
    else:
        # Spray: ballistic.
        t = wp.clamp(spray_drag * dt, 0.0, 1.0)
        vel = vel + (fluid_vel - vel) * t + gravity * dt

    p = p + vel * dt

    # Collision response against collider SDF.
    cvp = world_to_voxel(p, origin, voxel_size)
    csdf = sample_scalar_trilinear(
        collision_sdf, leaf_table, voxels_dim, leaves_dim, cvp, 1.0e6
    )
    if csdf < 0.0:
        eps = 0.5 * voxel_size
        gx = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim,
                                     cvp + wp.vec3(0.5, 0.0, 0.0), 1.0e6) - csdf
        gy = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim,
                                     cvp + wp.vec3(0.0, 0.5, 0.0), 1.0e6) - csdf
        gz = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim,
                                     cvp + wp.vec3(0.0, 0.0, 0.5), 1.0e6) - csdf
        n = wp.vec3(gx, gy, gz)
        if wp.length(n) > 1.0e-8:
            n = wp.normalize(n)
            p = p - n * csdf
            vn = wp.dot(vel, n)
            if vn < 0.0:
                vel = vel - n * vn

    # Age / lifespan.
    age = ww_age[tid] + dt
    ww_age[tid] = age
    dead = age >= ww_life[tid]

    # Domain containment.
    for axis in range(3):
        if p[axis] < domain_min[axis] or p[axis] > domain_max[axis]:
            dead = True

    if dead:
        ww_alive[tid] = 0
        free_slot = wp.atomic_add(ww_free_count, 0, 1)
        if free_slot < ww_capacity:
            ww_free_slots[free_slot] = tid
        else:
            wp.atomic_sub(ww_free_count, 0, 1)

    ww_positions[tid] = p
    ww_velocities[tid] = vel


register_fn(k_advect_whitewater, "fluidSimAdvectWhiteWater",
            notes="Foam/spray/bubble regimes classified by surface SDF.")


# ---------------------------------------------------------------------------
# Whitewater grid data (density splat used for rendering/emission limits)
# ---------------------------------------------------------------------------


@wp.kernel
def k_whitewater_set_grid_data(
    ww_positions: wp.array(dtype=wp.vec3),
    ww_alive: wp.array(dtype=wp.int32),
    ww_count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    density: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= ww_count[0]:
        return
    if ww_alive[tid] == 0:
        return
    p = ww_positions[tid]
    vp = world_to_voxel(p, origin, voxel_size)
    flat = voxel_flat_index(
        leaf_table, voxels_dim, leaves_dim,
        wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2]))),
    )
    if flat < 0:
        return
    wp.atomic_add(density, flat, 1.0)


register_fn(k_whitewater_set_grid_data, "fluidSimWhiteWaterSetGridData",
            notes="Per-voxel whitewater density splat.")


# ---------------------------------------------------------------------------
# Host-side whitewater particle container
# ---------------------------------------------------------------------------


class WhitewaterSystem:
    def __init__(self, capacity, device, candidate_capacity=None):
        self.capacity = int(capacity)
        self.candidate_capacity = max(
            1, int(self.capacity if candidate_capacity is None else candidate_capacity))
        self.device = device
        self.position = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.velocity = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.age = wp.zeros(self.capacity, dtype=float, device=device)
        self.life = wp.zeros(self.capacity, dtype=float, device=device)
        self.alive = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.free_slots = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        self.free_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.host_count = 0
        # Compaction double buffers.
        self._position2 = wp.zeros_like(self.position)
        self._velocity2 = wp.zeros_like(self.velocity)
        self._age2 = wp.zeros_like(self.age)
        self._life2 = wp.zeros_like(self.life)
        self._scan = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        self._count2 = wp.zeros(1, dtype=wp.int32, device=device)
        # Emission scans only the active liquid prefix, never whitewater capacity.
        self.candidates = wp.zeros(
            self.candidate_capacity, dtype=wp.int32, device=device)
        self.candidate_offsets = wp.zeros_like(self.candidates)
        self.admission_plan = wp.zeros(4, dtype=wp.int32, device=device)
        # required, admitted, dropped, and available are per-frame totals.
        self.diagnostics = wp.zeros(4, dtype=wp.int32, device=device)
        self.last_diagnostics = {
            "policy": WHITEWATER_POLICY_FIXED_CAPACITY,
            "status": WHITEWATER_STATUS_OK,
            "required": 0,
            "available": self.capacity,
            "admitted": 0,
            "dropped": 0,
            "message": "",
        }
        self._dependency_error = ""

    def clear(self):
        self.count.zero_()
        self.alive.zero_()
        self.free_count.zero_()
        self.host_count = 0
        self.reset_frame_diagnostics()

    def reset_frame_diagnostics(self):
        self.diagnostics.zero_()
        self.admission_plan.zero_()
        self._dependency_error = ""
        self.last_diagnostics = {
            "policy": WHITEWATER_POLICY_FIXED_CAPACITY,
            "status": WHITEWATER_STATUS_OK,
            "required": 0,
            "available": self.capacity,
            "admitted": 0,
            "dropped": 0,
            "message": "",
        }

    def allocate_candidate_workspace(self, capacity):
        """Allocate replacement emission scratch without mutating live state."""
        capacity = max(1, int(capacity))
        candidates = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        return candidates, wp.zeros_like(candidates)

    def swap_candidate_workspace(self, workspace):
        """Commit a fully allocated candidate workspace."""
        candidates, offsets = workspace
        self.candidates = candidates
        self.candidate_offsets = offsets
        self.candidate_capacity = len(candidates)

    def mark_sdf_unavailable(self, message):
        self._dependency_error = str(message or "Simulation liquid SDF build failed")

    def refresh_diagnostics(self):
        values = self.diagnostics.numpy()
        required, admitted, dropped, available = (
            int(values[0]), int(values[1]), int(values[2]), int(values[3]))
        if self._dependency_error:
            status = WHITEWATER_STATUS_SDF_UNAVAILABLE
            message = self._dependency_error
        elif dropped:
            status = WHITEWATER_STATUS_CAPACITY_EXHAUSTED
            message = (
                f"Whitewater fixed capacity admitted {admitted:,} of "
                f"{required:,}; dropped {dropped:,}")
        else:
            status = WHITEWATER_STATUS_OK
            message = ""
        self.last_diagnostics = {
            "policy": WHITEWATER_POLICY_FIXED_CAPACITY,
            "status": status,
            "required": required,
            "available": available,
            "admitted": admitted,
            "dropped": dropped,
            "message": message,
        }
        return self.last_diagnostics

    def emit(self, *, candidate_count, liquid_positions, liquid_velocities,
             liquid_count, churn, surface_sdf, leaf_table, voxels_dim,
             leaves_dim, origin, voxel_size, dt, churn_threshold,
             emission_rate, surface_band, min_speed, life_min, life_max, seed):
        """Run device-only candidate/scan/admit/commit emission."""
        life_min, life_max = normalized_life_range(life_min, life_max)
        n = max(1, min(int(candidate_count), self.candidate_capacity))
        wp.launch(
            k_mark_whitewater_candidates, dim=n,
            inputs=[
                liquid_positions, liquid_velocities, liquid_count, churn,
                surface_sdf, leaf_table, voxels_dim, leaves_dim, origin,
                voxel_size, dt, churn_threshold, emission_rate, surface_band,
                min_speed, seed,
            ],
            outputs=[self.candidates], device=self.device,
        )
        wp.utils.array_scan(
            self.candidates[:n], self.candidate_offsets[:n], inclusive=False)
        wp.launch(
            k_prepare_whitewater_admission, dim=1,
            inputs=[
                self.candidates, self.candidate_offsets, n, self.count,
                self.free_count, self.capacity,
            ],
            outputs=[self.admission_plan, self.diagnostics], device=self.device,
        )
        wp.launch(
            k_commit_whitewater_candidates, dim=n,
            inputs=[
                liquid_positions, liquid_velocities, self.candidates,
                self.candidate_offsets, voxel_size, life_min, life_max, seed,
                self.free_slots, self.admission_plan,
            ],
            outputs=[
                self.position, self.velocity, self.age, self.life, self.alive,
            ], device=self.device,
        )
        wp.launch(
            k_finalize_whitewater_admission, dim=1,
            inputs=[self.admission_plan],
            outputs=[self.count, self.free_count], device=self.device,
        )

    def clamp_count(self):
        wp.launch(
            k_clamp_count, dim=1, inputs=[self.count, self.capacity],
            device=self.device,
        )

    def refresh_host_count(self):
        self.clamp_count()
        self.host_count = int(self.count.numpy()[0])
        return self.host_count

    def compact(self, count_hint: int | None = None):
        """Stream-compact live particles without any host synchronization.

        Args:
            count_hint: Optional upper bound for the live particle prefix. When
                provided by the solver, scan and scatter only that active range.

        All bookkeeping stays on the device; `host_count` is refreshed lazily
        by callers that actually need a CPU-side value (end-of-frame stats,
        viewport export).
        """
        cap = self.capacity if count_hint is None else max(0, min(int(count_hint), self.capacity))
        if cap <= 0:
            self.count.zero_()
            self.alive.zero_()
            self.host_count = 0
            return
        self.clamp_count()

        # count_hint is the same host-side upper-bound contract used for
        # particle kernels. Keep the scan/reset bounded so whitewater does not
        # scale with maximum capacity when only a small live prefix exists.
        alive = self.alive if count_hint is None else self.alive[:cap]
        scan = self._scan if count_hint is None else self._scan[:cap]
        wp.utils.array_scan(alive, scan, inclusive=False)
        wp.launch(
            k_ww_compact_scatter,
            dim=cap,
            inputs=[
                self.alive, self._scan, self.position, self.velocity,
                self.age, self.life,
            ],
            outputs=[
                self._position2, self._velocity2, self._age2, self._life2,
                self._count2,
            ],
            device=self.device,
        )
        wp.launch(
            k_ww_final_count,
            dim=1,
            inputs=[self.alive, self._scan, cap],
            outputs=[self._count2],
            device=self.device,
        )
        self.position, self._position2 = self._position2, self.position
        self.velocity, self._velocity2 = self._velocity2, self.velocity
        self.age, self._age2 = self._age2, self.age
        self.life, self._life2 = self._life2, self.life
        self.count, self._count2 = self._count2, self.count
        if count_hint is None:
            self.alive.zero_()
        else:
            self.alive[:cap].zero_()
        wp.launch(
            k_ww_mark_alive, dim=cap, inputs=[self.count],
            outputs=[self.alive],
            device=self.device,
        )
        self.free_count.zero_()


@wp.kernel
def k_ww_compact_scatter(
    alive: wp.array(dtype=wp.int32),
    scan: wp.array(dtype=wp.int32),
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    out_position: wp.array(dtype=wp.vec3),
    out_velocity: wp.array(dtype=wp.vec3),
    out_age: wp.array(dtype=float),
    out_life: wp.array(dtype=float),
    out_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if alive[tid] == 0:
        return
    dst = scan[tid]
    out_position[dst] = position[tid]
    out_velocity[dst] = velocity[tid]
    out_age[dst] = age[tid]
    out_life[dst] = life[tid]


@wp.kernel
def k_ww_final_count(
    alive: wp.array(dtype=wp.int32),
    scan: wp.array(dtype=wp.int32),
    n: int,
    out_count: wp.array(dtype=wp.int32),
):
    out_count[0] = scan[n - 1] + alive[n - 1]


@wp.kernel
def k_ww_mark_alive(
    count: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < count[0]:
        alive[tid] = 1
