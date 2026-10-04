"""SpaceTime FLIP state: the jitter seed, the last step size, and the phase reference mass.

Space-time FLIP weights every particle's contribution to a MAC face by a
temporal kernel and decides whether a face is liquid or air by comparing the
accumulated space-time mass with a reference mass m0: the expected estimator at
a face under a uniform particle distribution. Three numbers have to survive
between substeps, across Reset and across checkpoint restore for that to be
reproducible, and ``SpaceTimeState`` is their only owner:

* ``seed`` -- the advection jitter stream, advanced once per substep;
* ``previous_dt`` -- the step size the previous substep advected by, which the
  next gather needs to place particles in time;
* ``reference_mass_cached`` -- m0 when the settings do not pin it. Calibrated
  once by a small deterministic Monte Carlo estimate over the kernels the
  Warp gather uses, then reused.

It also owns the part of live editing that depends on m0: when the particle
density changes mid-bake the calibration scales linearly with particles per
voxel, so ``remember_calibration`` / ``rescale_for_density`` carry it across
without paying for another ~86 ms estimate per slider value.

This module is host-only (numpy, no Warp, no Blender), so it runs and is
tested without a GPU. The solver owns one instance and hands the kernels the
numbers; nothing else holds ST state.
"""

from __future__ import annotations

import math

import numpy as np

_SEED_MASK = 0x7FFFFFFF
_LCG_MULTIPLIER = 1664525
_LCG_INCREMENT = 1013904223

# Calibration per unit of particle density, process-wide, keyed by jitter seed.
# Bounded so a long editing session cannot grow it without limit.
_DENSITY_COEFFICIENTS: dict[int, float] = {}
_DENSITY_COEFFICIENT_LIMIT = 64


def _poly6_1d(r: float) -> float:
    q = 1.0 - r * r
    if q <= 0.0:
        return 0.0
    return q * q * q


def _spatial_weight_1d(r: float) -> float:
    return (35.0 / 32.0) * _poly6_1d(r)


def _temporal_weight(tau: float) -> float:
    if tau > 0.5:
        return 0.0
    return (35.0 / 16.0) * _poly6_1d(tau - 0.5)


def _store_density_coefficient(key: int, coefficient: float) -> None:
    if len(_DENSITY_COEFFICIENTS) >= _DENSITY_COEFFICIENT_LIMIT:
        _DENSITY_COEFFICIENTS.pop(next(iter(_DENSITY_COEFFICIENTS)))
    _DENSITY_COEFFICIENTS[key] = coefficient


class SpaceTimeState:
    """The solver's ST-FLIP carry-over, built from its live settings and quality."""

    #: Checkpoint scalar names this state exports and restores.
    CHECKPOINT_SCALARS = ("st_step_seed", "st_previous_dt", "st_reference_mass_cached")

    def __init__(self, settings, quality):
        # ``settings`` is the solver's own settings object, edited in place by
        # live updates, so the manual reference mass and the density are always
        # read as they are now.
        self._settings = settings
        self._quality = quality
        self.seed = 0
        self.previous_dt = 0.0
        self.reference_mass_cached = 0.0
        self.reset_for_run()

    # -- lifecycle ---------------------------------------------------------

    def reset_for_run(self):
        """Start a run: first jitter seed, no previous step, m0 re-derived."""
        s = self._settings
        self.previous_dt = 0.0
        self.seed = int(s.spacetime_seed) & _SEED_MASK
        manual = float(getattr(s, "spacetime_reference_mass", 0.0))
        self.reference_mass_cached = (
            manual if manual > 0.0
            else self.calibrate_reference_mass()
            if s.spacetime_enabled else 0.0
        )

    # -- per-substep -------------------------------------------------------

    def advance_seed(self) -> int:
        """Step the jitter stream and return the seed this substep advects with."""
        self.seed = (self.seed * _LCG_MULTIPLIER + _LCG_INCREMENT) & _SEED_MASK
        return self.seed

    def record_step(self, dt) -> None:
        """Remember the step size the substep just advected by."""
        self.previous_dt = float(dt)

    def gather_dt(self, dt=None) -> float:
        """The step size the next gather places particles in time with.

        The previous substep's size once there is one; before that, ``dt``, and
        failing that one frame's worth of time.
        """
        if self.previous_dt > 0.0:
            return self.previous_dt
        s = self._settings
        return float(dt or (s.time_scale / max(s.frame_rate, 1.0)))

    # -- reference mass m0 -------------------------------------------------

    def reference_mass(self) -> float:
        """m0 for the phase test and the variable-density projection."""
        ref = float(getattr(self._settings, "spacetime_reference_mass", 0.0))
        if ref > 0.0 and math.isfinite(ref):
            return ref
        cached = float(self.reference_mass_cached)
        if cached > 0.0 and math.isfinite(cached):
            return cached
        return max(1.0, float(self._quality.rest_particle_count))

    def calibrate_reference_mass(self) -> float:
        """Monte Carlo estimate of the ST phase reference mass m0.

        The paper defines m0 as the expected space-time mass estimator at a
        face under a uniform particle distribution with Nppc particles per
        cell.  Use the exact kernels used by the Warp P2G path on a small
        particle-filled patch with tau uniformly sampled in [-1/2, 1/2].
        This keeps phase thresholds tied to the implemented kernel constants,
        support convention, and particle-per-cell density rather than assuming
        a closed-form normalization.
        """
        s = self._settings
        manual = float(getattr(s, "spacetime_reference_mass", 0.0))
        if manual > 0.0 and math.isfinite(manual):
            return manual
        ppc = max(1.0e-6, float(self._quality.liquid_sampling))
        particles_per_cell = max(1, int(math.ceil(ppc)))
        particle_mass = ppc / float(particles_per_cell)
        # Deterministic and intentionally small: 3 MAC face orientations,
        # 27 cells around the face support, 32 random realizations.
        rng = np.random.default_rng(int(getattr(s, "spacetime_seed", 1729)) & 0xFFFFFFFF)
        faces = (
            np.array((0.0, 0.5, 0.5), dtype=np.float64),
            np.array((0.5, 0.0, 0.5), dtype=np.float64),
            np.array((0.5, 0.5, 0.0), dtype=np.float64),
        )
        total = 0.0
        samples = 0
        for face in faces:
            lo = np.floor(face - 1.0).astype(int)
            hi = np.floor(face + 1.0).astype(int)
            for _ in range(32):
                accum = 0.0
                for ix in range(int(lo[0]), int(hi[0]) + 1):
                    for iy in range(int(lo[1]), int(hi[1]) + 1):
                        for iz in range(int(lo[2]), int(hi[2]) + 1):
                            cell = np.array((ix, iy, iz), dtype=np.float64)
                            for _p in range(particles_per_cell):
                                pos = cell + rng.random(3)
                                r = face - pos
                                ws = (
                                    _spatial_weight_1d(float(r[0]))
                                    * _spatial_weight_1d(float(r[1]))
                                    * _spatial_weight_1d(float(r[2]))
                                )
                                if ws <= 0.0:
                                    continue
                                tau = rng.random() - 0.5
                                accum += particle_mass * ws * _temporal_weight(float(tau))
                total += accum
                samples += 1
        if samples <= 0:
            return max(1.0, ppc)
        estimate = total / float(samples)
        if not math.isfinite(estimate) or estimate <= 1.0e-8:
            return max(1.0, ppc)
        return float(estimate)

    # -- live density edits ------------------------------------------------

    def remember_calibration(self) -> None:
        """Record the current automatic m0 as a per-density coefficient.

        Call before a live edit that may change particles per voxel, so
        ``rescale_for_density`` can scale it instead of re-estimating.
        """
        if not self._derives_reference_mass():
            return
        reference_mass = float(self.reference_mass_cached or 0.0)
        if reference_mass <= 0.0:
            return
        _store_density_coefficient(self._cache_key(), reference_mass / self._density())

    def rescale_for_density(self) -> bool:
        """Re-derive the automatic m0 for the current particle density.

        The estimator is linear in particles per voxel, so a remembered
        coefficient (or, failing that, one calibration) serves every density.
        Returns False, changing nothing, when m0 is pinned or ST is off.
        """
        if not self._derives_reference_mass():
            return False
        key = self._cache_key()
        density = self._density()
        coefficient = _DENSITY_COEFFICIENTS.get(key)
        if coefficient is None:
            coefficient = float(self.calibrate_reference_mass()) / density
            _store_density_coefficient(key, coefficient)
        self.reference_mass_cached = coefficient * density
        return True

    def _derives_reference_mass(self) -> bool:
        s = self._settings
        return bool(
            getattr(s, "spacetime_enabled", False)
            and float(getattr(s, "spacetime_reference_mass", 0.0)) <= 0.0
        )

    def _cache_key(self) -> int:
        return int(getattr(self._settings, "spacetime_seed", 1729))

    def _density(self) -> float:
        return max(1.0e-6, float(self._settings.particles_per_voxel))

    # -- checkpoint --------------------------------------------------------

    def checkpoint_scalars(self) -> dict:
        return {
            "st_step_seed": int(self.seed),
            "st_previous_dt": float(self.previous_dt),
            "st_reference_mass_cached": float(self.reference_mass_cached),
        }

    def restore_checkpoint_scalars(self, scalars) -> None:
        self.seed = int(scalars["st_step_seed"])
        self.previous_dt = float(scalars["st_previous_dt"])
        self.reference_mass_cached = float(scalars["st_reference_mass_cached"])

    def timing_is_finite(self) -> bool:
        return math.isfinite(self.previous_dt)
