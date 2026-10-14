"""Deterministic random liquid scenes for the learned pressure guess.

Pure Warp (no Blender): each scene is a closed box domain filled with one or
more liquid blocks, optionally thrown with an initial velocity, then stepped
with the production ``FluidSolver``. A scene is fully described by its
integer seed, so a dataset or an evaluation can be reproduced from a seed
list alone.

Two disjoint seed ranges: ``TRAIN_SEEDS`` feed ``export_dataset.py`` and
``train.py``; ``HELDOUT_SEEDS`` are only ever used by ``evaluate.py`` (the
spec's held-out set). Held-out scenes also use shapes the training scenes
never have (two separated blocks), so the held-out score measures
generalisation, not memorisation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TRAIN_SEEDS = range(0, 1000)
HELDOUT_SEEDS = range(1000, 1100)
VOXEL_SIZE = 1.0 / 32.0
PARTICLES_PER_AXIS = 2


@dataclass(frozen=True)
class Block:
    lo: tuple            # world-space min corner
    hi: tuple            # world-space max corner
    velocity: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SceneSpec:
    seed: int
    domain_max: tuple    # domain_min is the origin
    blocks: tuple = field(default_factory=tuple)

    @property
    def heldout(self):
        return self.seed in HELDOUT_SEEDS


def _cells(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def make_scene(seed):
    """The scene for ``seed`` (same seed, same scene, on every machine)."""
    seed = int(seed)
    rng = np.random.default_rng(np.random.SeedSequence([0x1E9A, seed]))
    h = VOXEL_SIZE
    # Domain 16..40 cells per side; leaves are 8 cells, so 2..5 leaves.
    dims = (_cells(rng, 16, 40), _cells(rng, 16, 40), _cells(rng, 16, 40))
    dmax = tuple(round(d * h, 6) for d in dims)
    heldout = seed in HELDOUT_SEEDS
    kind = "two_blocks" if heldout else ("pool", "column", "drop")[int(rng.integers(0, 3))]
    blocks = []

    def block(x0, x1, y0, y1, z0, z1, vel=(0.0, 0.0, 0.0)):
        blocks.append(Block(lo=(x0 * h, y0 * h, z0 * h), hi=(x1 * h, y1 * h, z1 * h),
                            velocity=tuple(float(v) for v in vel)))

    nx, ny, nz = dims
    if kind == "pool":            # a resting pool plus a sloshing push
        depth = _cells(rng, nz // 4, (3 * nz) // 4)
        push = rng.uniform(-1.0, 1.0, 2)
        block(0, nx, 0, ny, 0, depth, (push[0], push[1], 0.0))
    elif kind == "column":        # dam break: a column against one wall
        w = _cells(rng, nx // 4, nx // 2)
        d = _cells(rng, ny // 3, ny)
        top = _cells(rng, nz // 2, nz)
        block(0, w, 0, d, 0, top)
    elif kind == "drop":          # a shallow pool and a falling, thrown drop
        depth = _cells(rng, 2, nz // 4)
        block(0, nx, 0, ny, 0, depth)
        s = _cells(rng, 4, max(4, min(nx, ny, nz) // 3))
        x0 = _cells(rng, 0, nx - s)
        y0 = _cells(rng, 0, ny - s)
        z0 = _cells(rng, nz // 2, nz - s)
        v = rng.uniform(-1.5, 1.5, 3)
        block(x0, x0 + s, y0, y0 + s, z0, z0 + s, (v[0], v[1], min(0.0, v[2])))
    else:                         # held-out only: two separated columns
        w = _cells(rng, 3, max(3, nx // 4))
        top = _cells(rng, nz // 3, nz)
        block(0, w, 0, ny, 0, top)
        block(nx - w, nx, 0, ny, 0, _cells(rng, nz // 3, nz))
    return SceneSpec(seed=seed, domain_max=dmax, blocks=tuple(blocks))


def scenes(seeds):
    return [make_scene(seed) for seed in seeds]


def block_particles(block, per_axis=PARTICLES_PER_AXIS):
    """Jitter-free lattice of ``per_axis**3`` particles per voxel in a block."""
    spacing = VOXEL_SIZE / per_axis
    axes = []
    for lo, hi in zip(block.lo, block.hi):
        n = max(1, int(round((hi - lo) / spacing)))
        axes.append(lo + (np.arange(n) + 0.5) * spacing)
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def solver_settings(spec, device, **extra):
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    total = sum(len(block_particles(b)) for b in spec.blocks)
    values = dict(
        domain_min=(0.0, 0.0, 0.0), domain_max=spec.domain_max,
        voxel_size=VOXEL_SIZE, leaf_capacity=512,
        max_particles=max(50_000, 2 * total),
        surface_enabled=False, whitewater_enabled=False, device=device)
    values.update(extra)
    return SolverSettings(**values)


def build_solver(spec, device, **extra):
    """A ``FluidSolver`` for ``spec`` with its liquid injected; caller releases."""
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    solver = FluidSolver(solver_settings(spec, device, **extra))
    for block in spec.blocks:
        solver.inject_particles(block_particles(block), velocity=block.velocity)
    return solver


if __name__ == "__main__":
    for spec in scenes(list(TRAIN_SEEDS)[:5] + list(HELDOUT_SEEDS)[:3]):
        print(spec)
