"""Lane R: liquid the particles have left must return to air.

The persistent level set pins deep values (deeper than ``pin_depth``) against
the particle splat.  A droplet that crosses dry floor must therefore leave no
liquid behind: after it has moved on, every liquid cell of ``phi`` has to lie
close to a particle, and the band's liquid-voxel count has to track the full
band's.  The droplet radius is within the band, so no particle is ever killed
as interior and a liquid cell with no particle nearby is a ghost by
definition.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from _narrow_band_harness import H

from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

pytest.importorskip("warp")

DOMAIN_MAX = (1.0, 0.25, 0.5)
FRAMES = 12
CAPACITY = 60_000
# A liquid cell farther than this from every particle is a ghost.
GHOST_DISTANCE = 2.0 * H


def _droplet(radius_cells):
    spacing = H / 2.0
    centre = np.array([0.15, 0.125, 0.07])
    r = radius_cells * H
    axes = [np.arange(c - r, c + r, spacing) + 0.5 * spacing for c in centre]
    grid = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grid], axis=1)
    keep = np.linalg.norm(pts - centre, axis=1) <= r
    return pts[keep].astype(np.float32)


def _liquid_cell_centres(solver):
    nb = solver.narrow_band
    g = solver.grid
    count = int(g.refresh_topology_counts())
    coords = g.leaf_coords.numpy()[:count].astype(np.int64)
    phi = nb.phi.numpy()[:count * C.LEAF_VOLUME].reshape(count, C.LEAF_VOLUME)
    local = np.arange(C.LEAF_VOLUME)
    li = local & C.LEAF_MASK
    lj = (local >> C.LEAF_SHIFT) & C.LEAF_MASK
    lk = (local >> (2 * C.LEAF_SHIFT)) & C.LEAF_MASK
    ijk = (coords[:, None, :] << C.LEAF_SHIFT) + np.stack([li, lj, lk], axis=1)[None]
    origin = np.array([g.origin[0], g.origin[1], g.origin[2]], dtype=np.float64)
    centres = origin + (ijk.astype(np.float64) + 0.5) * float(g.voxel_size)
    return centres[phi < 0.0]


def _ghost_cells(solver):
    cells = _liquid_cell_centres(solver)
    pos = solver.particle_positions().astype(np.float64)
    if len(cells) == 0 or len(pos) == 0:
        return len(cells)
    ghosts = 0
    for c in cells:
        if np.sqrt(((pos - c) ** 2).sum(axis=1)).min() > GHOST_DISTANCE:
            ghosts += 1
    return ghosts


def _fluid_cells(solver):
    """Cells the pressure solve treats as liquid (both modes)."""
    count = int(solver.grid.refresh_topology_counts())
    vt = solver.voxel_type.data.numpy()[:count * C.LEAF_VOLUME]
    return int(np.count_nonzero((vt & C.VOXEL_FLUID) != 0))


def _run(device, narrow_band, radius_cells):
    solver = FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=512, max_particles=CAPACITY, surface_enabled=False,
        whitewater_enabled=False, narrow_band_enabled=narrow_band,
        narrow_band_voxels=3.0, device=device))
    solver.inject_particles(_droplet(radius_cells), velocity=(2.0, 0.0, 0.0))
    ghosts, liquid, counts, phi_cells = [], [], [], []
    try:
        for _ in range(FRAMES):
            solver.step_frame()
            wp.synchronize()
            counts.append(len(solver.particle_positions()))
            if narrow_band:
                ghosts.append(_ghost_cells(solver))
                phi_cells.append(len(_liquid_cell_centres(solver)))
            liquid.append(_fluid_cells(solver))
    finally:
        solver.release()
    return ghosts, liquid, counts, phi_cells


@pytest.mark.parametrize("radius_cells", [1.5, 2.5])
def test_droplet_over_dry_floor_leaves_no_ghost_liquid(cuda_device, radius_cells):
    ghosts, liquid_band, counts, phi_cells = _run(cuda_device, True, radius_cells)
    assert min(phi_cells) > 0, "the droplet must be liquid in phi (else vacuous)"
    assert max(ghosts) == 0, f"ghost liquid cells per frame: {ghosts}"
    # The droplet is entirely band, so no particle was killed or reseeded.
    assert counts[-1] <= counts[0]
    _, liquid_full, _, _ = _run(cuda_device, False, radius_cells)
    for frame, (b, f) in enumerate(zip(liquid_band, liquid_full), start=1):
        assert b <= 1.5 * f + 8, (
            f"frame {frame}: band liquid voxels {b} vs full band {f}")
