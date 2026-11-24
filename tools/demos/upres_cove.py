"""Upres Cove: the dam-break cove at showcase resolution (numpy only, no bpy).

The guided upres review scene. It is the shipped dam_break cove (same reservoir,
dam walls, beach, headlands, sea stack and jetty from
``tools/demos/dam_break_cove.py``) run at a smaller voxel size, with a guided
fine solve at double the coarse resolution:

    dam_break demo   0.03 m voxels,   ~256k particles
    upres_cove       0.020 m voxels,  ~864k particles  (the coarse sim)
    guided upres     0.010 m voxels, fine surface shell only
    full fine        0.010 m voxels,  ~6.9M particles  (the comparison)

The builder (``build_demo_scenes.py -- upres_cove``) and the review run
(``tools/profiling/compare_guided_upres.py --demo upres_cove``) both read these
numbers, and the unit tests check that they fit a 16 GB card.

Whitewater is off: the guided fine solve does not emit it, so leaving it on
would make the coarse and upres frames look different for the wrong reason.
Narrow Band is off on the coarse sim so the particle count is the full volume;
the guided solve forces its own narrow band.
"""

from __future__ import annotations

import numpy as np

from tools.demos import dam_break_cove as cove

NAME = "upres_cove"
VOXEL = 0.020
PARTICLES_PER_VOXEL = 8.0
FRAMES = 72
# Frames the review run turns into meshes by default (start, mid-surge, wash).
REVIEW_FRAMES = (24, 48, 72)

# Guided upres defaults for the review run.
UPRES_FACTOR = 2
SHELL_VOXELS = 4.0
CORE_SHELLS = 2.0

# The coarse particle reserve covers the ~864k Initial Fill. Leaves are
# fixed (the review run steps the solvers directly, without the add-on's leaf
# growth, which would swap the coarse solver out from under the upres). The
# reserves cover the entire domain at each default resolution, including
# the terrain collider bounds and the liquid as it spreads. SparseGrid
# clamps the reservation to the number of addressable leaves.
MAX_PARTICLES = 1_000_000
LEAF_CAPACITY = 8192
FINE_LEAF_CAPACITY = 36864

DOMAIN_LO = cove.DOMAIN_LO
DOMAIN_HI = cove.DOMAIN_HI
RESERVOIR_LO = cove.RESERVOIR_LO
RESERVOIR_HI = cove.RESERVOIR_HI

WATER_COLOR = (0.04, 0.24, 0.50, 1.0)
CAMERA = ((7.4, -3.1, 3.0), (2.5, 0.0, 0.3), 40.0)


def fine_voxel(factor=UPRES_FACTOR):
    return VOXEL / int(factor)


def reservoir_volume():
    return float(np.prod(np.subtract(RESERVOIR_HI, RESERVOIR_LO)))


def expected_particles(voxel=VOXEL):
    """Particles the reservoir's Initial Fill emits at ``voxel``."""
    return reservoir_volume() / voxel ** 3 * PARTICLES_PER_VOXEL


def leaf_budget_scale():
    """``GuidedUpres(leaf_budget_scale=...)`` that gives ``FINE_LEAF_CAPACITY``."""
    return FINE_LEAF_CAPACITY / LEAF_CAPACITY


def fine_settings(coarse_settings, factor=UPRES_FACTOR):
    """Settings for the unguided full fine comparison sim."""
    import dataclasses

    factor = int(factor)
    return dataclasses.replace(
        coarse_settings,
        voxel_size=float(coarse_settings.voxel_size) / factor,
        max_particles=int(coarse_settings.max_particles * factor ** 3),
        # The same rounding GuidedUpres applies to its own fine budget.
        leaf_capacity=int(coarse_settings.leaf_capacity * leaf_budget_scale()),
        whitewater_enabled=False,
    )
