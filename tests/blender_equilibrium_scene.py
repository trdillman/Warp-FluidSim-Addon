"""Build the ``headless_equilibrium_ab.py`` scene inside Blender, for eyes.

The headless harness reduces a run to eight numbers, which is enough to say
"volume is being lost" and not enough to say *how*.  Compaction, a collapsing
splash, particles tunnelling into a wall and clumping there, and a pressure
solve that has quietly stopped projecting all produce the same falling
occupied-cell count.  They look nothing alike in the viewport.

Geometry and settings are kept identical to ``equilibrium_settings`` /
``equilibrium_sources`` so what plays here is the run those numbers came from.
Anything changed on one side has to change on the other.

Run from Blender's Scripting workspace, or:

    blender --python tests/blender_equilibrium_scene.py

Then: Initialize Solver in the FluidSim sidebar, and play.
"""

import sys

import bmesh
import bpy

# Solver-side constants.  Mirrors of tests/headless_equilibrium_ab.py.
DOMAIN_MIN = (-1.0, -1.0, 0.0)
DOMAIN_MAX = (1.0, 1.0, 2.0)
SOURCE_CENTER = (-0.55, 0.0, 0.9)
SOURCE_HALF_EXTENTS = (0.45, 1.0, 0.9)
VOXEL_SIZE = 0.025
CFL = 1.0
SECONDS = 10.0

# Analytic end state, for the on-screen reference plane.
FILL_VOLUME = 8.0 * SOURCE_HALF_EXTENTS[0] * SOURCE_HALF_EXTENTS[1] * SOURCE_HALF_EXTENTS[2]
FOOTPRINT = ((DOMAIN_MAX[0] - DOMAIN_MIN[0]) * (DOMAIN_MAX[1] - DOMAIN_MIN[1]))
REST_DEPTH = DOMAIN_MIN[2] + FILL_VOLUME / FOOTPRINT


def _ensure_addon_registered():
    """Register the add-on from this worktree, not from wherever it is installed.

    An installed copy would silently benchmark different code than the one
    under test, which is the whole point of pinning the path here.
    """
    import pathlib
    here = pathlib.Path(__file__).resolve()
    repo = here.parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if hasattr(bpy.types.Scene, "fluidsim"):
        return "already registered"
    import fluidSim_warp_blender51 as addon
    addon.register()
    return f"registered from {repo}"


def _box(name, center, half_extents):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    obj.location = center
    obj.scale = tuple(2.0 * h for h in half_extents)
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def _clear_previous():
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Equilibrium "):
            bpy.data.objects.remove(obj, do_unlink=True)


def build(voxel_size=VOXEL_SIZE, cfl=CFL, spacetime=False):
    scene = bpy.context.scene
    fs = scene.fluidsim

    _clear_previous()

    center = tuple(0.5 * (a + b) for a, b in zip(DOMAIN_MIN, DOMAIN_MAX))
    half = tuple(0.5 * (b - a) for a, b in zip(DOMAIN_MIN, DOMAIN_MAX))
    domain = _box("Equilibrium Domain", center, half)
    fs.domain_object = domain

    source = _box("Equilibrium Source", SOURCE_CENTER, SOURCE_HALF_EXTENTS)
    from fluidSim_warp_blender51.blender import sources as _sources
    _sources.tag_source(source, shape="BOX", mode="PARTICLES")
    # tag_source already sets INITIAL_FILL; source_id comes from the object
    # name on this path, so the empty-id re-fire footgun the headless harness
    # has to work around cannot happen here.

    # Where the surface should end up once settled.  A plane, not a number:
    # the failure mode being chased is the fluid sitting well below it.
    ref = bpy.data.objects.new("Equilibrium Rest Level", bpy.data.meshes.new(
        "Equilibrium Rest Level"))
    scene.collection.objects.link(ref)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    bm.to_mesh(ref.data)
    bm.free()
    ref.location = (center[0], center[1], REST_DEPTH)
    ref.scale = (half[0], half[1], 1.0)
    ref.display_type = "WIRE"
    ref.hide_render = True

    fs.voxel_size = voxel_size
    fs.cfl = cfl
    fs.max_substeps = 256
    fs.max_particles = 8_000_000
    fs.spacetime_enabled = spacetime
    # Both are post-processing on top of the solve; off here for the same
    # reason the headless harness turns them off -- they are not what is
    # being looked at, and the surface mesher would hide the particle
    # clumping that is.
    fs.whitewater_enabled = False
    fs.surface_enabled = False
    # The default view mode is SURFACE, which with surface_enabled off draws
    # nothing at all and reads exactly like a solver that never ran.
    fs.liquid_view_mode = "PARTICLES"
    fs.particle_display_scale = 0.5

    scene.frame_start = 1
    # frame_rate is not a bridged setting -- the solver takes it from the
    # scene fps, so the simulated duration is set here the same way.
    scene.frame_end = int(round(SECONDS * scene.render.fps))
    scene.frame_set(1)
    return domain, source


if __name__ == "__main__":
    print("FluidSim:", _ensure_addon_registered())
    build()
    print(f"FluidSim: equilibrium scene ready. dx={VOXEL_SIZE} cfl={CFL} "
          f"fill_volume={FILL_VOLUME:.4f} rest_depth={REST_DEPTH:.4f}")
    print("FluidSim: Initialize Solver in the sidebar, then play.")
