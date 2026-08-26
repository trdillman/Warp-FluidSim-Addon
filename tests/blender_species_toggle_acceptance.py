"""Live Blender acceptance for disabling species and resuming simulation.

Run with ``demos/chromatic_vortex.blend`` loaded before this script.
"""

from __future__ import annotations

import os
import sys

import bpy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon


enable_repo_addon(REPO, default_set=False, persistent=False)
from fluidSim_warp_blender51.blender import scene_bridge

scene = bpy.context.scene
fs = scene.fluidsim
assert os.path.basename(bpy.data.filepath) == "chromatic_vortex.blend"
assert fs.species_enabled
assert fs.particle_attribute_mode == "CLEAR_DYE"
assert fs.surface_attribute_mode == "CLEAR_DYE"
assert fs.particles_attribute_species
assert fs.surface_attribute_species

sess = scene_bridge.initialize_session(scene)
enabled_solver = sess.solver
enabled_stats = scene_bridge.step_frame(scene, sess, write_cache=False)
enabled_count = int(enabled_solver.particles.host_count)
assert enabled_stats is not None
assert enabled_count > 0
assert enabled_solver.settings.species_enabled
assert enabled_solver.species is not None

# Preserve the authored presentation choices while replacing the solver.  This
# is the exact UI transition that previously raised a shape-alignment error.
fs.species_enabled = False
scene_bridge.reset_simulation(scene)

disabled_solver = sess.solver
assert disabled_solver is not enabled_solver
assert not disabled_solver.settings.species_enabled
assert getattr(disabled_solver, "species", None) is None
assert fs.particle_attribute_mode == "CLEAR_DYE"
assert fs.surface_attribute_mode == "CLEAR_DYE"
assert not scene_bridge.current_error(scene), scene_bridge.current_error(scene)

stats = scene_bridge.step_frame(scene, sess, write_cache=False)
disabled_count = int(disabled_solver.particles.host_count)
resumed_count = int(sess.solver.particles.host_count)
assert stats is not None
assert disabled_count > 0
assert resumed_count > 0
assert not scene_bridge.current_error(scene), scene_bridge.current_error(scene)
assert "Solver error" not in fs.status

surface = bpy.data.objects.get("FluidSim Surface")
assert surface is not None and surface.type == "MESH"
assert len(surface.data.vertices) > 0
assert surface.data.attributes.get("fluidsim_species") is None

# Re-enable without repairing or re-authoring the preserved Clear Dye choices.
fs.species_enabled = True
scene_bridge.reset_simulation(scene)
reenabled_solver = sess.solver
reenabled_stats = scene_bridge.step_frame(scene, sess, write_cache=False)
reenabled_count = int(reenabled_solver.particles.host_count)
assert reenabled_stats is not None and reenabled_count > 0
assert reenabled_solver.settings.species_enabled
assert getattr(reenabled_solver, "species", None) is not None
assert not scene_bridge.current_error(scene), scene_bridge.current_error(scene)
surface = bpy.data.objects.get("FluidSim Surface")
assert surface.data.attributes.get("fluidsim_color") is not None

print(
    "SPECIES-TOGGLE-ACCEPTANCE-PASS "
    f"enabled={enabled_count} disabled={disabled_count} resumed={resumed_count} "
    f"reenabled={reenabled_count} status={fs.status!r}",
    flush=True,
)

scene_bridge.release_session(scene)
