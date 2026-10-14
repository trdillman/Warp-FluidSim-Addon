"""Blender acceptance tests for the 12 Custom Effect example scenes.

Verifies:
  - All 12 scenes exist and open cleanly in Blender.
  - Authoritative Domain and Source configuration.
  - The effect stack ships Untrusted Imported and applies once trusted locally.
  - Multi-effect layering on scenes 11 and 12 (multiple stack layers).
  - Timeline playback advances the solver and updates particle positions.

These are add-on examples, not the canonical demos: this proves playback, not
per-effect dispatch evidence (see docs/DEMO_SCENE_CONTRACT.md).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "support"))
from _test_device import script_device  # noqa: E402
from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

SCENES_DIR = Path(REPO_ROOT) / "fluidSim_warp_blender51" / "resources" / "examples" / "custom_effects" / "scenes"

DEVICE = script_device()

LAYERED_EXAMPLES = frozenset(
    {"11_layered_atmospheric_tempest_example.blend", "12_layered_hydro_centrifuge_example.blend"}
)

EXPECTED_EXAMPLES = (
    "01_thermal_updraft_example.blend",
    "02_point_attractor_example.blend",
    "03_helical_vortex_example.blend",
    "04_directional_wind_example.blend",
    "05_centrifugal_spinner_example.blend",
    "06_surface_tension_bead_example.blend",
    "07_viscous_heating_example.blend",
    "08_periodic_ripples_example.blend",
    "09_bounding_box_confinement_example.blend",
    "10_voxel_viscosity_decay_example.blend",
    "11_layered_atmospheric_tempest_example.blend",
    "12_layered_hydro_centrifuge_example.blend",
)


def verify_scene(filename: str) -> None:
    path = SCENES_DIR / filename
    if not path.is_file():
        raise AssertionError(f"Example scene missing: {path}")

    bpy.ops.wm.open_mainfile(filepath=str(path))
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)

    from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings, session

    scene = bpy.context.scene
    fs = getattr(scene, "fluidsim", None)
    if fs is None or not fs.domain_object:
        raise AssertionError(f"{filename}: Scene lacks designated domain object")

    ds = active_domain_settings(scene)
    if ds is None:
        raise AssertionError(f"{filename}: Active domain settings are missing")

    sources = [obj for obj in scene.objects if getattr(getattr(obj, "fluidsim", None), "role", "") == "SOURCE"]
    if not sources:
        raise AssertionError(f"{filename}: No fluid source objects in scene")

    effects_stack = getattr(scene, "fluidsim_effects", None)
    if effects_stack is None or len(effects_stack.items) == 0:
        raise AssertionError(f"{filename}: Custom Effects stack is empty")

    if not effects_stack.applied_signature:
        raise AssertionError(f"{filename}: Applied effect signature is empty")

    # A shipped scene must not carry trust: against an empty store every
    # effect opens Untrusted Imported, and only an explicit local decision
    # (simulated here) lets the exact embedded definition apply.
    from fluidSim_warp_blender51.blender import effects
    from fluidSim_warp_blender51.sim.effects.trust import TrustStore, content_trust_digest

    store = TrustStore(Path(tempfile.mkdtemp(prefix="fluidsim-scene-trust-")) / "trust.json")
    effects.trust_store_for_scene = lambda _scene=None: store
    effects.refresh_stack_state(scene)
    for item in effects_stack.items:
        if item.trust_state != "UNTRUSTED_IMPORTED":
            raise AssertionError(f"{filename}: {item.display_name} ships pre-trusted ({item.trust_state})")
    for item in effects_stack.items:
        definition = effects.effect_instance_from_pg(scene, item).definition
        store.trust(content_trust_digest(definition), item.display_name)
    candidate = effects.apply_stack(scene, device=DEVICE)
    if not candidate.success:
        raise AssertionError(f"{filename}: embedded effect stack does not apply: {candidate.issues}")
    if scene.fluidsim.live:
        raise AssertionError(f"{filename}: Live stayed on with untrusted effects")
    scene.fluidsim.live = True  # the artist resumes Live after Review & Trust + Compile & Apply

    # Layered examples verification
    if filename in LAYERED_EXAMPLES:
        if len(effects_stack.items) < 2:
            raise AssertionError(f"{filename}: Expected at least 2 layered effects, found {len(effects_stack.items)}")
    else:
        if len(effects_stack.items) != 1:
            raise AssertionError(f"{filename}: Expected exactly 1 effect, found {len(effects_stack.items)}")

    # Verify live playback stepping
    try:
        scene.frame_set(2)
        sess = session()
        if not sess.active:
            raise AssertionError(
                f"{filename}: Solver session failed to activate at frame 2 "
                f"(live={scene.fluidsim.live}, status={scene.fluidsim.status!r}, "
                f"stack={effects_stack.status!r}, reason={effects_stack.blocking_reason!r}, "
                f"items={[(i.status, i.trust_state) for i in effects_stack.items]})"
            )
        initial_count = sess.solver.particles.refresh_host_count()
        if initial_count <= 0:
            raise AssertionError(f"{filename}: Solver initialized with 0 particles")

        pos_frame2 = sess.solver.particles.position[:min(1024, initial_count)].numpy().copy()
        scene.frame_set(3)
        pos_frame3 = sess.solver.particles.position[:min(1024, initial_count)].numpy()

        disp = float(np.max(np.linalg.norm(pos_frame3 - pos_frame2, axis=1)))
        if disp <= 1.0e-7:
            raise AssertionError(f"{filename}: Particles did not advance between frames (disp={disp})")
    finally:
        scene.fluidsim.live = False
        if session().active:
            bpy.ops.fluidsim.release()

    print(f"PASS: {filename} (particles={initial_count}, displacement={disp:.5f}m, effects={len(effects_stack.items)})")


def main() -> None:
    print(f"Running acceptance test on {len(EXPECTED_EXAMPLES)} example scenes...")
    for example in EXPECTED_EXAMPLES:
        verify_scene(example)

    print("CUSTOM-EFFECT-EXAMPLES-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
