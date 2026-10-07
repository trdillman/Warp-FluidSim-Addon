"""Headless Blender acceptance test for FluidSim Demo Gallery.

Verifies:
1. FLUIDSIM_PT_demo_gallery and fluidsim.load_demo are registered.
2. Opening a demo when the current scene is empty appends it into a New Scene.
3. The initial empty scene is preserved in bpy.data.scenes.
4. Loading a subsequent demo when the current scene is populated also creates a
   New Scene, preserving the previous scene.

Run:
    blender --background --factory-startup --python tests/acceptance/blender_demo_gallery_acceptance.py
"""

from __future__ import annotations

import os
import sys

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)
    if not cond:
        FAILURES.append((name, detail))


def run_acceptance():
    print("--- Starting Demo Gallery Acceptance ---", flush=True)

    # 1. Enable repo add-on
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)

    from fluidSim_warp_blender51.blender.demo_gallery import (
        DEMO_CATALOG,
        is_scene_empty,
    )
    from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings

    # 2. Verify operator and panel registration
    check(
        "operator_registered",
        hasattr(bpy.ops.fluidsim, "load_demo"),
        "fluidsim.load_demo is available in bpy.ops",
    )
    check(
        "panel_registered",
        hasattr(bpy.types, "FLUIDSIM_PT_demo_gallery"),
        "FLUIDSIM_PT_demo_gallery is registered in bpy.types",
    )

    initial_scene = bpy.context.scene
    initial_scene_name = initial_scene.name
    print(f"Initial scene name: {initial_scene_name}", flush=True)

    # Verify is_scene_empty on fresh scene
    initial_empty = is_scene_empty(initial_scene)
    check("initial_scene_is_empty", initial_empty, f"initial scene is empty: {initial_empty}")

    initial_scene_count = len(bpy.data.scenes)

    # 3. Load crown_splash demo into a New Scene from empty initial scene
    res = bpy.ops.fluidsim.load_demo(demo_id="crown_splash")
    check("load_crown_splash_finished", res == {"FINISHED"}, f"operator result: {res}")

    check(
        "scene_count_increased",
        len(bpy.data.scenes) == initial_scene_count + 1,
        f"scenes before={initial_scene_count}, after={len(bpy.data.scenes)}",
    )

    current_scene = bpy.context.scene
    check(
        "active_scene_switched",
        current_scene.name != initial_scene_name,
        f"active scene switched to '{current_scene.name}'",
    )
    check(
        "initial_scene_preserved",
        initial_scene_name in bpy.data.scenes,
        f"initial scene '{initial_scene_name}' still exists in bpy.data.scenes",
    )

    # Verify authoritative domain and seed objects exist in imported demo scene
    domain = active_domain_settings(current_scene)
    check("imported_scene_has_domain", domain is not None, "domain settings resolved")
    if domain:
        check(
            "crown_splash_voxel_size",
            abs(float(domain.voxel_size) - 0.01) < 1e-4,
            f"expected 0.01 voxel_size, got {domain.voxel_size}",
        )

    # 4. Now load dam_break when current scene is NOT empty
    check(
        "current_scene_not_empty",
        not is_scene_empty(current_scene),
        "crown_splash scene is recognized as not empty",
    )

    res2 = bpy.ops.fluidsim.load_demo(demo_id="dam_break")
    check("load_dam_break_finished", res2 == {"FINISHED"}, f"operator result: {res2}")

    check(
        "scene_count_increased_again",
        len(bpy.data.scenes) == initial_scene_count + 2,
        f"total scenes now={len(bpy.data.scenes)}",
    )

    active_dam_scene = bpy.context.scene
    check(
        "active_scene_is_dam_break",
        "Dam Break" in active_dam_scene.name,
        f"active scene is '{active_dam_scene.name}'",
    )
    check(
        "crown_splash_scene_preserved",
        current_scene.name in bpy.data.scenes,
        f"crown splash scene '{current_scene.name}' still exists",
    )

    dam_domain = active_domain_settings(active_dam_scene)
    check("dam_scene_has_domain", dam_domain is not None, "dam break domain resolved")
    if dam_domain:
        check(
            "dam_break_voxel_size",
            abs(float(dam_domain.voxel_size) - 0.03) < 1e-4,
            f"expected 0.03 voxel_size, got {dam_domain.voxel_size}",
        )
    check(
        "dam_break_starts_at_frame_start",
        active_dam_scene.frame_current == active_dam_scene.frame_start,
        f"frame_current={active_dam_scene.frame_current}, frame_start={active_dam_scene.frame_start}",
    )
    check(
        "dam_break_render_output_is_blend_relative",
        active_dam_scene.render.filepath == "//fluidsim_demo_renders/dam_break_",
        f"render.filepath={active_dam_scene.render.filepath!r}",
    )

    # 5. Loading the same demo again must not clobber the first copy.
    first_crown_name = current_scene.name
    res3 = bpy.ops.fluidsim.load_demo(demo_id="crown_splash")
    second_crown = bpy.context.scene
    check("reload_crown_splash_finished", res3 == {"FINISHED"}, f"operator result: {res3}")
    check(
        "reload_gets_a_distinct_scene",
        second_crown.name != first_crown_name and first_crown_name in bpy.data.scenes,
        f"first='{first_crown_name}', second='{second_crown.name}'",
    )
    check(
        "scene_count_after_reload",
        len(bpy.data.scenes) == initial_scene_count + 3,
        f"total scenes now={len(bpy.data.scenes)}",
    )

    # 6. Unknown ids are refused without touching the scene list.
    before = len(bpy.data.scenes)
    try:
        bad = bpy.ops.fluidsim.load_demo(demo_id="not_a_demo")
    except RuntimeError as exc:  # Blender raises for operators reporting ERROR
        bad = {"CANCELLED"}
        print(f"unknown demo refused: {exc}", flush=True)
    check(
        "unknown_demo_refused",
        bad == {"CANCELLED"} and len(bpy.data.scenes) == before,
        f"result={bad}, scenes {before}->{len(bpy.data.scenes)}",
    )

    # 7. Gallery is visible in a repo checkout and every demo resolves.
    from fluidSim_warp_blender51.blender.demo_gallery import available_demo_ids

    check(
        "all_demos_available",
        set(available_demo_ids()) == set(DEMO_CATALOG),
        f"available={available_demo_ids()}",
    )
    check(
        "gallery_poll_true",
        bpy.types.FLUIDSIM_PT_demo_gallery.poll(bpy.context),
        "panel polls visible with demo files present",
    )

    print("--- Demo Gallery Acceptance Results ---", flush=True)
    if FAILURES:
        print(f"FAILED with {len(FAILURES)} failure(s):", flush=True)
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}", flush=True)
        sys.exit(1)
    else:
        print("DEMO-GALLERY-ACCEPTANCE-PASS", flush=True)


if __name__ == "__main__":
    run_acceptance()
