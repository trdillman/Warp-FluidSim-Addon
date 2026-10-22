"""Focused Blender 5.2 UI acceptance via launch_profile_safe_addon_review.py.

Uses real RNA, registration and navigation. Draw recording checks bindings;
foreground redraw exercises native layouts. Pixel readability needs human review.
Does not advance simulation, save files/preferences or write simulation caches.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import bpy

RESULT = {"ok": False, "checks": [], "errors": [], "limitations": [
    "Draw recording is not pixel/readability certification.",
    "SolverSettings projection checked; no solver stepping or CUDA claim."
]}
OUT = Path(os.environ["FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"])

def check(name, condition, detail=""):
    RESULT["checks"].append({"name": name, "passed": bool(condition), "detail": str(detail)})
    if not condition:
        raise AssertionError(f"{name}: {detail}")

class RecordingLayout:
    def __init__(self, records=None):
        self.records = [] if records is None else records
    def prop(self, owner, name, **kw):
        assert hasattr(owner, name), name
        self.records.append(("prop", owner, name, kw))
    def label(self, **kw):
        self.records.append(("label", kw))
    def operator(self, *args, **kw):
        return SimpleNamespace()
    def __getattr__(self, name):
        return lambda *a, **kw: RecordingLayout(self.records)

def select(obj):
    for item in bpy.context.scene.objects:
        item.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def run():
    from fluidSim_warp_blender51.blender import ui_common, ui_physics, ui_scene
    from fluidSim_warp_blender51.blender.solver_settings import build_settings
    scene = bpy.context.scene
    scene.fluidsim.live = False
    domain = scene.fluidsim.domain_object
    check("designated domain present", domain is not None)
    settings = domain.fluidsim_domain
    other = bpy.data.objects.new("UI acceptance unrelated selection", None)
    scene.collection.objects.link(other)
    props = next(a for a in bpy.context.screen.areas if a.type == "PROPERTIES")
    check("Blender 5.2", bpy.app.version[:2] == (5, 2), bpy.app.version_string)
    check("29 domain panels registered", len(ui_scene.DOMAIN_DRAW_SOURCE_CLASSES) == 29
          and all(c.is_registered for c in ui_scene.DOMAIN_DRAW_SOURCE_CLASSES))
    seen = set()
    for cls in ui_scene.SCENE_TAB_PANEL_CLASSES:
        parent = getattr(cls, "bl_parent_id", "")
        check("parent order " + cls.bl_idname, not parent or parent in seen)
        seen.add(cls.bl_idname)
    mirrors = {
        source.bl_idname: mirror
        for sources, classes in (
            (ui_physics.DOMAIN_SOURCE_PANELS, ui_physics.DOMAIN_PANEL_CLASSES),
            (ui_physics.SURFACE_SOURCE_PANELS, ui_physics.SURFACE_PANEL_CLASSES),
            (ui_physics.WHITEWATER_SOURCE_PANELS, ui_physics.WHITEWATER_PANEL_CLASSES))
        for source, mirror in zip(sources, classes)
    }
    select(domain)
    check("domain navigation", bpy.ops.fluidsim.open_scene_properties() == {"FINISHED"}
          and props.spaces.active.context == "PHYSICS")
    select(other)
    check("unrelated selection navigation", bpy.ops.fluidsim.open_scene_properties() == {"FINISHED"}
          and props.spaces.active.context == "SCENE")
    for source in ui_scene.DOMAIN_DRAW_SOURCE_CLASSES:
        check("Scene poll " + source.bl_idname, source.poll(bpy.context))
        layout = RecordingLayout()
        source.draw(SimpleNamespace(layout=layout), bpy.context)
        bound = [(r[1], r[2]) for r in layout.records if r[0] == "prop"]
        check("domain bindings " + source.bl_idname,
              all(owner.id_data != other and (owner == settings if owner.bl_rna.identifier == settings.bl_rna.identifier else True) for owner, name in bound))
        select(domain)
        mirror = mirrors[source.bl_idname]
        check("Physics poll " + source.bl_idname, mirror.poll(bpy.context))
        mirrored = RecordingLayout()
        mirror.draw(SimpleNamespace(layout=mirrored), bpy.context)
        check("Scene/Physics bindings " + source.bl_idname,
              [(o.as_pointer(), n) for o, n in bound] ==
              [(r[1].as_pointer(), r[2]) for r in mirrored.records if r[0] == "prop"])
        select(other)
    authored = {"min_substeps": 1, "max_substeps": 4, "pressure_iterations": 40,
                "particles_per_voxel": 8, "voxel_size": 0.04}
    for name, value in authored.items():
        setattr(settings, name, value)
    for tier, factor in (("DRAFT", 0.5), ("FINAL", 2.0), ("PREVIEW", 1.0), ("CUSTOM", 1.0)):
        scene.fluidsim.quality = tier
        solver = build_settings(scene)
        for name in authored:
            before = getattr(settings, name)
            note = ui_common.quality_effective_note(settings, name)
            effective = getattr(solver, name)
            check(f"{tier} {name} note matches projection",
                  (not note) if effective == before else
                  (f"{effective:.4g} m" if name == "voxel_size" else f"{effective:g}") + f" ({tier.title()})" in note, note)
            layout = RecordingLayout()
            ui_common.draw_property(layout, settings, name)
            check(f"{tier} {name} editable binding",
                  any(r[0] == "prop" and r[1] == settings and r[2] == name for r in layout.records))
            check(f"{tier} {name} authored preserved", getattr(settings, name) == before)
        RESULT.setdefault("quality", {})[tier] = {
            n: {"authored": getattr(settings, n), "solver": getattr(solver, n),
                "note": ui_common.quality_effective_note(settings, n)} for n in authored}
    # Edge cases compare the label to the same unpadded grid contract the solver uses.
    from fluidSim_warp_blender51.blender.solver_settings import domain_bounds
    from fluidSim_warp_blender51.sim.quality_contract import (
        AuthoredQualitySettings,
        resolve_quality,
    )
    for divisions, tier in ((8, "DRAFT"), (17, "DRAFT"), (64, "FINAL")):
        settings.resolution_divisions = divisions
        scene.fluidsim.quality = tier
        solver = build_settings(scene)
        resolved = resolve_quality(AuthoredQualitySettings(voxel_size=solver.voxel_size), domain_bounds(scene))
        actual = max(resolved.unpadded_voxel_dimensions)
        note = ui_common.quality_effective_note(settings, "resolution_divisions")
        check(f"{divisions} {tier} resolution label", str(actual) + f" ({tier.title()})" in note
              if actual != divisions else not note, f"note={note!r}; unpadded={actual}")
    saved_domain = scene.fluidsim.domain_object
    scene.fluidsim.domain_object = None
    check("no domain has no settings owner", ui_scene._domain_owner(bpy.context) is None)
    for panel in ui_scene.DOMAIN_DRAW_SOURCE_CLASSES:
        if panel.poll(bpy.context):
            empty_layout = RecordingLayout()
            panel.draw(SimpleNamespace(layout=empty_layout), bpy.context)
            check("no domain bindings " + panel.bl_idname,
                  not any(r[0] == "prop" and r[1].bl_rna.identifier == settings.bl_rna.identifier
                          for r in empty_layout.records))
    scene.fluidsim.domain_object = saved_domain
    # Actual RNA edit while unrelated object selected, visible through both adapters.
    settings.max_substeps = 6
    check("edit owns designated domain", ui_scene._domain_owner(bpy.context).max_substeps == 6
          and other.fluidsim_domain.max_substeps != 6)
    scene.fluidsim.quality = "FINAL"
    props.spaces.active.context = "SCENE"
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
    select(domain)
    props.spaces.active.context = "PHYSICS"
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
    # Capture a native production panel, including the quality labels.
    props.spaces.active.context = "SCENE"
    with bpy.context.temp_override(area=props, region=next(r for r in props.regions if r.type == "WINDOW")):
        bpy.ops.wm.call_panel(name="FLUIDSIM_PT_sim_time", keep_open=False)
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
    capture = Path(__file__).resolve().parents[2] / "artifacts" / "ui41-layout-fixed-time-panel.png"
    bpy.ops.screen.screenshot(filepath=str(capture))
    RESULT["screenshot"] = str(capture)

def finish():
    try:
        run()
    except Exception as exc:  # noqa: BLE001 - retain failure evidence and close owned process
        RESULT["errors"].append(f"{type(exc).__name__}: {exc}")
        RESULT["traceback"] = traceback.format_exc()
    RESULT["ok"] = not RESULT["errors"]
    OUT.write_text(json.dumps(RESULT, indent=2), encoding="utf8")
    print("FLUIDSIM-41-UI-" + ("PASS" if RESULT["ok"] else "FAIL"))
    bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender(), first_interval=2.0)

bpy.app.timers.register(finish, first_interval=2.0)
