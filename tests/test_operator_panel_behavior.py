"""Focused UI/operator regression tests for Wave 2 remediation."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import struct
import sys
import textwrap
import time
import types
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _RnaProperty:
    """One registered RNA property's introspectable metadata."""

    def __init__(self, type, hard_min=None, hard_max=None):
        self.type = type
        self.hard_min = hard_min
        self.hard_max = hard_max


class _RnaPropertyGroup:
    """Model `bpy.types.PropertyGroup` assignment semantics.

    Real `IntProperty`/`FloatProperty` **silently clamp** an assignment into
    `[min, max]` and then fire `update=` with the *clamped* value.  Verified
    against Blender 5.1 with this add-on's own declarations:

        fs.max_particles = 120  -> 1024   (IntProperty(min=1024))
        fs.leaf_capacity  =  4  ->   64   (IntProperty(min=64))

    A `SimpleNamespace` stores whatever it is handed and fires nothing, so any
    production defect of the form "wrote a setting and never read it back" is
    invisible - the test asserts the value it just supplied.  Doubles for a
    library must reproduce the library's *surprising* behaviour; reproducing
    only the expected behaviour asserts nothing.
    """

    def __init__(self, properties, **values):
        object.__setattr__(self, "_rna_properties", dict(properties))
        object.__setattr__(self, "update_calls", [])
        object.__setattr__(
            self, "bl_rna", SimpleNamespace(properties=dict(properties)))
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        prop = self._rna_properties.get(name)
        if prop is None:  # unregistered attribute: plain Python storage
            object.__setattr__(self, name, value)
            return
        if prop.type == "INT":
            value = int(value)
        elif prop.type == "FLOAT":
            # RNA floats are single precision.
            value = struct.unpack("f", struct.pack("f", float(value)))[0]
        if prop.hard_min is not None:
            value = max(value, prop.hard_min)
        if prop.hard_max is not None:
            value = min(value, prop.hard_max)
        object.__setattr__(self, name, value)
        self.update_calls.append((name, value))


# Mirrors fluidSim_warp_blender51/properties.py:729 and :750.
CAPACITY_RNA = {
    "leaf_capacity": _RnaProperty("INT", 64, 262_144),
    "max_particles": _RnaProperty("INT", 1024, 50_000_000),
}


def _install_bpy_stub(monkeypatch, *, device="cuda:7"):
    fake_bpy = types.ModuleType("bpy")

    class _Operator:
        _poll_message = ""

        @classmethod
        def poll_message_set(cls, message):
            cls._poll_message = message

        def report(self, *_args, **_kwargs):
            return None

    class _Panel:
        pass

    class _AddonPreferences:
        pass

    def _prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(
        Operator=_Operator,
        Panel=_Panel,
        AddonPreferences=_AddonPreferences,
    )
    fake_bpy.props = SimpleNamespace(
        BoolProperty=lambda **kwargs: _prop(**kwargs),
        CollectionProperty=lambda **kwargs: _prop(**kwargs),
        EnumProperty=lambda **kwargs: _prop(**kwargs),
        FloatProperty=lambda **kwargs: _prop(**kwargs),
        FloatVectorProperty=lambda **kwargs: _prop(**kwargs),
        IntProperty=lambda **kwargs: _prop(**kwargs),
        PointerProperty=lambda **kwargs: _prop(**kwargs),
        StringProperty=lambda **kwargs: _prop(**kwargs),
    )
    fake_bpy.types.PropertyGroup = object
    fake_bpy.types.Text = object
    fake_bpy.types.Object = object
    fake_bpy.types.Material = object
    fake_bpy.types.Image = object
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(
            addons={
                "fluidSim_warp_blender51": SimpleNamespace(
                    preferences=SimpleNamespace(device=device)
                )
            }
        ),
        screen=SimpleNamespace(areas=[]),
    )
    fake_bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None,
        unregister_class=lambda _cls: None,
    )
    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_bpy.props)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    return fake_bpy


def _reload_module(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _pending_module():
    return importlib.import_module(
        "fluidSim_warp_blender51.blender.pending_settings")


def test_validation_operator_uses_configured_device(monkeypatch):
    _install_bpy_stub(monkeypatch, device="cuda:3")
    operators = _reload_module("fluidSim_warp_blender51.operators")

    seen = {}

    class _Result:
        name = "device route"
        passed = True

        def as_dict(self):
            return {"name": self.name, "passed": self.passed}

    fake_validation = types.ModuleType("fluidSim_warp_blender51.sim.validation")
    def _run_all(device):
        seen["device"] = device
        return [_Result()]

    fake_validation.run_all = _run_all
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.validation", fake_validation)
    import fluidSim_warp_blender51.sim as sim_pkg
    monkeypatch.setattr(sim_pkg, "validation", fake_validation, raising=False)
    monkeypatch.setattr(
        operators,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: SimpleNamespace(validation_results=[])),
    )
    context = SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(status="")))

    result = operators.FLUIDSIM_OT_run_validation().execute(context)

    assert result == {"FINISHED"}
    assert seen["device"] == "cuda:3"


def test_pending_operators_route_context_scene_and_contain_failures(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    scene = SimpleNamespace(fluidsim=SimpleNamespace())
    calls = []
    fake_bridge = SimpleNamespace(
        apply_pending_reinitialize=lambda actual: calls.append(("reinit", actual)) or True,
        apply_pending_reset=lambda actual: calls.append(("reset", actual)) or True,
        revert_pending_values=lambda actual: calls.append(("revert", actual)) or True,
    )
    monkeypatch.setattr(operators, "_pending", lambda: fake_bridge)
    context = SimpleNamespace(scene=scene)

    assert operators.FLUIDSIM_OT_apply_pending_reinitialize().execute(context) == {"FINISHED"}
    assert operators.FLUIDSIM_OT_apply_pending_reset().execute(context) == {"FINISHED"}
    assert operators.FLUIDSIM_OT_revert_pending().execute(context) == {"FINISHED"}
    assert calls == [("reinit", scene), ("reset", scene), ("revert", scene)]

    fake_bridge.apply_pending_reset = lambda _scene: (_ for _ in ()).throw(
        RuntimeError("blocked"))
    assert operators.FLUIDSIM_OT_apply_pending_reset().execute(context) == {"CANCELLED"}


def test_multi_frame_step_resyncs_scene_before_every_simulated_frame(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    scene = SimpleNamespace()
    sess = SimpleNamespace(active=True, error="")
    calls = []
    fake_bridge = SimpleNamespace(
        session=lambda: sess,
        sync_scene_to_solver=lambda actual_scene, actual_sess: calls.append(
            ("sync", actual_scene, actual_sess)),
        step_frame=lambda actual_scene, actual_sess: calls.append(
            ("step", actual_scene, actual_sess)),
    )
    monkeypatch.setattr(operators, "_bridge", lambda: fake_bridge)
    operator = operators.FLUIDSIM_OT_step()
    operator.frames = 3

    assert operator.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert [kind for kind, _scene, _sess in calls] == [
        "sync", "step", "sync", "step", "sync", "step"]
    assert all(actual_scene is scene and actual_sess is sess
               for _kind, actual_scene, actual_sess in calls)


def test_export_frame_invoke_opens_file_selector(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    calls = []
    context = SimpleNamespace(
        window_manager=SimpleNamespace(fileselect_add=lambda op: calls.append(op))
    )
    op = operators.FLUIDSIM_OT_export_frame()

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert calls == [op]


def test_undo_metadata_is_limited_to_mutating_operators(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")

    undoable = {
        operators.FLUIDSIM_OT_setup_scene,
        operators.FLUIDSIM_OT_tag_object,
        operators.FLUIDSIM_OT_snippet_add,
        operators.FLUIDSIM_OT_snippet_remove,
        operators.FLUIDSIM_OT_snippet_argument_add,
        operators.FLUIDSIM_OT_snippet_argument_remove,
        operators.FLUIDSIM_OT_clear_error,
    }
    not_undoable = {
        operators.FLUIDSIM_OT_initialize,
        operators.FLUIDSIM_OT_release,
        operators.FLUIDSIM_OT_run_validation,
        operators.FLUIDSIM_OT_run_benchmark,
        operators.FLUIDSIM_OT_write_report,
        operators.FLUIDSIM_OT_export_frame,
        operators.FLUIDSIM_OT_export_cache_sequence,
    }

    for cls in undoable:
        assert "UNDO" in getattr(cls, "bl_options", set()), cls.__name__
    for cls in not_undoable:
        assert "UNDO" not in getattr(cls, "bl_options", set()), cls.__name__


class _PermissiveSettings:
    """Settings stand-in that answers any property read.

    Panel greyout tests care about one toggle at a time; enumerating every
    unrelated property a panel happens to read would bury the assertion and
    break on every new control.  Unnamed properties read as ``0`` -- falsy, and
    safe for the ``> 0`` comparisons panels make.
    """

    def __init__(self, values):
        self.__dict__.update(values)

    def __getattr__(self, name):
        return 0


class _Layout:
    """Model `bpy.types.UILayout`, including greyout inheritance.

    `uiLayoutRow` / `uiLayoutColumn` copy the parent's `enabled` and `redalert`
    into the new sub-layout at *creation* time ("Children of disabled items
    automatically get disabled", `ui_item_init_from_parent`).  The consequence
    that matters for panel code is that greyout is a property of the layout a
    control was drawn into, not of the control -- so a master toggle drawn into
    the layout it disables locks itself off permanently, and no amount of
    reading `layout.props` will reveal that.  `prop_states` records the
    effective state each control was drawn under so tests can assert on it.

    Because inheritance is snapshotted at creation, the fake is only faithful
    for panels that set `enabled` before drawing into a sub-layout.  Every site
    in `panels.py` does; `test_panels_set_greyout_before_drawing_into_it`
    pins that so this fake cannot quietly drift out of agreement with Blender.
    """

    def __init__(self, _root=None, _box_scope=(), _enabled=True, _alert=False):
        if _root is None:
            self.labels = []
            self.operators = []
            self.props = []
            self.prop_calls = []
            self.prop_enum_calls = []
            self.prop_scopes = []
            self.prop_states = []
            self.popovers = []
            self._root = self
            self._next_box_id = 0
        else:
            self.labels = _root.labels
            self.operators = _root.operators
            self.props = _root.props
            self.prop_calls = _root.prop_calls
            self.prop_enum_calls = _root.prop_enum_calls
            self.prop_scopes = _root.prop_scopes
            self.prop_states = _root.prop_states
            self.popovers = _root.popovers
            self._root = _root
        self._box_scope = _box_scope
        self.enabled = _enabled
        self.alert = _alert

    def _record_state(self, name):
        self.prop_states.append(
            {"name": name, "enabled": self.enabled, "alert": self.alert})

    def drawn_enabled(self, name):
        """Effective greyout of `name`, or None if it was never drawn."""
        states = [s["enabled"] for s in self.prop_states if s["name"] == name]
        if not states:
            return None
        assert len(set(states)) == 1, f"{name} drawn with conflicting greyout"
        return states[0]

    def column(self, align=False):
        return _Layout(self._root, self._box_scope, self.enabled, self.alert)

    def row(self, align=False):
        return _Layout(self._root, self._box_scope, self.enabled, self.alert)

    def box(self):
        self._root._next_box_id += 1
        return _Layout(
            self._root,
            self._box_scope + (self._root._next_box_id,),
            self.enabled,
            self.alert,
        )

    def separator(self):
        return None

    def prop(self, *args, **kwargs):
        if len(args) >= 2:
            self.props.append(args[1])
            self.prop_calls.append({"name": args[1], **kwargs})
            self.prop_scopes.append({
                "name": args[1], "boxes": self._box_scope})
            self._record_state(args[1])
        return None

    def prop_enum(self, _data, name, value, **kwargs):
        self.props.append(name)
        self.prop_enum_calls.append({"name": name, "value": value, **kwargs})
        self.prop_scopes.append({"name": name, "boxes": self._box_scope})
        self._record_state(name)
        return None

    def label(self, text="", icon="NONE"):
        self.labels.append({"text": text, "icon": icon})

    def operator(self, operator_id, **kwargs):
        self.operators.append({"id": operator_id, **kwargs})
        return SimpleNamespace()

    def popover(self, **kwargs):
        self.popovers.append(kwargs)
        return None


def test_pending_banner_and_fixed_header_expose_real_actions(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(live=True),
        fluidsim_pending_action="REINIT_PENDING",
        fluidsim_pending_count=2,
        fluidsim_pending_reasons="Reinitialize: voxel_size\nReinitialize: max_particles",
        fluidsim_pending_blocked="",
    )
    layout = _Layout()

    panels._draw_pending_banner(layout, scene, show_actions=True)
    header = SimpleNamespace(layout=layout)
    panels._draw_header_interaction(header, SimpleNamespace(scene=scene))

    assert any(label["text"] == "2 change(s) staged for restart"
               for label in layout.labels)
    assert {op["id"] for op in layout.operators} >= {
        "fluidsim.apply_pending_reinitialize", "fluidsim.revert_pending"}
    assert any(op.get("text") == "Restart Now" for op in layout.operators)
    assert any(op.get("text") == "Discard Changes" for op in layout.operators)
    assert getattr(layout, "alert", False) is False
    assert layout.popovers == [{
        "panel": "FLUIDSIM_PT_interaction_popover", "text": "2", "icon": "INFO"}]


def test_header_registration_is_symmetric(monkeypatch):
    bpy = _install_bpy_stub(monkeypatch)

    class Header:
        callbacks = []

        @classmethod
        def append(cls, callback):
            cls.callbacks.append(callback)

        @classmethod
        def remove(cls, callback):
            if callback in cls.callbacks:
                cls.callbacks.remove(callback)

    bpy.types.VIEW3D_HT_header = Header
    panels = _reload_module("fluidSim_warp_blender51.panels")

    panels.register()
    assert Header.callbacks == [panels._draw_header_interaction]
    panels.unregister()
    assert Header.callbacks == []


def test_simulation_hides_playback_recovery_limit_in_advanced_foldout(monkeypatch):
    _install_bpy_stub(monkeypatch)
    _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        status="Solver ready", live=False, max_frame_catchup=8,
        playback_advanced_expanded=False,
        stat_particles=0, stat_voxels=0, stat_step_ms=0.0, stat_substeps=0,
        time_scale=1.0, cfl=1.0, min_substeps=1, max_substeps=4,
        use_scene_gravity=True, gravity=(0.0, 0.0, -9.81),
    )
    panel = panels.FLUIDSIM_PT_simulation()
    layout = _Layout()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert "playback_advanced_expanded" in layout.props
    assert "max_frame_catchup" not in layout.props
    toggle = next(
        call for call in layout.prop_calls
        if call["name"] == "playback_advanced_expanded")
    assert toggle["text"] == "Advanced Playback"
    assert toggle["icon"] == "TRIA_RIGHT"
    assert toggle["emboss"] is False

    fs.playback_advanced_expanded = True
    expanded = _Layout()
    panel.layout = expanded
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert expanded.props.index("playback_advanced_expanded") < (
        expanded.props.index("max_frame_catchup"))
    recovery = next(
        call for call in expanded.prop_calls
        if call["name"] == "max_frame_catchup")
    assert recovery["text"] == "Maximum Playback Recovery Frames"
    expanded_toggle = next(
        call for call in expanded.prop_calls
        if call["name"] == "playback_advanced_expanded")
    assert expanded_toggle["icon"] == "TRIA_DOWN"


def test_simulation_panel_shows_persistent_error_with_clear_action(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            status="Solver error: boom",
            live=False,
            stat_particles=0,
            stat_voxels=0,
            stat_step_ms=0.0,
            stat_substeps=0,
            time_scale=1.0,
            cfl=3.0,
            min_substeps=1,
            max_substeps=4,
            use_scene_gravity=True,
            gravity=(0.0, 0.0, -9.81),
        )
    )
    scene_bridge.set_error(scene, "frame failed")

    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout
    panel.draw(SimpleNamespace(scene=scene))

    assert {"text": "frame failed", "icon": "ERROR"} in layout.labels
    assert any(op["id"] == "fluidsim.clear_error" for op in layout.operators)


def test_simulation_panel_uses_only_verified_capacity_recommendation_flow(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        status="Init failed", live=False, stat_particles=0, stat_voxels=0,
        stat_step_ms=0.0, stat_substeps=0, time_scale=1.0, cfl=3.0,
        min_substeps=1, max_substeps=4, use_scene_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    ))
    scene_bridge.set_error(scene, "CUDA error 2: out of memory")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert not any(
        op["id"] == "fluidsim.apply_safe_settings" for op in layout.operators
    )
    assert not any(
        op["id"] == "fluidsim.initialize"
        and op.get("text") == "Ignore Estimate & Initialize"
        for op in layout.operators
    )
    assert any("verified" in label["text"] for label in layout.labels)


def test_simulation_panel_hides_unverified_memory_bypass(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        status="Init failed", live=False, stat_particles=0, stat_voxels=0,
        stat_step_ms=0.0, stat_substeps=0, time_scale=1.0, cfl=3.0,
        min_substeps=1, max_substeps=4, use_scene_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    ))
    scene_bridge.set_error(scene, "FluidSim memory safeguard stopped initialization")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert not any(
        op["id"] == "fluidsim.initialize"
        and op.get("text") == "Ignore Estimate & Initialize"
        for op in layout.operators
    )


def test_initialize_operator_passes_explicit_memory_override(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    seen = []
    solver = SimpleNamespace(
        device="cuda:0",
        grid=SimpleNamespace(capacity=1),
        particles=SimpleNamespace(capacity=1),
    )
    bridge = SimpleNamespace(
        initialize_session=lambda scene, **kwargs: (
            seen.append((scene, kwargs)) or SimpleNamespace(solver=solver)
        )
    )
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    op = operators.FLUIDSIM_OT_initialize()
    op.ignore_memory_safeguard = True

    assert op.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert seen == [(scene, {"ignore_memory_safeguard": True})]


def test_capacity_warning_offers_approval_gated_recommended_option(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    suggestion = SimpleNamespace(
        key="preserve_simulation",
        label="Preserve authored domain and resolution",
        changes={"max_particles": 120},
        tradeoffs=("Raises only fixed GPU capacity.",),
    )
    plan = SimpleNamespace(
        particle_demand=SimpleNamespace(value=120, quality=SimpleNamespace(value="exact")),
        leaf_demand=SimpleNamespace(value=1, quality=SimpleNamespace(value="exact")),
        estimated_bytes=1024,
        budget_bytes=2048,
        blockers=(SimpleNamespace(
            code="PARTICLE_CAPACITY", required=120, available=100,
            message="particle capacity"),),
        suggestions=(suggestion,),
    )
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        status="Capacity warning", live=False, stat_particles=0, stat_voxels=0,
        stat_step_ms=0.0, stat_substeps=0, time_scale=1.0, cfl=3.0,
        min_substeps=1, max_substeps=4, use_scene_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    ))
    scene_bridge.set_error(scene, "source capacity blocked")
    monkeypatch.setattr(
        scene_bridge, "session",
        lambda: SimpleNamespace(
            active=False, capacity_plan=plan,
            current_error=lambda actual: "source capacity blocked"),
    )
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_simulation()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert any(
        op["id"] == "fluidsim.apply_capacity_suggestion"
        and op.get("text") == "Review Recommended Changes & Run"
        for op in layout.operators
    )
    assert not any(
        op["id"] == "fluidsim.apply_safe_settings" for op in layout.operators)
    assert any("Nothing has been changed" in label["text"]
               for label in layout.labels)


def test_capacity_option_changes_nothing_until_approval_then_retries(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="preserve_simulation",
        label="Preserve authored domain and resolution",
        changes={"max_particles": 4_000_000, "leaf_capacity": 16_384},
        estimated_bytes=1024,
        budget_bytes=2048,
    )
    plan = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=())
    fs = _RnaPropertyGroup(
        CAPACITY_RNA,
        max_particles=2_000_000,
        leaf_capacity=8192,
        domain_object=None,
        status="Capacity warning",
    )
    scene = SimpleNamespace(fluidsim=fs)
    initialized = []
    bridge = SimpleNamespace(
        session=lambda: SimpleNamespace(capacity_plan=plan),
        refresh_capacity_plan=lambda actual: plan,
        clear_error=lambda actual: None,
        initialize_session=lambda actual: (
            initialized.append(actual) or SimpleNamespace(
                solver=SimpleNamespace(device="cuda:0"))),
    )
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    dialogs = []
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda actual, width: (
                dialogs.append((actual, width)) or {"RUNNING_MODAL"})),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "preserve_simulation"

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert dialogs == [(op, 560)]
    assert fs.max_particles == 2_000_000
    assert fs.leaf_capacity == 8192
    assert op.expected_scene_signature == "scene-v1"
    assert op.execute(context) == {"FINISHED"}
    assert fs.max_particles == 4_000_000
    assert fs.leaf_capacity == 16_384
    assert initialized == [scene]
    # RNA fires update= on every accepted write; approval must not skip them.
    assert ("max_particles", 4_000_000) in fs.update_calls
    assert ("leaf_capacity", 16_384) in fs.update_calls


def test_capacity_option_refuses_when_rna_clamps_the_approved_value(monkeypatch):
    """A silently clamped write must not be reported as an approved change.

    `capacity_planner` can propose a `leaf_capacity` below the RNA floor of 64
    (`balanced` takes `min(base, prod(leaf_dims))`, which is small on a compact
    domain).  RNA clamps the write upward, so the scene ends up reserving MORE
    memory than the dialog's estimate promised - the exact quantity the
    approval flow exists to bound.  Applying the option must fail loudly and
    roll back, not report success against a value that was never stored.
    """
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="balanced", label="Balanced",
        changes={"leaf_capacity": 32},  # below the declared min of 64
        estimated_bytes=1024, budget_bytes=2048)
    plan = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=())
    fs = _RnaPropertyGroup(
        CAPACITY_RNA, max_particles=2_000_000, leaf_capacity=8192,
        domain_object=None, status="Capacity warning")
    scene = SimpleNamespace(fluidsim=fs)
    initialized = []
    bridge = SimpleNamespace(
        session=lambda: SimpleNamespace(capacity_plan=plan),
        refresh_capacity_plan=lambda actual: plan,
        clear_error=lambda actual: None,
        initialize_session=lambda actual: (
            initialized.append(actual) or SimpleNamespace(
                solver=SimpleNamespace(device="cuda:0"))),
    )
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda actual, width: {"RUNNING_MODAL"}),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "balanced"

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert op.execute(context) == {"CANCELLED"}
    # Rolled back to the authored value, and no solver started against a
    # capacity the user never approved.
    assert fs.leaf_capacity == 8192
    assert initialized == []
    assert "not applied" in fs.status.lower()
    assert "leaf_capacity" in fs.status


def test_realtime_display_groups_surface_and_liquid_particles(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        liquid_view_mode="SURFACE_PARTICLES",
        show_surface=True,
        surface_enabled=True,
        particle_radius_scale=1.2,
        surface_splat_radius=2,
        surface_resolution_mode="MULTIPLIER",
        surface_resolution_multiplier=2.0,
        surface_voxel_size=0.0,
        surface_leaf_capacity_scale=8.0,
        surface_max_dense_dim=512,
        surface_update_cadence="FRAME",
        live_surface_publish_cadence=1,
        surface_build_lowres_sdf=False,
        surface_advanced_expanded=False,
        surface_filter_type="BOX",
        surface_filter_iterations=1,
        surface_filter_strength=0.7,
        surface_dilate_iterations=0,
        surface_dilate_strength=1.0,
        surface_erode_iterations=0,
        surface_erode_strength=1.0,
        surface_smooth_type="BOX",
        surface_smooth_iterations=1,
        surface_smooth_strength=0.7,
        surface_attribute_mode="NONE",
        surface_material=None,
        show_particles=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
        particle_material=None,
    )
    context = SimpleNamespace(scene=SimpleNamespace(fluidsim=fs))
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_realtime_display()
    panel.layout = layout

    panel.draw(context)

    main_controls = {
        "liquid_view_mode",
        "surface_enabled",
        "particle_radius_scale",
        "surface_splat_radius",
        "surface_resolution_mode",
        "surface_resolution_multiplier",
        "surface_update_cadence",
        "live_surface_publish_cadence",
        "surface_filter_type",
        "surface_filter_iterations",
        "surface_filter_strength",
        "surface_dilate_iterations",
        "surface_dilate_strength",
        "surface_erode_iterations",
        "surface_erode_strength",
        "surface_smooth_type",
        "surface_smooth_iterations",
        "surface_smooth_strength",
        "show_surface",
        "surface_attribute_mode",
        "surface_material",
        "show_particles",
        "particle_display_fraction",
        "particle_display_scale",
        "particle_attribute_mode",
        "particle_material",
    }
    advanced_controls = {
        "surface_leaf_capacity_scale",
        "surface_max_dense_dim",
        "surface_build_lowres_sdf",
    }
    assert layout.props[0] == "liquid_view_mode"
    assert main_controls.issubset(layout.props)
    assert advanced_controls.isdisjoint(layout.props)
    assert layout.labels == [
        {"text": "Live / Targeted Refresh", "icon": "INFO"},
        {"text": "Surface Mesh", "icon": "MESH_DATA"},
        {"text": "Liquid Particles", "icon": "PARTICLE_DATA"},
    ]
    advanced_toggle = next(
        call for call in layout.prop_calls
        if call["name"] == "surface_advanced_expanded")
    assert advanced_toggle["text"] == "Advanced Surface"
    assert advanced_toggle["icon"] == "TRIA_RIGHT"
    assert advanced_toggle["emboss"] is False
    scopes = {call["name"]: call["boxes"] for call in layout.prop_scopes}
    assert scopes["surface_enabled"]
    assert scopes["surface_advanced_expanded"] == scopes["surface_enabled"]
    assert scopes["show_particles"] != scopes["surface_enabled"]
    assert not hasattr(panels, "FLUIDSIM_PT_surface_advanced")

    fs.surface_advanced_expanded = True
    expanded_layout = _Layout()
    panel.layout = expanded_layout
    panel.draw(context)
    assert advanced_controls.issubset(expanded_layout.props)
    assert expanded_layout.props.index("surface_material") < (
        expanded_layout.props.index("surface_advanced_expanded"))
    assert expanded_layout.props.index("surface_build_lowres_sdf") < (
        expanded_layout.props.index("show_particles"))
    expanded_scopes = {
        call["name"]: call["boxes"] for call in expanded_layout.prop_scopes}
    for name in advanced_controls:
        assert expanded_scopes[name] == expanded_scopes["surface_enabled"]
    assert expanded_scopes["show_particles"] != expanded_scopes["surface_enabled"]
    expanded_toggle = next(
        call for call in expanded_layout.prop_calls
        if call["name"] == "surface_advanced_expanded")
    assert expanded_toggle["icon"] == "TRIA_DOWN"


def test_whitewater_panel_starts_with_simulation_toggle_and_owns_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        whitewater_enabled=True,
        max_whitewater=1000,
        ww_churn_gain=1.0,
        ww_churn_decay=1.0,
        ww_churn_threshold=0.5,
        ww_emission_rate=2.0,
        ww_min_speed=0.5,
        ww_life_min=1.0,
        ww_life_max=3.0,
        ww_bubble_drag=4.0,
        ww_spray_drag=0.2,
        show_whitewater=True,
        whitewater_display_scale=1.0,
        whitewater_attribute_mode="NONE",
        whitewater_material=None,
    )
    expected = {
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
        "max_whitewater",
        "ww_churn_gain",
        "ww_churn_decay",
        "ww_churn_threshold",
        "ww_emission_rate",
        "ww_min_speed",
        "ww_life_min",
        "ww_life_max",
        "ww_bubble_drag",
        "ww_spray_drag",
    }

    layout = _Layout()
    panel = panels.FLUIDSIM_PT_surface()
    panel.layout = layout
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert layout.props[0] == "whitewater_enabled"
    assert expected.issubset(layout.props)
    assert "Viewport / Render" not in {label["text"] for label in layout.labels}
    assert layout.labels == [
        {"text": "Reinitialize Required", "icon": "ERROR"},
        {"text": "Live", "icon": "INFO"},
    ]
    other_sources = "\n".join(inspect.getsource(cls.draw) for cls in (
        panels.FLUIDSIM_PT_realtime_display,
        panels.FLUIDSIM_PT_output,
    ))
    assert all(name not in other_sources for name in expected)


def test_panels_move_static_help_to_tooltips_but_keep_structural_tiers(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    panel_source = inspect.getsource(panels)
    prohibited_explanations = {
        "Solid Volume requires a watertight mesh",
        "Direct Append adds owned points once; it is not volume fill",
        "Targets colliders and closed domain walls",
        "Configure these in Blender Physics properties",
        "Mesh vertices, PointCloud points, or the first particle system",
        "Compatibility direct append; this is not volume fill",
        "FIELD operations never emit liquid",
    }

    assert all(text not in panel_source for text in prohibited_explanations)
    assert "def _draw_interaction_tier" in panel_source
    assert '_draw_interaction_tier(layout, "Live / Targeted Refresh")' in inspect.getsource(
        panels.FLUIDSIM_PT_realtime_display.draw)
    assert '_draw_interaction_tier(layout, "Live")' in inspect.getsource(
        panels.FLUIDSIM_PT_pressure.draw)


@pytest.mark.parametrize("height_enabled", [False, True])
def test_unrelated_height_fill_keeps_original_disabled_controls_visible(
        monkeypatch, height_enabled):
    """Height-fill settings must grey out, never vanish.

    Hiding them behind `if not fs.height_enabled: return` would make the panel
    change height as the toggle flips and hide the authored values; greying
    keeps them readable.  Asserted on the drawn layout rather than on the
    source text, so a rewrite that still greys correctly stays green.
    """
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        height_enabled=height_enabled, height_back_mode="FLAT",
        height_error="")
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_domain_height_fill()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    governed = ("height_base", "height_emission_mode", "height_velocity",
                "height_emit_particles", "particles_per_voxel",
                "height_back_mode")
    for name in governed:
        assert layout.drawn_enabled(name) is height_enabled, name
    # The master toggle lives outside the layout it governs, so it can always
    # be switched back on.
    assert layout.drawn_enabled("height_enabled") is True


@pytest.mark.parametrize(
    "panel_name, toggle, fs_fields",
    [
        ("FLUIDSIM_PT_domain_height_fill", "height_enabled",
         {"height_back_mode": "FLAT", "height_error": ""}),
        ("FLUIDSIM_PT_surface", "whitewater_enabled",
         {"show_whitewater": False}),
        ("FLUIDSIM_PT_surface", "show_whitewater",
         {"whitewater_enabled": False}),
        ("FLUIDSIM_PT_realtime_display", "surface_enabled",
         {"liquid_view_mode": "SURFACE", "surface_resolution_mode": "MULTIPLIER",
          "show_surface": False, "show_particles": False,
          "surface_advanced_expanded": True}),
        ("FLUIDSIM_PT_realtime_display", "show_surface",
         {"liquid_view_mode": "SURFACE", "surface_resolution_mode": "MULTIPLIER",
          "surface_enabled": False, "show_particles": False,
          "surface_advanced_expanded": False}),
        ("FLUIDSIM_PT_realtime_display", "show_particles",
         {"liquid_view_mode": "SURFACE", "surface_resolution_mode": "MULTIPLIER",
          "surface_enabled": False, "show_surface": False,
          "surface_advanced_expanded": False}),
    ],
)
def test_master_toggles_are_not_self_locking(
        monkeypatch, panel_name, toggle, fs_fields):
    """A toggle drawn into the sub-layout it disables can never be undone.

    Greyout is inherited by the layout, not evaluated per control, so
    `col.enabled = fs.x; col.prop(fs, "x")` renders `x` unclickable the moment
    it goes false -- a one-way door with no keyboard escape.  Every master
    toggle must be drawn into a layout its own value does not govern.
    """
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = _PermissiveSettings({**fs_fields, toggle: False})
    layout = _Layout()
    panel = getattr(panels, panel_name)()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(
        fluidsim=fs, fluidsim_pending_action="", fluidsim_pending_count=0,
        fluidsim_pending_reasons="", fluidsim_pending_blocked="")))

    assert layout.drawn_enabled(toggle) is True, (
        f"{panel_name}.{toggle} greys itself out when off")


def test_panels_set_greyout_before_drawing_into_a_sub_layout(monkeypatch):
    """Guard the assumption `_Layout` is built on.

    Blender snapshots `enabled` into a sub-layout when the sub-layout is
    created, so `sub = col.column(); sub.prop(...); sub.enabled = False` does
    not behave like the same three statements reordered.  `_Layout` models the
    snapshot; if panel code ever assigns greyout after drawing into the target,
    the fake and Blender part ways and the greyout tests above go quietly
    wrong.  Fail here instead, where the cause is legible.
    """
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    tree = ast.parse(inspect.getsource(panels))
    offenders = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        # ast.walk is breadth-first, so collect (line, event) and replay in
        # source order -- the whole point of this check is the ordering.
        # Rank orders events sharing a line: a name rebound to a fresh
        # sub-layout (`row = layout.row()`, reused several times per function)
        # starts a new layout with a clean slate, exactly as in Blender.
        events = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if (node.func.attr in {"prop", "prop_enum", "operator", "label"}
                        and isinstance(target, ast.Name)):
                    events.append((node.lineno, 1, "draw", target.id, None))
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        events.append((node.lineno, 0, "rebind", tgt.id, None))
                    elif (isinstance(tgt, ast.Attribute)
                            and tgt.attr in {"enabled", "alert"}
                            and isinstance(tgt.value, ast.Name)):
                        events.append(
                            (node.lineno, 2, "set", tgt.value.id, tgt.attr))
        drawn_into = set()
        for lineno, _rank, kind, name, attr in sorted(events):
            if kind == "rebind":
                drawn_into.discard(name)
            elif kind == "draw":
                drawn_into.add(name)
            elif name in drawn_into:
                offenders.append(
                    f"{func.name}: {name}.{attr} set at line {lineno}, "
                    "after drawing into it")
    assert offenders == []


def test_collider_panel_has_only_per_object_material_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = inspect.getsource(panels.FLUIDSIM_PT_colliders.draw)

    assert 'col.prop(pg, "collider_friction")' in source
    assert 'col.prop(pg, "collider_restitution")' in source
    assert 'col.prop(fs, "friction")' not in source
    assert 'col.prop(fs, "restitution")' not in source


def test_realtime_property_callbacks_request_surface_or_output_refresh(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    from fluidSim_warp_blender51.blender.pending_settings import SceneUpdate

    calls = []
    scene = SimpleNamespace()
    fake_bridge = SimpleNamespace(
        request_scene_update=lambda scene, action: calls.append((scene, action))
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "pending_settings", fake_bridge, raising=False)

    properties._request_surface_refresh(None, SimpleNamespace(scene=scene))
    properties._request_output_refresh(None, SimpleNamespace(scene=scene))

    assert calls == [
        (scene, SceneUpdate.SURFACE),
        (scene, SceneUpdate.OUTPUT),
    ]


def test_hot_solver_property_callback_requests_live_update(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    from fluidSim_warp_blender51.blender.pending_settings import SceneUpdate

    calls = []
    scene = SimpleNamespace()
    fake_bridge = SimpleNamespace(
        request_scene_update=lambda scene, action: calls.append((scene, action))
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "pending_settings", fake_bridge, raising=False)

    properties._request_live(None, SimpleNamespace(scene=scene))

    assert calls == [(scene, SceneUpdate.LIVE)]


def test_live_solver_settings_apply_without_resetting_particle_state(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    particle_state = object()
    solver = SimpleNamespace(
        settings=SimpleNamespace(flip_ratio=0.95, cfl=1.0),
        particles=particle_state,
    )
    sess = scene_bridge.session()
    sess.solver = solver
    reset_calls = []
    monkeypatch.setattr(sess, "reset", lambda _scene: reset_calls.append("reset"))
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(flip_ratio=0.25, cfl=2.0, status=""),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.LIVE)

    assert solver.settings.flip_ratio == 0.25
    assert solver.settings.cfl == 2.0
    assert solver.particles is particle_state
    assert reset_calls == []
    assert scene.fluidsim.status == ""


def test_live_setting_errors_are_atomic_contained_and_visible(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(sample_type="previous", flip_ratio=0.9)
    solver = SimpleNamespace(settings=settings, particles=object())
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            sample_type="NOT_A_REAL_ENUM", flip_ratio=0.2, status="Frame 8"),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.LIVE)

    assert sess.solver is solver
    assert settings.sample_type == "previous"
    assert settings.flip_ratio == 0.9
    assert "NOT_A_REAL_ENUM" in sess.current_error(scene)
    assert scene.fluidsim.status.startswith("Live settings error:")


def test_liquid_sampling_is_reset_tier_and_does_not_mutate_live_reference(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(
        particles_per_voxel=8.0,
        spacetime_enabled=True,
        spacetime_reference_mass=0.0,
    )
    solver = SimpleNamespace(
        settings=settings,
        particles=object(),
        _st_reference_mass_cached=4.0,
    )
    solver._calibrate_st_reference_mass = lambda: (_ for _ in ()).throw(
        AssertionError("existing calibration should be density-rescaled"))
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(particles_per_voxel=12.0, status="Frame 4"),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )
    sess.scene_identity = scene_bridge._scene_identity(scene)

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.LIVE)

    assert settings.particles_per_voxel == 8.0
    assert solver._st_reference_mass_cached == 4.0
    assert "particles_per_voxel" in _pending_module()._RESET_SETTING_FIELDS
    assert "particles_per_voxel" not in _pending_module()._LIVE_SETTING_FIELDS


def test_live_gravity_uses_scene_or_custom_branch_without_reset(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(gravity=(0.0, 0.0, 0.0))
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(settings=settings, particles=object())
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            gravity=(1.0, 2.0, 3.0), use_scene_gravity=True, status="stats"),
        use_gravity=True,
        gravity=(0.0, 0.0, -4.0),
    )

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.LIVE)
    assert settings.gravity == (0.0, 0.0, -4.0)

    scene.fluidsim.use_scene_gravity = False
    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.LIVE)
    assert settings.gravity == (1.0, 2.0, 3.0)
    assert scene.fluidsim.status == "stats"


def test_inactive_live_update_status_is_truthful(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    scene_bridge.session().solver = None
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.LIVE)

    assert scene.fluidsim.status == "Live setting stored; applies on initialization"


def test_timeline_reset_reapplies_settings_and_resets_session_behavior(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    from fluidSim_warp_blender51.blender import simulation_admission
    import fluidSim_warp_blender51.blender.sources as sources
    import fluidSim_warp_blender51.blender.output_publication as output_publication
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    events = []
    solver = SimpleNamespace(
        device="cuda:0",
        settings=SolverSettings(
            max_particles=4096, flip_ratio=0.95, spacetime_seed=1),
        particles=SimpleNamespace(capacity=4096),
        sourcing=SimpleNamespace(plans=(), last_apply_result={}),
        reset=lambda: events.append((
            "reset", solver.settings.flip_ratio,
            solver.settings.spacetime_seed)),
        set_sources=lambda values: events.append(("sources", tuple(values))),
        set_source_plans=lambda values: events.append(("plans", tuple(values))),
        set_colliders=lambda values: events.append(("colliders", tuple(values))),
        set_force_fields=lambda values: events.append(("forces", tuple(values))),
        set_height_source=lambda value: events.append(("height", value)),
        set_custom_snippets=lambda values: events.append(
            ("snippets", tuple(values))) or (),
        release=lambda: events.append("old-release"),
    )
    sess = scene_bridge.session()
    sess.solver = solver
    sess.scene_identity = None
    sess.start_frame = 99
    sess.cache_written_start = 2
    sess.cache_written_end = 8
    sess.cache_settings_records = {2: object()}
    sess.last_frame = 8
    sess.frame_stats = [{"frame": 8}]

    class Inputs:
        def clear(self):
            events.append("inputs-clear")

        def materialize_device_meshes(self, descriptors, colliders, device):
            events.append((
                "materialize", tuple(descriptors), tuple(colliders), device))

    class Snapshot:
        frame_start = 3
        settings = SolverSettings(
            device="cuda:0", max_particles=4096,
            flip_ratio=0.2, spacetime_seed=77)
        capacity_plan = SimpleNamespace(blockers=())
        descriptors = ("source",)
        extra_plans = ("plan",)
        colliders = ("collider",)
        force_fields = ("force",)
        height_source = "height-source"
        snippets = ("snippet",)
        seeds = ("seed",)
        configured_device = "cuda:0"

        def __init__(self):
            self.inputs = Inputs()
            self.release_calls = 0

        def transfer_scene_inputs(self):
            value, self.inputs = self.inputs, None
            return value

        def release_transients(self):
            self.release_calls += 1
            if self.inputs is not None:
                self.inputs.clear()
                self.inputs = None

    snapshot = Snapshot()
    preflight_calls = []

    def preflight(_owner, _scene, **kwargs):
        preflight_calls.append(kwargs)
        return snapshot, {"safe": True}, None

    monkeypatch.setattr(
        simulation_admission.SimulationAdmission, "_preflight", preflight)
    candidate = SimpleNamespace(
        device="cuda:0", settings=snapshot.settings,
        particles=SimpleNamespace(capacity=4096),
    )
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission,
        "_prepare_solver_replacement",
        lambda *_args, **_kwargs: (
            events.extend(("candidate-configure", "candidate-seeds"))
            or (candidate, ())),
    )
    monkeypatch.setattr(
        output_publication, "update_outputs",
        lambda actual_scene, owner: events.append(
            ("outputs", actual_scene, owner)))
    applied = []
    monkeypatch.setattr(
        _pending_module(), "capture_applied_state",
        lambda actual_scene, owner, **kwargs: applied.append(
            (actual_scene, owner, kwargs)))
    depsgraph = object()
    scene = SimpleNamespace(
        frame_start=3,
        fluidsim=SimpleNamespace(
            flip_ratio=0.2, spacetime_seed=77, snippets=[], status="",
            gravity=(0.0, 0.0, -4.0), use_scene_gravity=False),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )

    sess.reset(scene, depsgraph=depsgraph)

    assert preflight_calls == [{
        "depsgraph": depsgraph,
        "runtime_capacity": 4096,
        "operation": "RESET",
    }]
    assert "candidate-configure" in events
    assert "candidate-seeds" in events
    assert "old-release" in events
    assert sess.solver is candidate
    assert sess.solver.settings.flip_ratio == 0.2
    assert sess.solver.settings.spacetime_seed == 77
    assert any(event[0] == "outputs" for event in events if isinstance(event, tuple))
    assert sess.start_frame == 3
    assert sess.cache_written_start is None
    assert sess.cache_written_end is None
    assert sess.cache_settings_records == {}
    assert sess.last_frame == -1
    assert sess.frame_stats == []
    assert scene.fluidsim.status == "Simulation reset"
    assert snapshot.release_calls == 1
    assert applied == [(scene, sess, {"reset_only": True})]


def test_solver_property_interaction_tiers_are_mechanically_exhaustive(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(properties.FLUIDSIM_PG_settings)))
    callbacks = {}
    for node in tree.body[0].body:
        if not (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Call)):
            continue
        update = next(
            (kw.value for kw in node.annotation.keywords if kw.arg == "update"),
            None,
        )
        if isinstance(update, ast.Name):
            callbacks[node.target.id] = update.id

    live_props = {name for name, callback in callbacks.items()
                  if callback == "_request_live"}
    assert live_props == set(_pending_module()._LIVE_SETTING_FIELDS) | {
        "gravity", "use_scene_gravity"}
    declared_callbacks = {
        "_on_live_changed", "_request_rebuild", "_request_live",
        "_request_reset", "_request_surface_refresh",
        "_request_whitewater_refresh", "_apply_liquid_view_mode",
        "_sync_liquid_view_mode", "_request_output_refresh",
        "_request_particle_capacity", "_on_cache_playback_changed",
    }
    assert set(callbacks.values()) == declared_callbacks

    allocation = {"voxel_size", "leaf_capacity", "max_particles", "max_whitewater"}
    assert allocation.isdisjoint(
        _pending_module()._LIVE_SETTING_FIELDS | _pending_module()._RESET_SETTING_FIELDS)
    assert {
        name for name, callback in callbacks.items()
        if callback == "_request_particle_capacity"
    } == {
        "max_particles", "auto_grow_particles",
        "particle_growth_failure_policy",
    }
    assert _pending_module()._HOT_SETTING_FIELDS.isdisjoint(_pending_module()._RESET_SETTING_FIELDS)
    assert _pending_module()._HOT_SETTING_FIELDS.isdisjoint(_pending_module()._SURFACE_SETTING_FIELDS)
    assert _pending_module()._LIVE_REFRESH_SETTING_FIELDS == set()
    assert "particles_per_voxel" in _pending_module()._RESET_SETTING_FIELDS
    enum_sources = {source for source, _enum in _pending_module()._enum_setting_fields().values()}
    appliable = set(_pending_module()._DIRECT_SETTING_FIELDS) | enum_sources
    assert (_pending_module()._LIVE_SETTING_FIELDS | _pending_module()._RESET_SETTING_FIELDS).issubset(
        appliable)
    assert _pending_module()._RESET_SETTING_FIELDS.issubset(
        {name for name, callback in callbacks.items()
         if callback == "_request_reset"})
    assert _pending_module()._SURFACE_SETTING_FIELDS.issubset(
        {name for name, callback in callbacks.items()
         if callback == "_request_surface_refresh"})
    _pending_module().validate_transition_registry()


def test_live_reconcile_is_bounded_after_registry_warmup(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    settings = SimpleNamespace(flip_ratio=0.9, cfl=1.0)
    sess = bridge.session()
    sess.solver = SimpleNamespace(settings=settings)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(flip_ratio=0.2, cfl=2.0),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )
    assert _pending_module()._apply_realtime_solver_settings(scene, sess)

    started = time.perf_counter()
    for _ in range(500):
        assert _pending_module()._apply_realtime_solver_settings(scene, sess)
    elapsed = time.perf_counter() - started

    assert elapsed / 500 < 0.001


def test_live_property_lifecycle_initializes_once_and_updates_status(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    playback = _reload_module("fluidSim_warp_blender51.blender.playback")

    init_calls = []
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    sess = scene_bridge.session()
    sess.solver = None

    def fake_initialize(scene_arg):
        init_calls.append(scene_arg)
        sess.solver = object()
        return sess

    monkeypatch.setattr(sess, "initialize", fake_initialize)

    playback.set_live_enabled(scene, True)
    playback.set_live_enabled(scene, True)
    playback.set_live_enabled(scene, False)

    assert init_calls == [scene]
    assert sess.active
    assert scene.fluidsim.status == "Live simulation disabled"


def test_live_property_callback_toggles_playback_directly(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")

    calls = []
    scene = SimpleNamespace()
    fake_playback = SimpleNamespace(
        set_live_enabled=lambda scene, enabled: calls.append((scene, enabled))
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "playback", fake_playback, raising=False)

    properties._on_live_changed(SimpleNamespace(live=True, status=""), SimpleNamespace(scene=scene))

    assert calls == [(scene, True)]


def test_playback_set_live_enabled_updates_status(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    playback = _reload_module("fluidSim_warp_blender51.blender.playback")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    sess = scene_bridge.session()

    def fake_initialize(_scene):
        sess.solver = object()
        return sess

    monkeypatch.setattr(sess, "initialize", fake_initialize)

    playback.set_live_enabled(scene, True)
    assert scene.fluidsim.status == "Live simulation enabled - press play"

    playback.set_live_enabled(scene, False)
    assert scene.fluidsim.status == "Live simulation disabled"


def test_frame_change_solver_error_disables_live_and_releases_session(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    playback = _reload_module("fluidSim_warp_blender51.blender.playback")
    scene = SimpleNamespace(
        frame_current=2,
        fluidsim=SimpleNamespace(
            live=True,
            cache_playback_enabled=False,
            status="",
        ),
    )
    sess = scene_bridge.session()
    sess.solver = object()
    sess.start_frame = 1
    sess.last_frame = 1

    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sess,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    playback._on_frame_change(scene, depsgraph=object())

    assert scene.fluidsim.live is False
    assert scene_bridge.session().active is False
    assert scene_bridge.current_error(scene) == "RuntimeError('boom')"
    assert scene.fluidsim.status == "Solver error: boom"


def test_preferences_warp_probe_is_cached_until_explicit_refresh(monkeypatch):
    _install_bpy_stub(monkeypatch)
    preferences = _reload_module("fluidSim_warp_blender51.preferences")
    calls = []
    fake_warp = SimpleNamespace(
        init=lambda: calls.append("init"),
        is_cuda_available=lambda: True,
        get_device=lambda _name: SimpleNamespace(name="Test GPU"),
        config=SimpleNamespace(version="test"),
    )
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    preferences._WARP_PROBE_RESULT = None

    first = preferences._probe_warp()
    second = preferences._probe_warp()
    assert first == second == (True, "Warp test - Test GPU")
    assert calls == ["init"]

    preferences._probe_warp(force=True)
    assert calls == ["init", "init"]


def test_addon_unregister_releases_runtime_handlers(monkeypatch):
    fake_bpy = _install_bpy_stub(monkeypatch)
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(
            frame_change_post=[],
            render_pre=[],
            render_post=[],
        )
    )
    addon = _reload_module("fluidSim_warp_blender51")
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    pending_settings = _reload_module(
        "fluidSim_warp_blender51.blender.pending_settings")
    playback = _reload_module("fluidSim_warp_blender51.blender.playback")
    render = _reload_module("fluidSim_warp_blender51.blender.render")
    viewport_overlay = _reload_module("fluidSim_warp_blender51.blender.viewport_overlay")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    preferences = _reload_module("fluidSim_warp_blender51.preferences")
    properties = _reload_module("fluidSim_warp_blender51.properties")

    calls = []
    monkeypatch.setattr(
        viewport_overlay, "unregister_overlay",
        lambda: calls.append("viewport"))
    monkeypatch.setattr(render, "unregister_handlers", lambda: calls.append("render"))
    monkeypatch.setattr(playback, "unregister_handlers", lambda: calls.append("playback"))
    monkeypatch.setattr(
        pending_settings, "unregister_handlers", lambda: calls.append("pending"))
    monkeypatch.setattr(panels, "unregister", lambda: calls.append("panels"))
    monkeypatch.setattr(operators, "unregister", lambda: calls.append("operators"))
    monkeypatch.setattr(preferences, "unregister", lambda: calls.append("preferences"))
    monkeypatch.setattr(properties, "unregister", lambda: calls.append("properties"))

    addon.unregister()

    assert calls == [
        "viewport",
        "render",
        "pending",
        "playback",
        "panels",
        "operators",
        "preferences",
        "properties",
    ]


def test_surface_refresh_updates_active_solver_and_output(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    output_calls = []
    surface_calls = []
    fake_viewport = SimpleNamespace(
        prepare_outputs=lambda scene: None,
        update_outputs=lambda scene, sess, **kwargs: output_calls.append(
            (scene, sess, kwargs)),
    )
    import fluidSim_warp_blender51.blender.output_publication as output_publication

    monkeypatch.setattr(
        output_publication, "prepare_outputs", fake_viewport.prepare_outputs)
    monkeypatch.setattr(output_publication, "update_outputs", fake_viewport.update_outputs)
    solver = SimpleNamespace(
        settings=SimpleNamespace(),
        rebuild_surface=lambda: surface_calls.append("rebuilt"),
    )
    sess = scene_bridge.session()
    sess.solver = solver
    fs = SimpleNamespace(
        surface_enabled=False,
        particle_radius_scale=2.0,
        surface_splat_radius=2,
        surface_resolution_mode="MULTIPLIER",
        surface_resolution_multiplier=2.0,
        surface_voxel_size=0.0,
        surface_leaf_capacity_scale=8.0,
        surface_max_dense_dim=512,
        surface_update_cadence="FRAME",
        live_surface_publish_cadence=4,
        surface_build_lowres_sdf=True,
        surface_filter_type="BOX",
        surface_filter_iterations=1,
        surface_filter_strength=0.7,
        surface_dilate_iterations=1,
        surface_dilate_strength=0.2,
        surface_erode_iterations=4,
        surface_erode_strength=0.45,
        surface_smooth_type="MEAN_CURVATURE_FLOW",
        surface_smooth_iterations=3,
        surface_smooth_strength=0.6,
        status="",
    )
    scene = SimpleNamespace(fluidsim=fs)

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.SURFACE)

    assert solver.settings.surface_enabled is False
    assert solver.settings.particle_radius_scale == 2.0
    assert solver.settings.surface_dilate_iterations == 1
    assert solver.settings.surface_dilate_strength == 0.2
    assert solver.settings.surface_erode_iterations == 4
    assert solver.settings.surface_erode_strength == 0.45
    assert solver.settings.surface_smooth_type.name == "MEAN_CURVATURE_FLOW"
    assert solver.settings.surface_smooth_iterations == 3
    assert solver.settings.surface_smooth_strength == 0.6
    assert surface_calls == ["rebuilt"]
    assert output_calls == [(scene, sess, {"force_surface": True})]


def test_whitewater_refresh_updates_active_solver_and_output(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    output_calls = []
    fake_viewport = SimpleNamespace(
        prepare_outputs=lambda scene: None,
        update_outputs=lambda scene, sess: output_calls.append((scene, sess)),
    )
    import fluidSim_warp_blender51.blender.output_publication as output_publication

    monkeypatch.setattr(
        output_publication, "prepare_outputs", fake_viewport.prepare_outputs)
    monkeypatch.setattr(output_publication, "update_outputs", fake_viewport.update_outputs)
    solver = SimpleNamespace(settings=SimpleNamespace(whitewater_enabled=True))
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(whitewater_enabled=False, status="")
    )

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.WHITEWATER)

    assert solver.settings.whitewater_enabled is False
    assert output_calls == [(scene, sess)]


def test_output_refresh_updates_existing_outputs_without_active_solver(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")

    refresh_calls = []
    fake_viewport = SimpleNamespace(
        refresh_existing_output_settings=lambda scene: refresh_calls.append(scene)
    )
    import fluidSim_warp_blender51.blender.output_publication as output_publication

    monkeypatch.setattr(
        output_publication,
        "refresh_existing_output_settings",
        fake_viewport.refresh_existing_output_settings,
    )
    scene_bridge.session().solver = None
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))

    _pending_module().request_scene_update(scene, _pending_module().SceneUpdate.OUTPUT)

    assert refresh_calls == [scene]
    assert scene.fluidsim.status == "Viewport display settings changed"


def test_clear_error_operator_resets_session_and_scene_error(monkeypatch):
    _install_bpy_stub(monkeypatch)
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")

    class _Scene(dict):
        pass

    scene = _Scene()
    scene.fluidsim = SimpleNamespace(status="Solver error: boom")
    scene_bridge.session().solver = object()
    scene_bridge.set_error(scene, "boom")
    assert scene_bridge.current_error(scene) == "boom"

    result = operators.FLUIDSIM_OT_clear_error().execute(SimpleNamespace(scene=scene))

    assert result == {"FINISHED"}
    assert scene_bridge.current_error(scene) == ""
    assert scene.fluidsim.status == "Solver ready"


def test_output_panel_exposes_cache_sequence_and_playback_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        liquid_view_mode="SURFACE",
        show_surface=True,
        surface_attribute_mode="NONE",
        show_particles=False,
        particle_display_fraction=1.0,
        particle_attribute_mode="NONE",
        show_whitewater=False,
        whitewater_attribute_mode="NONE",
        surface_material=None,
        whitewater_material=None,
        output_collection_name="FluidSim Output",
        show_overlay=True,
        cache_enabled=False,
        cache_playback_enabled=True,
        cache_path="//fluidsim_cache/",
        cache_use_custom_range=True,
        cache_frame_start=3,
        cache_frame_end=5,
    )
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_output()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert "cache_path" in layout.props
    assert "cache_playback_enabled" in layout.props
    assert "cache_use_custom_range" in layout.props
    assert "cache_frame_start" in layout.props
    assert "cache_frame_end" in layout.props
    assert any(op["id"] == "fluidsim.export_cache_sequence" for op in layout.operators)


def _phase1_object(role, **overrides):
    values = dict(
        role=role,
        enabled=True,
        geometry_mode="ANALYTIC",
        shape="BOX",
        mesh_object=None,
        mesh_evaluation_mode="FIRST_SIM_FRAME",
        mesh_surface_thickness=0.0,
        sink_region_mode="SOLID",
        mesh_error="",
        emission_mode="CONTINUOUS_INFLOW",
        emit_rate=8.0,
        initial_velocity=(0.0, 0.0, 0.0),
        use_object_velocity=True,
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        use_lifespan=False,
        life_min=2.0,
        life_max=4.0,
        force_type="OFF",
        force_strength=1.0,
        force_direction=(0.0, 0.0, 1.0),
        collider_friction=0.0,
        collider_restitution=0.0,
        transfer_velocity=True,
    )
    values.update(overrides)
    return SimpleNamespace(
        name=f"{role.title()} Object", type="MESH",
        fluidsim=SimpleNamespace(**values), display_type="TEXTURED",
        hide_render=False,
    )


def test_phase1_source_draw_exposes_real_lifecycle_and_hides_compatibility(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = _phase1_object("SOURCE")
    fs = SimpleNamespace(
        height_enabled=True, height_base=0.3,
        height_velocity=(0.0, 0.0, 0.0), height_emit_particles=True,
    )
    scene = SimpleNamespace(fluidsim=fs, objects=[source])
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_liquid_emitters()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene, active_object=source))

    assert {
        "emission_mode", "source_sampling_mode", "emit_rate", "geometry_mode"
    }.issubset(layout.props)
    assert {
        ("ANALYTIC", "Analytic Volume"),
        ("MESH", "Mesh Surface Shell"),
    }.issubset({(call["value"], call["text"])
                for call in layout.prop_enum_calls})
    assert "source_mode" not in layout.props
    assert "height_back_mode" not in layout.props
    assert any(call == {"name": "emit_rate", "text": "Legacy Absolute Target"}
               for call in layout.prop_calls)
    assert any(op["id"] == "fluidsim.tag_liquid_emitter"
               for op in layout.operators)
    assert not any(op["id"] == "fluidsim.inject_particles"
                   for op in layout.operators)


def test_phase1_particle_attributes_are_source_only_and_collapsed(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = _phase1_object("SOURCE", use_lifespan=True)
    context = SimpleNamespace(active_object=source)
    assert panels.FLUIDSIM_PT_liquid_emitter_attributes.poll(context)
    assert "DEFAULT_CLOSED" in panels.FLUIDSIM_PT_liquid_emitter_attributes.bl_options
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_liquid_emitter_attributes()
    panel.layout = layout
    panel.draw(context)
    assert {"temperature", "viscosity", "density", "use_lifespan",
            "life_min", "life_max"}.issubset(layout.props)
    assert any(call == {"name": "density", "text": "Density Attribute"}
               for call in layout.prop_calls)
    assert not panels.FLUIDSIM_PT_liquid_emitter_attributes.poll(
        SimpleNamespace(active_object=_phase1_object("SINK")))


def test_phase1_sink_uses_shared_region_editor_in_sink_workflow(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    sink = _phase1_object(
        "SINK", geometry_mode="MESH", mesh_error="Mesh must be watertight")
    scene = SimpleNamespace(objects=[sink], fluidsim=SimpleNamespace())
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_liquid_sinks()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene, active_object=sink))

    assert {"geometry_mode", "mesh_object", "mesh_evaluation_mode",
            "sink_region_mode"}.issubset(layout.props)
    assert "mesh_surface_thickness" not in layout.props
    assert {
        ("ANALYTIC", "Analytic Volume"), ("MESH", "Mesh")
    }.issubset({(call["value"], call["text"])
                for call in layout.prop_enum_calls})
    assert {"text": "Mesh must be watertight", "icon": "ERROR"} in layout.labels
    assert any(op["id"] == "fluidsim.tag_liquid_sink"
               for op in layout.operators)


def test_phase1_force_editor_requires_force_role_and_tag_action(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    fs = SimpleNamespace(
        directional_force=(0.0, 0.0, 0.0),
        omnidirectional_force_strength=0.0,
        omnidirectional_force_center=(0.0, 0.0, 0.0),
        omnidirectional_force_radius=1.0,
        force_diffusion=0.0,
        dissipation_enabled=False,
        dissipation_attribute="TEMPERATURE",
        dissipation_rate=0.0,
        dissipation_target=0.0,
        attraction_enabled=False,
        attraction_strength=0.0,
        attraction_max_distance=1.0,
    )
    scene = SimpleNamespace(fluidsim=fs)

    force_layout = _Layout()
    panel = panels.FLUIDSIM_PT_forces()
    panel.layout = force_layout
    panel.draw(SimpleNamespace(scene=scene, active_object=_phase1_object("FORCE")))
    assert {"enabled", "shape", "force_type"}.issubset(force_layout.props)
    assert any(op["id"] == "fluidsim.tag_force_region"
               for op in force_layout.operators)

    source_layout = _Layout()
    panel.layout = source_layout
    panel.draw(SimpleNamespace(scene=scene, active_object=_phase1_object("SOURCE")))
    assert "force_type" not in source_layout.props
    assert any("Select a tagged Force Region" in label["text"]
               for label in source_layout.labels)


def test_phase1_tag_operator_supports_force_and_has_disabled_state(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    obj = _phase1_object("NONE")
    op = operators.FLUIDSIM_OT_tag_object()
    op.role = "FORCE"
    context = SimpleNamespace(selected_objects=[obj])

    assert operators.FLUIDSIM_OT_tag_object.poll(context)
    assert op.execute(context) == {"FINISHED"}
    assert obj.fluidsim.role == "FORCE"
    assert obj.display_type == "WIRE"
    assert obj.hide_render is True
    assert not operators.FLUIDSIM_OT_tag_object.poll(
        SimpleNamespace(selected_objects=[]))
    assert "Select at least one object" in operators.FLUIDSIM_OT_tag_object._poll_message


def test_phase3_field_volume_input_is_drawn_once_when_region_and_payload_use_volume(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    field = _phase1_object(
        "FIELD", source_mode="FIELD", field_operation="SCALAR",
        field_region_kind="VOLUME_GRID", field_value_source="VOLUME_GRID",
        volume_input_object=None, volume_scalar_grid="temperature",
        volume_vector_grid="", volume_vector_x_grid="",
        volume_vector_y_grid="", volume_vector_z_grid="",
        blend_mode="REPLACE", field_amount=1.0,
        source_status_message="", source_status_severity="",
        source_status_code="", source_status_capacity_limited=False)
    layout = _Layout()
    panel = panels.FLUIDSIM_PT_advanced_field_sources(); panel.layout = layout
    panel.draw(SimpleNamespace(
        scene=SimpleNamespace(objects=[field], fluidsim=SimpleNamespace()),
        active_object=field))
    assert [call["name"] for call in layout.prop_calls].count(
        "volume_input_object") == 1


def test_phase3_aggregate_emission_is_not_broadcast_as_per_source_truth():
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    def obj(pointer):
        pg = SimpleNamespace(
            role="SOURCE", source_mode="PARTICLES", enabled=True,
            source_status_code="", source_status_message="",
            source_status_severity="", source_status_emitted=0,
            source_status_capacity_limited=False)
        return SimpleNamespace(fluidsim=pg, as_pointer=lambda: pointer)

    first, second = obj(1), obj(2)
    inputs = SceneInputs()
    inputs.update_source_status(
        SimpleNamespace(objects=[first, second]), (),
        {"emitted": 12, "capacity_limited": True})
    assert first.fluidsim.source_status_emitted == 0
    assert second.fluidsim.source_status_emitted == 0
    second.fluidsim.enabled = False
    inputs.update_source_status(
        SimpleNamespace(objects=[first, second]), (),
        {"emitted": 12, "capacity_limited": True})
    assert first.fluidsim.source_status_emitted == 12
    assert first.fluidsim.source_status_capacity_limited is True


def test_phase1_point_injection_poll_is_truthful(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    sess = SimpleNamespace(active=True)
    monkeypatch.setattr(operators, "_bridge", lambda: SimpleNamespace(session=lambda: sess))

    mesh = SimpleNamespace(type="MESH", particle_systems=[])
    assert operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=mesh))
    unsupported = SimpleNamespace(type="LIGHT", particle_systems=[])
    assert not operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=unsupported))
    assert "mesh vertices" in operators.FLUIDSIM_OT_inject_particles._poll_message
    sess.active = False
    assert not operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=mesh))
    assert "Initialize the solver" in operators.FLUIDSIM_OT_inject_particles._poll_message


def test_phase1_point_injection_capacity_refusal_is_recoverable_cancel(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import output_publication, sources

    result = SimpleNamespace(
        admitted=False, emitted=0, required=5, available=4,
        capacity_limited=True, total=1)
    solver = SimpleNamespace(sourcing=SimpleNamespace(
        last_injection_result=result))
    sess = SimpleNamespace(active=True, solver=solver)
    fatal_errors = []
    bridge = SimpleNamespace(
        session=lambda: sess,
        set_error=lambda *_args: fatal_errors.append(_args))
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    monkeypatch.setattr(sources, "inject_object_particles", lambda *_args: 0)
    viewport_updates = []
    monkeypatch.setattr(
        output_publication, "update_outputs",
        lambda *_args: viewport_updates.append(_args))

    reports = []
    op = operators.FLUIDSIM_OT_inject_particles()
    op.report = lambda level, message: reports.append((level, message))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status="Solver ready"))
    obj = SimpleNamespace(name="Seed", type="MESH", particle_systems=[])

    assert op.execute(SimpleNamespace(
        scene=scene, active_object=obj)) == {"CANCELLED"}
    assert not viewport_updates
    assert not fatal_errors
    assert scene.fluidsim.status == "Solver ready"
    assert "5 slots" in reports[-1][1]
    assert "4" in reports[-1][1]
    assert "Max Particles" in reports[-1][1]


def test_phase1_empty_point_injection_clears_refusal_and_finishes(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import output_publication, sources

    sourcing_state = SimpleNamespace(last_injection_result=SimpleNamespace(
        admitted=False, required=5, available=4))

    class Solver:
        sourcing = sourcing_state

        def inject_particle_packets_exact(self, packets):
            assert tuple(packets) == ()
            result = SimpleNamespace(
                admitted=True, emitted=0, required=0, available=4,
                capacity_limited=False, total=0)
            self.sourcing.last_injection_result = result
            return result

    sess = SimpleNamespace(active=True, solver=Solver())
    monkeypatch.setattr(
        operators, "_bridge", lambda: SimpleNamespace(session=lambda: sess))
    monkeypatch.setattr(sources, "prepare_object_particles", lambda *_args: None)
    viewport_updates = []
    monkeypatch.setattr(
        output_publication, "update_outputs",
        lambda *_args: viewport_updates.append(_args))

    reports = []
    op = operators.FLUIDSIM_OT_inject_particles()
    op.report = lambda level, message: reports.append((level, message))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status="Solver ready"))
    obj = SimpleNamespace(name="Empty", type="MESH", particle_systems=[])

    assert op.execute(SimpleNamespace(
        scene=scene, active_object=obj)) == {"FINISHED"}
    assert len(viewport_updates) == 1
    assert reports[-1][0] == {"INFO"}
    assert "Injected 0 particles" in reports[-1][1]


def test_capacity_suggestion_poll_and_dialog_are_scene_owned(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="safe", label="Safe", changes={"max_particles": 12},
        estimated_bytes=1, budget_bytes=2)
    plan = SimpleNamespace(
        scene_signature="identical-content", suggestions=(suggestion,), blockers=())
    first = SimpleNamespace(fluidsim=SimpleNamespace(max_particles=8))
    second = SimpleNamespace(fluidsim=SimpleNamespace(max_particles=8))
    bridge.session()._set_capacity_plan(first, plan)
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)

    assert operators.FLUIDSIM_OT_apply_capacity_suggestion.poll(
        SimpleNamespace(scene=first)) is True
    assert operators.FLUIDSIM_OT_apply_capacity_suggestion.poll(
        SimpleNamespace(scene=second)) is False

    dialogs = []
    context_first = SimpleNamespace(
        scene=first,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda op, width: (
                dialogs.append((op, width)) or {"RUNNING_MODAL"})),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "safe"
    assert op.invoke(context_first, None) == {"RUNNING_MODAL"}

    refreshed = []
    monkeypatch.setattr(
        bridge, "refresh_capacity_plan",
        lambda scene: refreshed.append(scene) or plan)
    assert op.execute(SimpleNamespace(scene=second)) == {"CANCELLED"}
    assert refreshed == []
    assert second.fluidsim.max_particles == 8


def test_capacity_dialog_rechecks_grown_pool_reinitialize_with_authored_model(
        monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    fs = SimpleNamespace(max_particles=1024, status="")
    scene = SimpleNamespace(fluidsim=fs)
    sess = bridge.session()
    sess.solver = SimpleNamespace(
        settings=SimpleNamespace(device="cuda:0", max_particles=65_536),
        particles=SimpleNamespace(capacity=65_536),
        device="cuda:0",
    )
    sess.scene_identity = bridge._scene_identity(scene)
    from fluidSim_warp_blender51.blender import simulation_admission
    monkeypatch.setattr(
        simulation_admission, "build_settings",
        lambda _scene: SimpleNamespace(
            device="cuda:0", max_particles=int(fs.max_particles)))
    from fluidSim_warp_blender51.sim import memory_safety

    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {
        "gpu_total_bytes": 16 * 1024**3,
        "gpu_free_bytes": 12 * 1024**3,
    })
    planned = []

    def plan_scene(
            _scene, settings, _resources, *,
            additional_transaction_bytes=0):
        capacity = int(settings.max_particles)
        planned.append((capacity, additional_transaction_bytes))
        suggestion = SimpleNamespace(
            key="preserve_simulation", label="Preserve Simulation",
            changes={"max_particles": 2048}, estimated_bytes=1,
            budget_bytes=2)
        blocked = capacity < 2048
        return SimpleNamespace(capacity_plan=SimpleNamespace(
            scene_signature=f"capacity-{capacity}",
            suggestions=(suggestion,) if blocked else (),
            blockers=((SimpleNamespace(code="PARTICLE_CAPACITY"),)
                      if blocked else ()),
        ))

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    original = plan_scene(
        scene, SimpleNamespace(max_particles=1024), {}).capacity_plan
    sess._set_capacity_plan(scene, original, operation="REINITIALIZE")
    expected_signature = sess.capacity_plan_signature_for_scene(scene)
    initialized = []
    monkeypatch.setattr(
        bridge, "initialize_session",
        lambda actual: initialized.append(actual) or sess)
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda _op, width: {"RUNNING_MODAL"}),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "preserve_simulation"

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert op.expected_scene_signature == expected_signature
    assert op.execute(context) == {"FINISHED"}
    assert planned[-1][0] == 1024
    assert planned[-1][1] == memory_safety.estimate_solver_bytes(
        sess.solver.settings)
    assert fs.max_particles == 2048
    assert initialized == [scene]


def test_capacity_dialog_rechecks_same_scene_signature_before_mutation(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="safe", label="Safe", changes={"max_particles": 12},
        estimated_bytes=1, budget_bytes=2)
    original = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=())
    changed = SimpleNamespace(
        scene_signature="scene-v2", suggestions=(suggestion,), blockers=())
    fs = SimpleNamespace(max_particles=8, status="")
    scene = SimpleNamespace(fluidsim=fs)
    bridge.session()._set_capacity_plan(scene, original)
    monkeypatch.setattr(operators, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda _op, width: {"RUNNING_MODAL"}),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "safe"
    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    monkeypatch.setattr(
        bridge, "refresh_capacity_plan",
        lambda actual: bridge.session()._set_capacity_plan(actual, changed))

    assert op.execute(context) == {"CANCELLED"}
    assert fs.max_particles == 8
    assert "changed" in fs.status.lower()


def test_panel_hierarchy_has_seven_task_groups_and_advanced_capacities(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    expected_top = {
        "Simulation", "Quality", "Inputs", "Physics",
        "Surface & Whitewater", "Output & Cache", "Advanced",
    }
    actual_top = {
        cls.bl_label for cls in panels.CLASSES
        if issubclass(cls, panels.FluidSimPanel)
        and "bl_parent_id" not in cls.__dict__
    }
    assert actual_top == expected_top
    assert panels.FLUIDSIM_PT_advanced.bl_options == {"DEFAULT_CLOSED"}
    assert panels.FLUIDSIM_PT_domain.bl_parent_id == "FLUIDSIM_PT_advanced"
    assert panels.FLUIDSIM_PT_particles.bl_parent_id == "FLUIDSIM_PT_advanced"
    assert panels.FLUIDSIM_PT_liquid_emitters.bl_parent_id == "FLUIDSIM_PT_inputs"
    assert panels.FLUIDSIM_PT_pressure.bl_parent_id == "FLUIDSIM_PT_physics"
    assert panels.FLUIDSIM_PT_realtime_display.bl_parent_id == "FLUIDSIM_PT_surface"


def test_basic_quality_panel_exposes_only_semantic_controls(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    source = inspect.getsource(panels.FLUIDSIM_PT_quality.draw)
    for field in (
            "voxel_size", "particles_per_voxel", "sample_type", "flip_ratio",
            "cfl", "max_substeps"):
        assert f'prop(fs, "{field}"' in source
    for technical_capacity in ("max_particles", "max_active_leaves", "max_whitewater"):
        assert technical_capacity not in source
    assert "Requires Reinitialize" in source
    assert "Requires Reset" in source


def test_long_capacity_copy_wraps_instead_of_truncating(monkeypatch):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module("fluidSim_warp_blender51.panels")
    layout = _Layout()
    text = "This capacity warning contains enough detail to exceed one narrow panel line."
    panels._draw_wrapped(layout, text, icon="INFO", width=30)
    assert len(layout.labels) >= 2
    assert "".join(label["text"] for label in layout.labels).replace(" ", "") == text.replace(" ", "")
