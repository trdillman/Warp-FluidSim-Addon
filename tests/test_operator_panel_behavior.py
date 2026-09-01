"""Operator and runtime regressions retained after the Properties UI migration.

Scene/Object authoring and the combined 1.0.255 View3D/Physics presentation are
exercised by the Properties UI suites and release integration contract.
"""

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
        object.__setattr__(self, "bl_rna", SimpleNamespace(properties=dict(properties)))
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

    class _UIList:
        pass

    class _AddonPreferences:
        pass

    def _prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(
        Operator=_Operator,
        Panel=_Panel,
        UIList=_UIList,
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
    prefix = f"{name}."
    operator_domain_prefix = "fluidSim_warp_blender51.operator_domains"
    for loaded_name in tuple(sys.modules):
        if (
            loaded_name == name
            or (name != "fluidSim_warp_blender51" and loaded_name.startswith(prefix))
            or (
                name == "fluidSim_warp_blender51.operators"
                and loaded_name.startswith(operator_domain_prefix)
            )
        ):
            sys.modules.pop(loaded_name, None)
    return importlib.import_module(name)


def test_reloading_addon_root_preserves_loaded_submodule_identities(monkeypatch):
    _install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.sim import enums

    _reload_module("fluidSim_warp_blender51")

    assert sys.modules["fluidSim_warp_blender51.sim.enums"] is enums


def _pending_module():
    return importlib.import_module("fluidSim_warp_blender51.blender.pending_settings")


def test_validation_operator_uses_configured_device(monkeypatch):
    _install_bpy_stub(monkeypatch, device="cuda:3")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    preferences = _reload_module("fluidSim_warp_blender51.preferences")
    monkeypatch.setattr(
        preferences,
        "probe_runtime",
        lambda: SimpleNamespace(cuda_devices=(SimpleNamespace(alias="cuda:3"),)),
    )

    seen = {}

    class _Result:
        name = "device route"
        passed = True

        def as_dict(self):
            return {"name": self.name, "passed": self.passed}

    fake_validation = types.ModuleType("fluidSim_warp_blender51.diagnostics.validation")

    def _run_all(device):
        seen["device"] = device
        return [_Result()]

    fake_validation.run_all = _run_all
    monkeypatch.setitem(
        sys.modules,
        "fluidSim_warp_blender51.diagnostics.validation",
        fake_validation,
    )
    import fluidSim_warp_blender51.diagnostics as diagnostics_pkg

    monkeypatch.setattr(diagnostics_pkg, "validation", fake_validation, raising=False)
    monkeypatch.setattr(
        operators._operator_runtime,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: SimpleNamespace(validation_results=[])),
    )
    context = SimpleNamespace(
        scene=SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    )

    result = operators.FLUIDSIM_OT_run_validation().execute(context)

    assert result == {"FINISHED"}
    assert seen["device"] == "cuda:3"


def test_pending_operators_route_context_scene_and_contain_failures(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    scene = SimpleNamespace(fluidsim=SimpleNamespace())
    calls = []
    fake_bridge = SimpleNamespace(
        apply_pending_reinitialize=lambda actual: (
            calls.append(("reinit", actual)) or True
        ),
        apply_pending_reset=lambda actual: calls.append(("reset", actual)) or True,
        revert_pending_values=lambda actual: calls.append(("revert", actual)) or True,
    )
    monkeypatch.setattr(operators._operator_runtime, "_pending", lambda: fake_bridge)
    context = SimpleNamespace(scene=scene)

    assert operators.FLUIDSIM_OT_apply_pending_reinitialize().execute(context) == {
        "FINISHED"
    }
    assert operators.FLUIDSIM_OT_apply_pending_reset().execute(context) == {"FINISHED"}
    assert operators.FLUIDSIM_OT_revert_pending().execute(context) == {"FINISHED"}
    assert calls == [("reinit", scene), ("reset", scene), ("revert", scene)]

    fake_bridge.apply_pending_reset = lambda _scene: (_ for _ in ()).throw(
        RuntimeError("blocked")
    )
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
            ("sync", actual_scene, actual_sess)
        ),
        step_frame=lambda actual_scene, actual_sess: calls.append(
            ("step", actual_scene, actual_sess)
        ),
    )
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: fake_bridge)
    operator = operators.FLUIDSIM_OT_step()
    operator.frames = 3

    assert operator.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert [kind for kind, _scene, _sess in calls] == [
        "sync",
        "step",
        "sync",
        "step",
        "sync",
        "step",
    ]
    assert all(
        actual_scene is scene and actual_sess is sess
        for _kind, actual_scene, actual_sess in calls
    )


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


def test_liquid_view_hidden_state_round_trips_both_visibility_toggles(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    refreshes = []
    monkeypatch.setattr(
        properties,
        "_request_output_refresh",
        lambda _self, _context: refreshes.append(True),
    )
    state = SimpleNamespace(
        liquid_view_mode="SURFACE", show_surface=False, show_particles=False
    )

    assert "HIDDEN" in {item[0] for item in properties.LIQUID_VIEW_ITEMS}
    properties._sync_liquid_view_mode(state, None)
    assert state.liquid_view_mode == "HIDDEN"

    state.show_surface = True
    state.show_particles = True
    properties._apply_liquid_view_mode(state, None)
    assert (state.show_surface, state.show_particles) == (False, False)
    assert len(refreshes) == 2


def test_legacy_snippet_rna_persists_permission_and_type_defaults(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")

    argument_schema = properties.FLUIDSIM_PG_snippet_argument.__annotations__
    snippet_schema = properties.FLUIDSIM_PG_snippet.__annotations__
    permission = argument_schema["field_permission"]
    snippet_type = snippet_schema["snippet_type"]

    assert "CustomKernelFieldPermission" in permission
    assert "READ_WRITE" in permission
    assert "CustomKernelType" in snippet_type
    assert "KERNEL" in snippet_type


def test_legacy_apply_forward_never_mutates_preserved_snippets(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    legacy = [SimpleNamespace(kernel_name="preserved")]
    state = SimpleNamespace(snippets=legacy, active_snippet=0)
    context = SimpleNamespace(scene=SimpleNamespace(fluidsim=state))
    operators.bpy.ops = SimpleNamespace(
        fluidsim=SimpleNamespace(effect_apply_stack=lambda: {"FINISHED"})
    )
    before = (tuple(state.snippets), state.active_snippet)

    assert operators.FLUIDSIM_OT_snippet_apply().execute(context) == {"FINISHED"}
    assert (tuple(state.snippets), state.active_snippet) == before


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
            {"name": name, "enabled": self.enabled, "alert": self.alert}
        )

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
            self.prop_scopes.append({"name": args[1], "boxes": self._box_scope})
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
        # `enabled`/`alert` record the effective greyout/red-alert state of the
        # sub-layout the operator was drawn into (operators, unlike props, carry
        # no `prop_states` entry). Additive keys; existing tests read `["id"]`
        # and specific kwargs only.
        self.operators.append(
            {"id": operator_id, "enabled": self.enabled, "alert": self.alert, **kwargs}
        )
        return SimpleNamespace()

    def popover(self, **kwargs):
        self.popovers.append(kwargs)
        return None


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
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    op = operators.FLUIDSIM_OT_initialize()
    op.ignore_memory_safeguard = True

    assert op.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert seen == [(scene, {"ignore_memory_safeguard": True})]


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
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=()
    )
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
            initialized.append(actual)
            or SimpleNamespace(solver=SimpleNamespace(device="cuda:0"))
        ),
    )
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)
    dialogs = []
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda actual, width: (
                dialogs.append((actual, width)) or {"RUNNING_MODAL"}
            )
        ),
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
        key="balanced",
        label="Balanced",
        changes={"leaf_capacity": 32},  # below the declared min of 64
        estimated_bytes=1024,
        budget_bytes=2048,
    )
    plan = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=()
    )
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
            initialized.append(actual)
            or SimpleNamespace(solver=SimpleNamespace(device="cuda:0"))
        ),
    )
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda actual, width: {"RUNNING_MODAL"}
        ),
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
                if node.func.attr in {
                    "prop",
                    "prop_enum",
                    "operator",
                    "label",
                } and isinstance(target, ast.Name):
                    events.append((node.lineno, 1, "draw", target.id, None))
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        events.append((node.lineno, 0, "rebind", tgt.id, None))
                    elif (
                        isinstance(tgt, ast.Attribute)
                        and tgt.attr in {"enabled", "alert"}
                        and isinstance(tgt.value, ast.Name)
                    ):
                        events.append((node.lineno, 2, "set", tgt.value.id, tgt.attr))
        drawn_into = set()
        for lineno, _rank, kind, name, attr in sorted(events):
            if kind == "rebind":
                drawn_into.discard(name)
            elif kind == "draw":
                drawn_into.add(name)
            elif name in drawn_into:
                offenders.append(
                    f"{func.name}: {name}.{attr} set at line {lineno}, "
                    "after drawing into it"
                )
    assert offenders == []


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
    sess.scene_identity = id(scene)

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
            sample_type="NOT_A_REAL_ENUM", flip_ratio=0.2, status="Frame 8"
        ),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )
    sess.scene_identity = id(scene)

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
        AssertionError("existing calibration should be density-rescaled")
    )
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
            gravity=(1.0, 2.0, 3.0), use_scene_gravity=True, status="stats"
        ),
        use_gravity=True,
        gravity=(0.0, 0.0, -4.0),
    )
    sess.scene_identity = id(scene)

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
        settings=SolverSettings(max_particles=4096, flip_ratio=0.95, spacetime_seed=1),
        particles=SimpleNamespace(capacity=4096),
        sourcing=SimpleNamespace(plans=(), last_apply_result={}),
        reset=lambda: events.append(
            ("reset", solver.settings.flip_ratio, solver.settings.spacetime_seed)
        ),
        set_sources=lambda values: events.append(("sources", tuple(values))),
        set_source_plans=lambda values: events.append(("plans", tuple(values))),
        set_colliders=lambda values: events.append(("colliders", tuple(values))),
        set_force_fields=lambda values: events.append(("forces", tuple(values))),
        set_height_source=lambda value: events.append(("height", value)),
        set_custom_snippets=lambda values: (
            events.append(("snippets", tuple(values))) or ()
        ),
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
            events.append(("materialize", tuple(descriptors), tuple(colliders), device))

    class Snapshot:
        frame_start = 3
        settings = SolverSettings(
            device="cuda:0", max_particles=4096, flip_ratio=0.2, spacetime_seed=77
        )
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
        simulation_admission.SimulationAdmission, "_preflight", preflight
    )
    candidate = SimpleNamespace(
        device="cuda:0",
        settings=snapshot.settings,
        particles=SimpleNamespace(capacity=4096),
    )
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission,
        "_prepare_solver_replacement",
        lambda *_args, **_kwargs: (
            events.extend(("candidate-configure", "candidate-seeds")) or (candidate, ())
        ),
    )
    monkeypatch.setattr(
        output_publication,
        "update_outputs",
        lambda actual_scene, owner: events.append(("outputs", actual_scene, owner)),
    )
    applied = []
    monkeypatch.setattr(
        _pending_module(),
        "capture_applied_state",
        lambda actual_scene, owner, **kwargs: applied.append(
            (actual_scene, owner, kwargs)
        ),
    )
    depsgraph = object()
    scene = SimpleNamespace(
        frame_start=3,
        fluidsim=SimpleNamespace(
            flip_ratio=0.2,
            spacetime_seed=77,
            snippets=[],
            status="",
            gravity=(0.0, 0.0, -4.0),
            use_scene_gravity=False,
        ),
        use_gravity=False,
        gravity=(0.0, 0.0, -9.81),
    )
    sess.scene_identity = id(scene)

    sess.reset(scene, depsgraph=depsgraph)

    assert preflight_calls == [
        {
            "depsgraph": depsgraph,
            "runtime_capacity": 4096,
            "operation": "RESET",
        }
    ]
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
    sess.scene_identity = id(scene)
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
        sess.scene_identity = id(scene_arg)
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

    properties._on_live_changed(
        SimpleNamespace(live=True, status=""), SimpleNamespace(scene=scene)
    )

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
    sess.scene_identity = id(scene)
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


def test_runtime_warp_probe_is_cached_until_explicit_refresh(monkeypatch):
    _install_bpy_stub(monkeypatch)
    runtime = _reload_module("fluidSim_warp_blender51.runtime.warp_runtime")
    calls = []
    fake_warp = SimpleNamespace(
        init=lambda: calls.append("init"),
        get_devices=lambda: [
            SimpleNamespace(
                alias="cuda:3",
                name="Test GPU",
                total_memory=8 * 1024**3,
                is_cuda=True,
            )
        ],
        __version__="1.16.0",
        __file__="warp/__init__.py",
    )
    monkeypatch.setattr(runtime, "_import_warp", lambda: fake_warp)
    runtime._cached_probe.cache_clear()

    first = runtime.probe_runtime()
    second = runtime.probe_runtime()
    assert first is second
    assert first.available
    assert first.cuda_devices[0].alias == "cuda:3"
    assert calls == ["init"]

    runtime.probe_runtime(force=True)
    assert calls == ["init", "init"]


def test_runtime_warp_probe_accepts_certified_external_runtime(monkeypatch):
    _install_bpy_stub(monkeypatch)
    runtime = _reload_module("fluidSim_warp_blender51.runtime.warp_runtime")
    fake_warp = SimpleNamespace(
        init=lambda: None,
        get_devices=lambda: [
            SimpleNamespace(
                alias="cuda:0",
                name="Test GPU",
                total_memory=8 * 1024**3,
                is_cuda=True,
            )
        ],
        __version__="1.16.0",
        __file__="warp/__init__.py",
    )
    monkeypatch.setattr(runtime, "_import_warp", lambda: fake_warp)
    runtime._cached_probe.cache_clear()

    probe = runtime.probe_runtime()

    assert probe.available
    assert probe.warp_version == "1.16.0"
    assert "Detected" in probe.message


@pytest.mark.parametrize("version", ["1.12.1", "1.16.1", "1.17.0", "unknown"])
def test_runtime_warp_probe_rejects_noncertified_versions(
    monkeypatch,
    version,
):
    _install_bpy_stub(monkeypatch)
    runtime = _reload_module("fluidSim_warp_blender51.runtime.warp_runtime")
    fake_warp = SimpleNamespace(
        init=lambda: None,
        get_devices=lambda: [
            SimpleNamespace(
                alias="cuda:0",
                name="Test GPU",
                total_memory=8 * 1024**3,
                is_cuda=True,
            )
        ],
        __version__=version,
        __file__="warp/__init__.py",
    )
    monkeypatch.setattr(runtime, "_import_warp", lambda: fake_warp)
    runtime._cached_probe.cache_clear()

    probe = runtime.probe_runtime()

    assert not probe.available
    assert "requires external Warp 1.16.0" in probe.message


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
        "fluidSim_warp_blender51.blender.pending_settings"
    )
    playback = _reload_module("fluidSim_warp_blender51.blender.playback")
    render = _reload_module("fluidSim_warp_blender51.blender.render")
    viewport_overlay = _reload_module(
        "fluidSim_warp_blender51.blender.viewport_overlay"
    )
    cuda_reduction = _reload_module(
        "fluidSim_warp_blender51.sim.cuda_reduction"
    )
    operators = _reload_module("fluidSim_warp_blender51.operators")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    preferences = _reload_module("fluidSim_warp_blender51.preferences")
    properties = _reload_module("fluidSim_warp_blender51.properties")

    calls = []
    monkeypatch.setattr(
        viewport_overlay, "unregister_overlay", lambda: calls.append("viewport")
    )
    monkeypatch.setattr(render, "unregister_handlers", lambda: calls.append("render"))
    monkeypatch.setattr(
        playback, "unregister_handlers", lambda: calls.append("playback")
    )
    monkeypatch.setattr(
        cuda_reduction,
        "reset_cuda_module_cache",
        lambda: calls.append("cuda"),
    )
    monkeypatch.setattr(
        pending_settings, "unregister_handlers", lambda: calls.append("pending")
    )
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
        "cuda",
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
            (scene, sess, kwargs)
        ),
    )
    import fluidSim_warp_blender51.blender.output_publication as output_publication

    monkeypatch.setattr(
        output_publication, "prepare_outputs", fake_viewport.prepare_outputs
    )
    monkeypatch.setattr(
        output_publication, "update_outputs", fake_viewport.update_outputs
    )
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
    sess.scene_identity = id(scene)

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
        output_publication, "prepare_outputs", fake_viewport.prepare_outputs
    )
    monkeypatch.setattr(
        output_publication, "update_outputs", fake_viewport.update_outputs
    )
    solver = SimpleNamespace(settings=SimpleNamespace(whitewater_enabled=True))
    sess = scene_bridge.session()
    sess.solver = solver
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(whitewater_enabled=False, status="")
    )
    sess.scene_identity = id(scene)

    _pending_module().request_scene_update(
        scene, _pending_module().SceneUpdate.WHITEWATER
    )

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


def test_modal_bake_registers_and_clears_deep_production_ownership(
    monkeypatch, tmp_path
):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import production_bake

    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            cache_path=str(tmp_path / "cache"),
            checkpoint_interval=1,
            cache_compress=False,
            bake_start=1,
            bake_end=2,
            bake_status="IDLE",
            bake_progress=0.0,
            bake_message="",
            status="",
        )
    )
    window_manager = SimpleNamespace(
        event_timer_add=lambda *_args, **_kwargs: object(),
        event_timer_remove=lambda _timer: None,
        modal_handler_add=lambda _operator: None,
    )
    context = SimpleNamespace(
        scene=scene, window_manager=window_manager, window=object()
    )
    production_bake.clear_active_bake(scene)
    operator = operators.FLUIDSIM_OT_bake_start()
    operator.restart_existing = False

    assert operator.execute(context) == {"RUNNING_MODAL"}
    assert production_bake.active_bake(scene) is operator._controller
    assert production_bake.cache_path_is_active(tmp_path / "cache")

    operator._controller.status = production_bake.BakeStatus.COMPLETE
    operator._controller.message = "Production bake complete"
    assert operator._finish(context) == {"FINISHED"}
    assert production_bake.active_bake(scene) is None


def test_solver_mutation_operators_are_disabled_during_production_bake(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import production_bake

    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=""))
    context = SimpleNamespace(scene=scene)
    operators._bridge().session().solver = object()
    controller = SimpleNamespace(status=production_bake.BakeStatus.RUNNING)
    production_bake.register_active_bake(scene, controller)
    monkeypatch.setattr(operators._operator_runtime, "_runtime_poll", lambda _cls: True)
    try:
        for operator_type in (
            operators.FLUIDSIM_OT_initialize,
            operators.FLUIDSIM_OT_release,
            operators.FLUIDSIM_OT_reset,
            operators.FLUIDSIM_OT_step,
            operators.FLUIDSIM_OT_toggle_live,
        ):
            assert operator_type.poll(context) is False, operator_type.__name__
    finally:
        production_bake.clear_active_bake(scene, controller)


def test_cache_playback_operators_arm_and_stop_timeline_mode(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            live=False,
            cache_playback_enabled=False,
            status="Idle",
        )
    )
    context = SimpleNamespace(scene=scene)

    assert operators.FLUIDSIM_OT_cache_playback_start().execute(context) == {"FINISHED"}
    assert scene.fluidsim.cache_playback_enabled is True
    assert scene.fluidsim.live is True
    assert scene.fluidsim.status == "Cache Playback ready - press Timeline Play"

    assert operators.FLUIDSIM_OT_cache_playback_stop().execute(context) == {"FINISHED"}
    assert scene.fluidsim.cache_playback_enabled is False
    assert scene.fluidsim.live is False
    assert scene.fluidsim.status == "Cache Playback stopped"


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
        name=f"{role.title()} Object",
        type="MESH",
        fluidsim=SimpleNamespace(**values),
        display_type="TEXTURED",
        hide_render=False,
    )


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
        SimpleNamespace(selected_objects=[])
    )
    assert (
        "Select at least one object" in operators.FLUIDSIM_OT_tag_object._poll_message
    )


def test_phase3_aggregate_emission_is_not_broadcast_as_per_source_truth():
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    def obj(pointer):
        pg = SimpleNamespace(
            role="SOURCE",
            source_mode="PARTICLES",
            enabled=True,
            source_status_code="",
            source_status_message="",
            source_status_severity="",
            source_status_emitted=0,
            source_status_capacity_limited=False,
        )
        return SimpleNamespace(fluidsim=pg, as_pointer=lambda: pointer)

    first, second = obj(1), obj(2)
    inputs = SceneInputs()
    inputs.update_source_status(
        SimpleNamespace(objects=[first, second]),
        (),
        {"emitted": 12, "capacity_limited": True},
    )
    assert first.fluidsim.source_status_emitted == 0
    assert second.fluidsim.source_status_emitted == 0
    second.fluidsim.enabled = False
    inputs.update_source_status(
        SimpleNamespace(objects=[first, second]),
        (),
        {"emitted": 12, "capacity_limited": True},
    )
    assert first.fluidsim.source_status_emitted == 12
    assert first.fluidsim.source_status_capacity_limited is True


def test_phase1_point_injection_poll_is_truthful(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    sess = SimpleNamespace(active=True)
    monkeypatch.setattr(
        operators._operator_runtime,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: sess),
    )

    mesh = SimpleNamespace(type="MESH", particle_systems=[])
    assert operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=mesh)
    )
    unsupported = SimpleNamespace(type="LIGHT", particle_systems=[])
    assert not operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=unsupported)
    )
    assert "mesh vertices" in operators.FLUIDSIM_OT_inject_particles._poll_message
    sess.active = False
    assert not operators.FLUIDSIM_OT_inject_particles.poll(
        SimpleNamespace(active_object=mesh)
    )
    assert (
        "Initialize the solver" in operators.FLUIDSIM_OT_inject_particles._poll_message
    )


def test_phase1_point_injection_capacity_refusal_is_recoverable_cancel(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    from fluidSim_warp_blender51.blender import output_publication, sources

    result = SimpleNamespace(
        admitted=False,
        emitted=0,
        required=5,
        available=4,
        capacity_limited=True,
        total=1,
    )
    solver = SimpleNamespace(sourcing=SimpleNamespace(last_injection_result=result))
    sess = SimpleNamespace(active=True, solver=solver)
    fatal_errors = []
    bridge = SimpleNamespace(
        session=lambda: sess, set_error=lambda *_args: fatal_errors.append(_args)
    )
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)
    monkeypatch.setattr(sources, "inject_object_particles", lambda *_args: 0)
    viewport_updates = []
    monkeypatch.setattr(
        output_publication,
        "update_outputs",
        lambda *_args: viewport_updates.append(_args),
    )

    reports = []
    op = operators.FLUIDSIM_OT_inject_particles()
    op.report = lambda level, message: reports.append((level, message))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status="Solver ready"))
    obj = SimpleNamespace(name="Seed", type="MESH", particle_systems=[])

    assert op.execute(SimpleNamespace(scene=scene, active_object=obj)) == {"CANCELLED"}
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

    sourcing_state = SimpleNamespace(
        last_injection_result=SimpleNamespace(admitted=False, required=5, available=4)
    )

    class Solver:
        sourcing = sourcing_state

        def inject_particle_packets_exact(self, packets):
            assert tuple(packets) == ()
            result = SimpleNamespace(
                admitted=True,
                emitted=0,
                required=0,
                available=4,
                capacity_limited=False,
                total=0,
            )
            self.sourcing.last_injection_result = result
            return result

    sess = SimpleNamespace(active=True, solver=Solver())
    monkeypatch.setattr(
        operators._operator_runtime,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: sess),
    )
    monkeypatch.setattr(sources, "prepare_object_particles", lambda *_args: None)
    viewport_updates = []
    monkeypatch.setattr(
        output_publication,
        "update_outputs",
        lambda *_args: viewport_updates.append(_args),
    )

    reports = []
    op = operators.FLUIDSIM_OT_inject_particles()
    op.report = lambda level, message: reports.append((level, message))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status="Solver ready"))
    obj = SimpleNamespace(name="Empty", type="MESH", particle_systems=[])

    assert op.execute(SimpleNamespace(scene=scene, active_object=obj)) == {"FINISHED"}
    assert len(viewport_updates) == 1
    assert reports[-1][0] == {"INFO"}
    assert "Injected 0 particles" in reports[-1][1]


def test_single_output_removal_requires_owned_stable_identity(monkeypatch):
    _install_bpy_stub(monkeypatch)
    output_publication = _reload_module(
        "fluidSim_warp_blender51.blender.output_publication"
    )
    monkeypatch.setattr(output_publication, "_require_bpy", lambda: None)

    class Output:
        def __init__(self, *, owned=True):
            self._values = {
                output_publication.OUTPUT_TAG: output_publication.SURFACE_NAME,
                "fluidsim_owned": owned,
            }

        def get(self, name, default=None):
            return self._values.get(name, default)

    owned = Output()
    removed = []
    monkeypatch.setattr(
        output_publication, "output_object", lambda _scene, _kind: owned
    )
    monkeypatch.setattr(output_publication, "_remove_output_object", removed.append)

    with pytest.raises(ValueError, match="unsupported FluidSim output kind"):
        output_publication.remove_output(object(), "user-object")
    with pytest.raises(RuntimeError, match="identity changed"):
        output_publication.remove_output(
            object(),
            output_publication.SURFACE_NAME,
            expected_object=Output(),
        )

    unowned = Output(owned=False)
    monkeypatch.setattr(
        output_publication, "output_object", lambda _scene, _kind: unowned
    )
    with pytest.raises(RuntimeError, match="not owned by FluidSim"):
        output_publication.remove_output(
            object(),
            output_publication.SURFACE_NAME,
            expected_object=unowned,
        )

    monkeypatch.setattr(
        output_publication, "output_object", lambda _scene, _kind: owned
    )
    assert (
        output_publication.remove_output(
            object(),
            output_publication.SURFACE_NAME,
            expected_object=owned,
        )
        is True
    )
    assert removed == [owned]


def test_contextual_domain_actions_revalidate_object_role_and_identity(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")

    domain = SimpleNamespace(
        name="Domain", type="MESH", fluidsim=SimpleNamespace(role="NONE")
    )
    settings = SimpleNamespace(domain_object=None, status="")
    context = SimpleNamespace(
        scene=SimpleNamespace(fluidsim=settings),
        active_object=domain,
        object=domain,
    )

    assert operators.FLUIDSIM_OT_use_active_as_domain.poll(context)
    assign = operators.FLUIDSIM_OT_use_active_as_domain()
    assert assign.execute(context) == {"FINISHED"}
    assert settings.domain_object is domain

    assert operators.FLUIDSIM_OT_clear_domain_assignment.poll(context)
    clear = operators.FLUIDSIM_OT_clear_domain_assignment()
    assert clear.execute(context) == {"FINISHED"}
    assert settings.domain_object is None

    domain.fluidsim.role = "SOURCE"
    assert not operators.FLUIDSIM_OT_use_active_as_domain.poll(context)
    assert "Clear the object's FluidSim input role" in (
        operators.FLUIDSIM_OT_use_active_as_domain._poll_message
    )


def test_viewport_full_workflow_is_collapsed_below_compact_status(monkeypatch):
    _install_bpy_stub(monkeypatch)
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
    module_path = root / "blender" / "ui_view3d.py"
    assert module_path.is_file(), "1.0.255 full View3D workflow adapter is missing"

    ui_view3d = _reload_module("fluidSim_warp_blender51.blender.ui_view3d")
    panels = _reload_module("fluidSim_warp_blender51.panels")
    assert panels.FLUIDSIM_PT_viewport_status in panels.CLASSES

    full_settings = ui_view3d.FLUIDSIM_VIEW3D_PT_full_settings
    assert full_settings.bl_parent_id == "FLUIDSIM_PT_viewport_status"
    assert full_settings.bl_options == {"DEFAULT_CLOSED"}

    classes = ui_view3d.VIEW3D_PANEL_CLASSES
    assert len({cls.bl_idname for cls in classes}) == len(classes)
    mirrors = {cls.bl_idname: cls for cls in classes}
    for source in ui_view3d.VIEW3D_SOURCE_PANELS:
        mirror = mirrors[ui_view3d._mirror_id(source.bl_idname)]
        source_parent = str(getattr(source, "bl_parent_id", "") or "")
        expected_parent = (
            ui_view3d._mirror_id(source_parent)
            if source_parent
            else full_settings.bl_idname
        )
        assert mirror.bl_parent_id == expected_parent


def test_physics_context_uses_existing_scene_and_object_owners(monkeypatch):
    _install_bpy_stub(monkeypatch)
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
    module_path = root / "blender" / "ui_physics.py"
    assert module_path.is_file(), "1.0.255 contextual Physics adapter is missing"

    ui_physics = _reload_module("fluidSim_warp_blender51.blender.ui_physics")
    output_publication = _reload_module(
        "fluidSim_warp_blender51.blender.output_publication"
    )
    scene_settings = SimpleNamespace(domain_object=None)
    scene = SimpleNamespace(fluidsim=scene_settings)

    input_obj = SimpleNamespace(
        fluidsim=SimpleNamespace(role="SOURCE"),
        get=lambda _key, default=None: default,
    )
    context = SimpleNamespace(scene=scene, object=input_obj)
    assert ui_physics.physics_object_kind(context) == "INPUT"

    scene_settings.domain_object = input_obj
    assert ui_physics.physics_object_kind(context) == "DOMAIN"

    output_obj = SimpleNamespace(
        fluidsim=SimpleNamespace(role="NONE"),
        get=lambda key, default=None: (
            output_publication.SURFACE_NAME
            if key == output_publication.OUTPUT_TAG
            else default
        ),
    )
    scene_settings.domain_object = None
    context.object = output_obj
    assert ui_physics.physics_object_kind(context) == "SURFACE"

    assert len({cls.bl_idname for cls in ui_physics.PHYSICS_PANEL_CLASSES}) == len(
        ui_physics.PHYSICS_PANEL_CLASSES
    )
    assert all(cls.bl_context == "physics" for cls in ui_physics.PHYSICS_PANEL_CLASSES)


def test_capacity_suggestion_poll_and_dialog_are_scene_owned(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="safe",
        label="Safe",
        changes={"max_particles": 12},
        estimated_bytes=1,
        budget_bytes=2,
    )
    plan = SimpleNamespace(
        scene_signature="identical-content", suggestions=(suggestion,), blockers=()
    )
    first = SimpleNamespace(fluidsim=SimpleNamespace(max_particles=8))
    second = SimpleNamespace(fluidsim=SimpleNamespace(max_particles=8))
    bridge.session()._set_capacity_plan(first, plan)
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)

    assert (
        operators.FLUIDSIM_OT_apply_capacity_suggestion.poll(
            SimpleNamespace(scene=first)
        )
        is True
    )
    assert (
        operators.FLUIDSIM_OT_apply_capacity_suggestion.poll(
            SimpleNamespace(scene=second)
        )
        is False
    )

    dialogs = []
    context_first = SimpleNamespace(
        scene=first,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda op, width: (
                dialogs.append((op, width)) or {"RUNNING_MODAL"}
            )
        ),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "safe"
    assert op.invoke(context_first, None) == {"RUNNING_MODAL"}

    refreshed = []
    monkeypatch.setattr(
        bridge, "refresh_capacity_plan", lambda scene: refreshed.append(scene) or plan
    )
    assert op.execute(SimpleNamespace(scene=second)) == {"CANCELLED"}
    assert refreshed == []
    assert second.fluidsim.max_particles == 8


def test_capacity_dialog_rechecks_grown_pool_reinitialize_with_authored_model(
    monkeypatch,
):
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
        simulation_admission,
        "build_settings",
        lambda _scene: SimpleNamespace(
            device="cuda:0", max_particles=int(fs.max_particles)
        ),
    )
    from fluidSim_warp_blender51.sim import memory_safety

    monkeypatch.setattr(
        memory_safety,
        "vram_snapshot",
        lambda _device: {
            "gpu_total_bytes": 16 * 1024**3,
            "gpu_free_bytes": 12 * 1024**3,
        },
    )
    planned = []

    def plan_scene(_scene, settings, _resources, *, additional_transaction_bytes=0):
        capacity = int(settings.max_particles)
        planned.append((capacity, additional_transaction_bytes))
        suggestion = SimpleNamespace(
            key="preserve_simulation",
            label="Preserve Simulation",
            changes={"max_particles": 2048},
            estimated_bytes=1,
            budget_bytes=2,
        )
        blocked = capacity < 2048
        return SimpleNamespace(
            capacity_plan=SimpleNamespace(
                scene_signature=f"capacity-{capacity}",
                suggestions=(suggestion,) if blocked else (),
                blockers=(
                    (SimpleNamespace(code="PARTICLE_CAPACITY"),) if blocked else ()
                ),
            )
        )

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    original = plan_scene(scene, SimpleNamespace(max_particles=1024), {}).capacity_plan
    sess._set_capacity_plan(scene, original, operation="REINITIALIZE")
    expected_signature = sess.capacity_plan_signature_for_scene(scene)
    initialized = []
    monkeypatch.setattr(
        bridge, "initialize_session", lambda actual: initialized.append(actual) or sess
    )
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda _op, width: {"RUNNING_MODAL"}
        ),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "preserve_simulation"

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert op.expected_scene_signature == expected_signature
    assert op.execute(context) == {"FINISHED"}
    assert planned[-1][0] == 1024
    assert planned[-1][1] == memory_safety.estimate_solver_bytes(sess.solver.settings)
    assert fs.max_particles == 2048
    assert initialized == [scene]


def test_capacity_dialog_rechecks_same_scene_signature_before_mutation(monkeypatch):
    _install_bpy_stub(monkeypatch)
    bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    operators = _reload_module("fluidSim_warp_blender51.operators")
    suggestion = SimpleNamespace(
        key="safe",
        label="Safe",
        changes={"max_particles": 12},
        estimated_bytes=1,
        budget_bytes=2,
    )
    original = SimpleNamespace(
        scene_signature="scene-v1", suggestions=(suggestion,), blockers=()
    )
    changed = SimpleNamespace(
        scene_signature="scene-v2", suggestions=(suggestion,), blockers=()
    )
    fs = SimpleNamespace(max_particles=8, status="")
    scene = SimpleNamespace(fluidsim=fs)
    bridge.session()._set_capacity_plan(scene, original)
    monkeypatch.setattr(operators._operator_runtime, "_bridge", lambda: bridge)
    context = SimpleNamespace(
        scene=scene,
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda _op, width: {"RUNNING_MODAL"}
        ),
    )
    op = operators.FLUIDSIM_OT_apply_capacity_suggestion()
    op.suggestion_key = "safe"
    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    monkeypatch.setattr(
        bridge,
        "refresh_capacity_plan",
        lambda actual: bridge.session()._set_capacity_plan(actual, changed),
    )

    assert op.execute(context) == {"CANCELLED"}
    assert fs.max_particles == 8
    assert "changed" in fs.status.lower()


def test_status_message_error_is_heavier_than_warning(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_common = _reload_module("fluidSim_warp_blender51.blender.ui_common")

    class _Rec:
        def __init__(self, alert=False):
            self.alert = alert
            self.rows = []
            self.labels = []
        def row(self, align=False):
            child = _Rec(self.alert)
            self.rows.append(child)
            return child
        def label(self, text="", icon="NONE"):
            self.labels.append({"text": text, "icon": icon, "alert": self.alert})

    def _all_labels(rec):
        found = list(rec.labels)
        for child in rec.rows:
            found.extend(_all_labels(child))
        return found

    warn = _Rec()
    ui_common.draw_status_message(warn, "low memory", "WARNING")
    err = _Rec()
    ui_common.draw_status_message(err, "disk full", "ERROR")

    warn_label = _all_labels(warn)[0]
    err_label = _all_labels(err)[0]

    assert warn_label["text"].startswith("Warning:")
    assert warn_label["alert"] is False
    assert err_label["text"].startswith("Error:")
    assert err_label["alert"] is True
    assert err_label["icon"] != warn_label["icon"]


def test_display_panel_exposes_hud_overlay_toggle(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    fs = SimpleNamespace(
        liquid_view_mode="PARTICLES",
        surface_enabled=False,
        whitewater_enabled=False,
        show_overlay=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
    )
    panel = ui_scene.FLUIDSIM_PT_output_display()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))
    assert "show_overlay" in panel.layout.props


def test_setup_scene_invoke_opens_mode_dialog(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    calls = []
    context = SimpleNamespace(
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda op: calls.append(op) or {"RUNNING_MODAL"}
        )
    )
    op = operators.FLUIDSIM_OT_setup_scene()
    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert calls == [op]
    op.layout = _Layout()
    op.draw(context)
    assert "mode" in op.layout.props


def test_runtime_recovery_button_is_labeled_and_status_not_truncated(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", False, "no CUDA device"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": False, "label": "Idle", "icon": "PAUSE"})
    long_status = "Solver halted: " + "x" * 200
    fs = SimpleNamespace(status=long_status, height_error="")
    ctx = SimpleNamespace(scene=SimpleNamespace(fluidsim=fs))
    panel = ui_scene.FLUIDSIM_PT_scene_control()
    panel.layout = _Layout()
    panel.draw(ctx)
    probe_ops = [op for op in panel.layout.operators
                 if op["id"] == "fluidsim.refresh_runtime_probe"]
    assert probe_ops and probe_ops[0].get("text")
    joined = "".join(l["text"] for l in panel.layout.labels)
    assert long_status[-20:] in joined


def test_transport_keeps_step_buttons_present_but_greyed_when_not_ready(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", False, "not ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": True,
                                       "label": "Idle", "icon": "PAUSE"})
    panel = ui_scene.FLUIDSIM_PT_scene_transport()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(status=""))))

    op_ids = [op["id"] for op in panel.layout.operators]
    assert "fluidsim.toggle_live" in op_ids
    # Step One + Step Many always present (anti-reflow), never hidden.
    assert op_ids.count("fluidsim.step") == 2
    # Both step buttons drawn under a disabled sub-layout (greyed, not hidden).
    step_ops = [op for op in panel.layout.operators if op["id"] == "fluidsim.step"]
    assert step_ops and all(op["enabled"] is False for op in step_ops)


def test_transport_enables_step_buttons_when_runtime_ready(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", True, "ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": True,
                                       "label": "Idle", "icon": "PAUSE"})
    panel = ui_scene.FLUIDSIM_PT_scene_transport()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(status=""))))

    step_ops = [op for op in panel.layout.operators if op["id"] == "fluidsim.step"]
    assert len(step_ops) == 2 and all(op["enabled"] is True for op in step_ops)


def test_lifecycle_marks_release_and_reset_as_destructive(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", True, "ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": True,
                                       "label": "Idle", "icon": "PAUSE"})
    panel = ui_scene.FLUIDSIM_PT_scene_lifecycle()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace())))

    ops = {op["id"]: op for op in panel.layout.operators}
    assert "fluidsim.release" in ops
    assert "fluidsim.reset" in ops
    # Both destructive ops are drawn under a red-alert sub-layout.
    assert ops["fluidsim.release"]["alert"] is True
    assert ops["fluidsim.reset"]["alert"] is True


def test_lifecycle_offers_initialize_when_not_owning_scene(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", True, "ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": False,
                                       "label": "Idle", "icon": "PAUSE"})
    panel = ui_scene.FLUIDSIM_PT_scene_lifecycle()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace())))

    ops = {op["id"]: op for op in panel.layout.operators}
    assert "fluidsim.initialize" in ops
    # Constructive Initialize is not marked destructive.
    assert ops["fluidsim.initialize"]["alert"] is False
    # Nothing destructive is offered before the solver owns the scene.
    assert "fluidsim.release" not in ops and "fluidsim.reset" not in ops


def test_status_hub_is_read_only_with_live_strip(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", True, "ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "LIVE_PREVIEW", "owns_scene": True,
                                       "label": "Live Preview", "icon": "PLAY"})
    fs = SimpleNamespace(status="", height_error="",
                         stat_step_ms=12.5, stat_substeps=3, stat_cfl_clamped=True)
    panel = ui_scene.FLUIDSIM_PT_scene_control()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    labels = "".join(l["text"] for l in panel.layout.labels)
    # State name is shown.
    assert "Live Preview" in labels
    # Labeled runtime recovery (from Task 0.4) is retained.
    probe = [op for op in panel.layout.operators
             if op["id"] == "fluidsim.refresh_runtime_probe"]
    assert probe and probe[0].get("text")
    # Compact one-line step_ms · substeps · CFL strip.
    assert "·" in labels and "3" in labels and "12.5" in labels
    # No transport or lifecycle controls leak back into the read-only hub.
    op_ids = {op["id"] for op in panel.layout.operators}
    assert op_ids.isdisjoint({
        "fluidsim.toggle_live", "fluidsim.step", "fluidsim.reset",
        "fluidsim.release", "fluidsim.initialize",
    })


def test_settings_group_has_ui_mode_and_quality(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    assert {"BASIC", "ADVANCED"} <= {i[0] for i in properties.UI_MODE_ITEMS}
    assert {"DRAFT", "PREVIEW", "FINAL", "CUSTOM"} <= {i[0] for i in properties.QUALITY_ITEMS}
    ann = properties.FLUIDSIM_PG_settings.__annotations__
    assert "ui_mode" in ann and "quality" in ann
