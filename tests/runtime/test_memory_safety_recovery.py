"""Regression contracts for pre-allocation GPU memory recovery."""

import ast
import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

from fluidSim_warp_blender51.sim.memory_safety import (
    SIM_BYTES_PER_VOXEL,
    WHITEWATER_BYTES,
    WHITEWATER_EMISSION_BYTES_PER_LIQUID,
    estimate_solver_bytes,
    is_memory_capacity_error,
    safe_memory_budget,
)


def test_memory_estimate_includes_compact_pcg_and_whitewater_free_list():
    assert SIM_BYTES_PER_VOXEL == 356
    assert WHITEWATER_BYTES == 76
    assert WHITEWATER_EMISSION_BYTES_PER_LIQUID == 8


def _risky_settings():
    return SimpleNamespace(
        leaf_capacity=262_144,
        max_particles=8_000_000,
        max_whitewater=4_000_000,
        voxel_size=0.02,
        particles_per_voxel=16.0,
        topology_expand=3,
        domain_min=(-6.0, -2.0, 0.0),
        domain_max=(6.0, 2.0, 6.0),
        surface_resolution_mode=1,
        surface_resolution_multiplier=4.0,
        surface_leaf_capacity_scale=64.0,
        surface_max_dense_dim=2048,
    )


def test_16_gib_card_can_use_11_6_gib_when_14_gib_is_free():
    gib = 1024**3

    assert safe_memory_budget(16 * gib, 14 * gib) >= int(11.6 * gib)


def test_memory_estimate_flags_capacity_that_exceeds_budget_before_allocation():
    settings = _risky_settings()
    budget = 8 * 1024**3

    assert estimate_solver_bytes(settings) > budget


def test_memory_errors_are_routed_to_recovery_ui():
    assert is_memory_capacity_error("FluidSim: leaf capacity exceeded (9 > 8)")
    assert is_memory_capacity_error("CUDA error 2: out of memory")
    assert is_memory_capacity_error("FluidSim memory safeguard stopped initialization")
    assert not is_memory_capacity_error("pressure solver stalled")


# --- The capacity-recovery UI, driven through its real code (REVIEW-024) -----------
#
# This section replaces two tests that searched the add-on's source text for
# strings. A string survives in a comment or a dead branch after the behaviour is
# gone, and a reference to a file survives in an unrelated list. Each check below
# registers, polls, draws or calls the real code against a recording stand-in and
# asserts on what happened, so removing the guarded behaviour turns it red.

_PACKAGE = "fluidSim_warp_blender51"
_GIB = 1024**3
_FRESH_MODULES = (
    f"{_PACKAGE}.operators",
    f"{_PACKAGE}.panels",
    f"{_PACKAGE}.blender.ui_scene",
)
_SUGGEST = "fluidsim.apply_capacity_suggestion"
_SAFE_SETTINGS = "fluidsim.apply_safe_settings"


class _Layout:
    """Recording ``bpy.types.UILayout``: keeps every operator, label and prop drawn."""

    def __init__(self, root=None):
        self.root = root or self
        if root is None:
            self.operators = []
            self.labels = []
            self.props = []

    def _child(self):
        return _Layout(self.root)

    column = row = box = grid_flow = lambda self, **_kwargs: self._child()

    def label(self, text="", icon="NONE"):
        self.root.labels.append(text)

    def prop(self, _data, name, **_kwargs):
        self.root.props.append(name)

    def operator(self, idname, **kwargs):
        # Blender returns the operator's properties here; callers set them.
        op = SimpleNamespace(bl_idname=idname, **kwargs)
        self.root.operators.append(op)
        return op

    def separator(self):
        return None

    @property
    def operator_ids(self):
        return [op.bl_idname for op in self.root.operators]


def _bpy_stub(registered):
    bpy = types.ModuleType("bpy")

    class _Operator:
        def __init__(self):
            self.reports = []

        @classmethod
        def poll_message_set(cls, _message):
            return None

        def report(self, level, message):
            self.reports.append((tuple(level), message))

    class _Namespace:
        pass

    def _prop(**kwargs):
        return kwargs.get("default")

    bpy.types = SimpleNamespace(
        Operator=_Operator,
        Panel=_Namespace,
        UIList=_Namespace,
        AddonPreferences=_Namespace,
        PropertyGroup=object,
        Text=object,
        Object=_Namespace,
        Scene=_Namespace,
        Material=object,
        Image=object,
    )
    bpy.props = SimpleNamespace(
        BoolProperty=_prop,
        BoolVectorProperty=_prop,
        CollectionProperty=_prop,
        EnumProperty=_prop,
        FloatProperty=_prop,
        FloatVectorProperty=_prop,
        IntProperty=_prop,
        PointerProperty=_prop,
        StringProperty=_prop,
    )
    bpy.path = SimpleNamespace(abspath=lambda value: value)
    bpy.context = SimpleNamespace(screen=SimpleNamespace(areas=[]))
    bpy.utils = SimpleNamespace(
        register_class=registered.append,
        unregister_class=lambda _cls: None,
    )
    return bpy


@pytest.fixture
def addon(monkeypatch):
    """The add-on's operators, panels and scene UI imported under a recording ``bpy``.

    ``classifier_calls`` records every call to ``is_memory_capacity_error``. The
    function is replaced before the UI is imported, so a ``from ... import`` of it
    binds the recorder too. ``sys.modules`` and the package attributes are put back
    on exit, so a later test never sees a stub-bound module.
    """
    before = dict(sys.modules)
    registered, classifier_calls = [], []
    bpy = _bpy_stub(registered)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", bpy.props)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=lambda v: v))
    for name in tuple(sys.modules):
        if name in _FRESH_MODULES or name.startswith(f"{_PACKAGE}.operator_domains"):
            del sys.modules[name]

    from fluidSim_warp_blender51.sim import memory_safety

    def classifier(message):
        classifier_calls.append(message)
        return False

    monkeypatch.setattr(memory_safety, "is_memory_capacity_error", classifier)
    load = importlib.import_module
    try:
        yield SimpleNamespace(
            registered=registered,
            classifier=classifier,
            classifier_calls=classifier_calls,
            operators=load(f"{_PACKAGE}.operators"),
            panels=load(f"{_PACKAGE}.panels"),
            ui_scene=load(f"{_PACKAGE}.blender.ui_scene"),
            scene_bridge=load(f"{_PACKAGE}.blender.scene_bridge"),
            session=load(f"{_PACKAGE}.operator_domains.session"),
        )
    finally:
        for name in tuple(set(sys.modules) | set(before)):
            if sys.modules.get(name) is before.get(name):
                continue
            parent, _, child = name.rpartition(".")
            parent_module = sys.modules.get(parent)
            if name in before:
                sys.modules[name] = before[name]
                if parent_module is not None:
                    setattr(parent_module, child, before[name])
            elif name not in ("bpy", "bpy.props", "mathutils"):  # monkeypatch undoes these
                del sys.modules[name]


def _scene(*, alternatives=False, status=""):
    return SimpleNamespace(
        fluidsim=SimpleNamespace(
            # The capacity panel is Advanced-tier; the real default is BASIC.
            ui_mode="ADVANCED",
            capacity_show_alternatives=alternatives,
            status=status,
            leaf_capacity=262_144,
            max_particles=8_000_000,
            stat_step_ms=0.0,
            stat_substeps=0,
            stat_cfl_clamped=False,
        )
    )


def _plan(*keys):
    """A capacity plan with one blocker and one verified option per key."""
    return SimpleNamespace(
        estimated_bytes=3 * _GIB,
        budget_bytes=2 * _GIB,
        blockers=(SimpleNamespace(message="Leaf capacity does not fit the live budget"),),
        suggestions=tuple(
            SimpleNamespace(key=key, label=f"Option {key}", tradeoffs=(f"{key} costs detail",))
            for key in keys
        ),
    )


def _draw_capacity_panel(addon, monkeypatch, scene, plan):
    """Draw the Scene Properties capacity panel; return its layout and the scenes asked about."""
    asked = []
    monkeypatch.setattr(
        addon.scene_bridge,
        "capacity_plan_for_scene",
        lambda asked_scene: asked.append(asked_scene) or plan,
    )
    monkeypatch.setattr(
        addon.ui_scene,
        "active_domain_settings",
        lambda _scene: SimpleNamespace(auto_grow_particles=False, height_error=""),
    )
    context = SimpleNamespace(scene=scene)
    panel = addon.ui_scene.FLUIDSIM_PT_scene_capacity()
    assert panel.poll(context), "the capacity panel must be visible for a scene with a domain"
    panel.layout = _Layout()
    panel.draw(context)
    return panel.layout, asked


def test_session_domain_registers_the_capacity_recovery_operators(addon):
    addon.operators.register()

    registered = {cls.bl_idname: cls for cls in addon.registered if hasattr(cls, "bl_idname")}
    assert registered[_SAFE_SETTINGS].__module__ == f"{_PACKAGE}.operator_domains.session"
    assert registered[_SUGGEST].__module__ == f"{_PACKAGE}.operator_domains.session"


def test_safe_settings_operator_is_pollable_only_with_fluidsim_scene_properties(addon):
    operator = addon.session.FLUIDSIM_OT_apply_safe_settings

    assert operator.poll(SimpleNamespace(scene=_scene()))
    assert not operator.poll(SimpleNamespace(scene=SimpleNamespace()))
    assert not operator.poll(SimpleNamespace(scene=None))


def _run_safe_settings(addon, monkeypatch, refresh):
    scene = _scene()
    before = vars(scene.fluidsim).copy()
    asked = []
    bridge = SimpleNamespace(
        refresh_capacity_plan=lambda asked_scene: asked.append(asked_scene) or refresh())
    monkeypatch.setattr(addon.session._runtime, "_bridge", lambda: bridge)
    operator = addon.session.FLUIDSIM_OT_apply_safe_settings()

    result = operator.execute(SimpleNamespace(scene=scene))

    assert asked == [scene], "the operator must inspect the scene it was invoked on"
    return result, operator, scene, before


def test_safe_settings_routes_to_the_reviewed_options_and_changes_no_capacity(
        addon, monkeypatch):
    result, operator, scene, before = _run_safe_settings(
        addon, monkeypatch, lambda: _plan("trim-particles"))

    assert result == {"CANCELLED"}  # never applied from this entry point
    assert scene.fluidsim.status == "Capacity options ready — review and confirm one option"
    assert operator.reports == [(("WARNING",), scene.fluidsim.status)]
    changed = {k for k, v in vars(scene.fluidsim).items() if before[k] != v}
    assert changed == {"status"}


def test_safe_settings_reports_when_no_verified_option_exists_or_inspection_fails(
        addon, monkeypatch):
    result, operator, scene, _ = _run_safe_settings(addon, monkeypatch, lambda: _plan())
    assert result == {"CANCELLED"}
    assert scene.fluidsim.status == "No verified capacity-only option is available"
    assert operator.reports == [(("ERROR",), scene.fluidsim.status)]

    def broken():
        raise RuntimeError("probe failed")

    result, operator, scene, _ = _run_safe_settings(addon, monkeypatch, broken)
    assert result == {"CANCELLED"}
    assert scene.fluidsim.status == "Capacity inspection failed: probe failed"
    assert operator.reports == [(("ERROR",), scene.fluidsim.status)]


def test_capacity_panel_offers_each_verified_option_from_the_scenes_live_plan(
        addon, monkeypatch):
    scene = _scene(alternatives=True)

    layout, asked = _draw_capacity_panel(
        addon, monkeypatch, scene, _plan("trim-particles", "smaller-domain"))

    assert asked == [scene], "the panel must read the plan for the scene it draws"
    offered = [op for op in layout.root.operators if op.bl_idname == _SUGGEST]
    assert [op.suggestion_key for op in offered] == ["trim-particles", "smaller-domain"]
    assert "Leaf capacity does not fit the live budget" in layout.root.labels
    # With a plan, the compatibility operator is only the refresh entry.
    safe = [op for op in layout.root.operators if op.bl_idname == _SAFE_SETTINGS]
    assert [op.text for op in safe] == ["Recalculate"]


def test_capacity_panel_shows_one_option_until_alternatives_are_enabled_on_the_scene(
        addon, monkeypatch):
    plan = _plan("trim-particles", "smaller-domain")

    folded, _ = _draw_capacity_panel(addon, monkeypatch, _scene(alternatives=False), plan)
    shown, _ = _draw_capacity_panel(addon, monkeypatch, _scene(alternatives=True), plan)

    assert folded.operator_ids.count(_SUGGEST) == 1
    assert shown.operator_ids.count(_SUGGEST) == 2
    assert "capacity_show_alternatives" in folded.root.props  # the toggle lives on the scene


def test_capacity_panel_without_a_plan_only_offers_to_calculate_one(addon, monkeypatch):
    layout, asked = _draw_capacity_panel(addon, monkeypatch, _scene(), None)

    assert len(asked) == 1
    assert layout.operator_ids == [_SAFE_SETTINGS]
    assert layout.root.operators[0].text == "Calculate Safe Options"


def test_capacity_panel_offers_no_option_when_the_plan_has_none(addon, monkeypatch):
    layout, _ = _draw_capacity_panel(addon, monkeypatch, _scene(alternatives=True), _plan())

    assert _SUGGEST not in layout.operator_ids
    assert "Leaf capacity does not fit the live budget" in layout.root.labels


def _draw_launcher(addon, monkeypatch, *, state_id="IDLE", owns_scene=False,
                   pending_action="NONE", runtime_ready=True, status=""):
    """Draw the pinned sidebar strip against doubles for the state it reads."""
    panels = addon.panels
    monkeypatch.setattr(panels, "activity_ui_state", lambda _scene: {
        "id": state_id, "label": state_id.title(), "icon": "INFO", "owns_scene": owns_scene})
    monkeypatch.setattr(panels, "pending_ui_state", lambda _scene: {
        "action": pending_action, "summary": "Edits pending"})
    monkeypatch.setattr(panels, "probe_runtime", lambda: None)
    monkeypatch.setattr(panels, "configured_device", lambda: "cuda:0")
    monkeypatch.setattr(panels, "runtime_status_for_device", lambda _device, probe=None: (
        runtime_ready, status or ("CUDA ready" if runtime_ready else "no CUDA")))
    monkeypatch.setattr(
        addon.scene_bridge, "session", lambda: SimpleNamespace(active=owns_scene))
    context = SimpleNamespace(
        scene=_scene(status=status), screen=SimpleNamespace(is_animation_playing=False))
    panel = panels.FLUIDSIM_PT_viewport_status()
    panel.layout = _Layout()
    panel.draw(context)
    return panel.layout


_LAUNCHER_STATES = (
    {},
    {"owns_scene": True},
    {"owns_scene": True, "pending_action": "REINIT"},
    {"state_id": "PRODUCTION_BAKE"},
    {"state_id": "CACHE_PLAYBACK"},
    {"runtime_ready": False, "status": "CUDA error 2: out of memory"},
)


@pytest.mark.parametrize("state", _LAUNCHER_STATES, ids=lambda s: ",".join(s) or "idle")
def test_viewport_launcher_never_offers_the_capacity_compatibility_operator(
        addon, monkeypatch, state):
    layout = _draw_launcher(addon, monkeypatch, **state)

    assert "fluidsim.refresh_runtime_probe" in layout.operator_ids  # the strip was drawn
    assert _SAFE_SETTINGS not in layout.operator_ids


def test_launcher_check_would_see_a_compatibility_button_planted_in_the_strip(
        addon, monkeypatch):
    """The check above is only worth having if the recording layout can fail it."""
    monkeypatch.setattr(
        addon.panels, "draw_transport",
        lambda layout, _context, _state, _ready: layout.operator(_SAFE_SETTINGS, text="Fix"))

    layout = _draw_launcher(addon, monkeypatch)

    assert _SAFE_SETTINGS in layout.operator_ids


def test_recovery_ui_is_driven_by_the_verified_plan_not_by_classifying_error_text(
        addon, monkeypatch):
    memory_error = "CUDA error 2: out of memory"
    for name in ("operators", "panels", "ui_scene"):
        assert getattr(addon, name).__dict__.get("is_memory_capacity_error") is not addon.classifier

    _draw_capacity_panel(addon, monkeypatch, _scene(status=memory_error), _plan("trim"))
    _draw_launcher(addon, monkeypatch, runtime_ready=False, status=memory_error)
    monkeypatch.setattr(
        addon.ui_scene, "_runtime_ui_state",
        lambda: (SimpleNamespace(), "cuda:0", False, memory_error))
    monkeypatch.setattr(
        addon.ui_scene, "activity_ui_state",
        lambda _scene: {"id": "IDLE", "owns_scene": False, "label": "Idle", "icon": "PAUSE"})
    monkeypatch.setattr(addon.scene_bridge, "current_error", lambda _scene: memory_error)
    control = addon.ui_scene.FLUIDSIM_PT_scene_control()
    control.layout = _Layout()
    control.draw(SimpleNamespace(scene=_scene(status=memory_error)))

    # The error box was drawn with the memory error in it, and nothing asked a
    # classifier whether the text was a capacity error.
    assert "fluidsim.clear_error" in control.layout.operator_ids
    assert addon.classifier_calls == []


_GATE_PATH = Path(__file__).resolve().parents[2] / "tools" / "run_production_gate.py"
_MEMORY_LANE = "tests/acceptance/blender_memory_safety_gate.py"
_MEMORY_MARKER = "MEMORY-SAFETY-GATE-PASS"


class _ReachedNextLane(Exception):
    """Raised once the gate has moved past every background script lane."""


def _gate_lanes(monkeypatch, *, drop=None):
    """Record the commands the production gate would run, without running any.

    This executes the gate's own control flow (``_production_main``) up to the
    performance budget, which runs after every background Blender script lane
    (including the quiet batch that follows the concurrent lanes). Every helper that would touch git,
    the disk or Blender is replaced, ``subprocess`` is unreachable, and ``_run``
    only records, so nothing starts. ``drop`` removes one background script from
    the gate's table first.
    """
    spec = importlib.util.spec_from_file_location("run_production_gate_memory_lane", _GATE_PATH)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    lanes = []

    def run(label, command, env=None, reject_output=False, expect_marker=None):
        lanes.append(SimpleNamespace(
            label=label, command=list(command), reject_output=reject_output,
            marker=expect_marker))

    def no_process(*_args, **_kwargs):
        raise AssertionError("inspecting the gate must not start a process")

    def reached_budget():
        raise _ReachedNextLane

    for name, replacement in {
        "_run": run,
        "_reusable_budget_demos": reached_budget,
        "_env": lambda: {},
        "_clear_gate_manifest": lambda: None,
        "_print_gate_metadata": lambda *_args: None,
        "_check_no_tracked_npz": lambda: None,
        "_check_no_tracked_ignored_files": lambda: None,
        "_check_certified_warp": lambda _python, _env: "warp-root",
        "subprocess": SimpleNamespace(
            run=no_process, check_output=no_process, call=no_process, Popen=no_process),
    }.items():
        monkeypatch.setattr(gate, name, replacement)
    if drop is not None:
        monkeypatch.setattr(gate, "BACKGROUND_ACCEPTANCE", tuple(
            entry for entry in gate.BACKGROUND_ACCEPTANCE if entry[0] != drop))
    monkeypatch.setenv("BLENDER_EXE", "blender-under-test")
    monkeypatch.delenv("FLUIDSIM_FULL_GUI_ACCEPTANCE", raising=False)
    with pytest.raises(_ReachedNextLane):
        gate._production_main([])
    return lanes


def _blender_runs_of(lanes, script):
    """Lanes that start Blender on ``script`` (not merely mention its path)."""
    return [
        lane for lane in lanes
        if lane.command[0] == "blender-under-test" and "--python" in lane.command
        and lane.command[lane.command.index("--python") + 1] == script
    ]


def test_production_gate_runs_the_memory_recovery_lane_in_blender(monkeypatch):
    lanes = _gate_lanes(monkeypatch)

    runs = _blender_runs_of(lanes, _MEMORY_LANE)

    assert len(runs) == 1
    assert "--background" in runs[0].command
    # The gate only trusts a lane that prints its marker and rejects failure text.
    assert runs[0].marker == _MEMORY_MARKER and runs[0].reject_output
    printed = [
        node.args[0].value for node in ast.walk(ast.parse(
            (_GATE_PATH.parents[1] / _MEMORY_LANE).read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print"
        and node.args and isinstance(node.args[0], ast.Constant)
    ]
    assert _MEMORY_MARKER in printed, "the script must print the marker the gate waits for"


def test_dropping_the_memory_lane_from_the_gate_is_detected(monkeypatch):
    """The path is still named by the syntax-check lane, so a text search would pass."""
    lanes = _gate_lanes(monkeypatch, drop=_MEMORY_LANE)

    assert _blender_runs_of(lanes, _MEMORY_LANE) == []
    assert any(_MEMORY_LANE in lane.command for lane in lanes if lane.label == "python syntax")
    # Limitation: this proves the gate would start the script, not that Blender
    # runs it successfully; only a gate run on a Blender host proves that.


def test_estimate_caps_leaf_allocation_to_domain_cells():
    settings = _risky_settings()
    settings.leaf_capacity = 1_000_000
    settings.domain_min = (0.0, 0.0, 0.0)
    settings.domain_max = (0.8, 0.8, 0.8)
    settings.voxel_size = 0.1
    settings.max_particles = 0
    settings.max_whitewater = 0
    settings.surface_resolution_mode = 0

    one_leaf = SimpleNamespace(**vars(settings))
    one_leaf.leaf_capacity = 1
    assert estimate_solver_bytes(settings) == estimate_solver_bytes(one_leaf)


def test_surface_dense_memory_is_counted_in_simulation_mode():
    settings = _risky_settings()
    settings.surface_resolution_mode = 0
    settings.surface_max_dense_dim = 512
    with_surface = estimate_solver_bytes(settings)
    settings.surface_enabled = False

    assert with_surface > estimate_solver_bytes(settings)


def test_disabled_surface_does_not_allocate_secondary_grid_estimate():
    settings = _risky_settings()
    settings.surface_enabled = False
    secondary_mode = estimate_solver_bytes(settings)
    settings.surface_resolution_mode = 0

    assert secondary_mode == estimate_solver_bytes(settings)


def test_solver_release_breaks_gpu_owning_backreferences():
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    solver = FluidSolver.__new__(FluidSolver)
    extractor = SimpleNamespace(reset_buffers=lambda: None)
    surface = SimpleNamespace(
        solver=solver, extractor=extractor, render_extractor=extractor
    )
    host_frame = SimpleNamespace(solver=solver)
    sourcing = SimpleNamespace(owner=solver)
    release_calls = []
    pressure_solver = SimpleNamespace(
        release=lambda: release_calls.append("pressure"))
    solver.surface = surface
    solver.sourcing = sourcing
    solver.pressure_solver = pressure_solver
    solver._host_frame_cache = host_frame

    solver.release()

    assert surface.solver is None
    assert sourcing.owner is None
    assert host_frame.solver is None
    # Dropped, not nulled: _render_surface_voxel_size (solver.py:795) and
    # io.cache._frame_payload (cache.py:123) both gate on hasattr, so a None
    # left in place turns the released-state guard into an AttributeError.
    assert not hasattr(solver, "surface")
    assert solver.sourcing is None
    assert solver._host_frame_cache is None
    assert release_calls == ["pressure"]


def test_cuda_release_collects_cycles_and_flushes_pool_before_restoring_threshold(
    monkeypatch,
):
    from fluidSim_warp_blender51.sim import memory_safety

    calls = []
    device = SimpleNamespace(is_cuda=True, is_mempool_supported=True)
    fake_warp = SimpleNamespace(
        get_device=lambda name: calls.append(("device", name)) or device,
        get_mempool_release_threshold=lambda dev: calls.append(("get", dev)) or 123,
        set_mempool_release_threshold=lambda dev, value: calls.append(
            ("set", dev, value)
        ),
        synchronize_device=lambda dev: calls.append(("sync", dev)),
    )
    monkeypatch.setattr(memory_safety.gc, "collect", lambda: calls.append(("gc",)))
    monkeypatch.setitem(sys.modules, "warp", fake_warp)

    memory_safety.release_cuda_memory("cuda:0")

    assert calls == [
        ("gc",),
        ("device", "cuda:0"),
        ("get", device),
        ("set", device, 0),
        ("sync", device),
        ("set", device, 123),
    ]
