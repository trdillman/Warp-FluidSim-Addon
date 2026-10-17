"""Guard regressions: operator polls, panel draw safety, released solvers,
and out-of-domain topology boxes."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from types import SimpleNamespace

import pytest
from settings_doubles import split_settings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Operators whose execute() touches the live session but which must stay
# clickable without one:
#   initialize             - it is what creates the session
#   clear_error            - clearing a stale error is the no-session recovery
#   toggle_live            - initializes the solver itself when enabling
#   export_cache_sequence  - Session.export_cache_sequence calls
#                            self.initialize(scene) internally, so it
#                            legitimately runs with no live session
#   write_report           - reports available diagnostics even when no solver
#                            session is active
#   apply_product_preset   - only touches the session to reset in place when one
#                            already owns the scene; with no session the preset
#                            values sit on the domain object and apply at
#                            initialize time, so it must stay clickable
_NO_POLL_REQUIRED = {
    "FLUIDSIM_OT_initialize",
    "FLUIDSIM_OT_clear_error",
    "FLUIDSIM_OT_toggle_live",
    "FLUIDSIM_OT_export_cache_sequence",
    "FLUIDSIM_OT_write_report",
    "FLUIDSIM_OT_apply_product_preset",
    # Explicit reinitialize is also a no-session recovery path when a scene
    # retains a per-scene pending snapshot after its solver was released.
    "FLUIDSIM_OT_apply_pending_reinitialize",
}

# Sanity floor for the source detector below: if it stops recognising these,
# the sweep test would pass vacuously.
_KNOWN_SESSION_OPERATORS = {
    "FLUIDSIM_OT_release",
    "FLUIDSIM_OT_reset",
    "FLUIDSIM_OT_step",
    "FLUIDSIM_OT_inject_particles",
    "FLUIDSIM_OT_run_validation",
    "FLUIDSIM_OT_run_benchmark",
    "FLUIDSIM_OT_write_report",
    "FLUIDSIM_OT_export_frame",
}


def _install_bpy_stub(monkeypatch):
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

    def _prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(
        Operator=_Operator,
        Panel=_Panel,
        AddonPreferences=object,
        PropertyGroup=object,
        Text=object,
        Object=object,
        Material=object,
        Image=object,
    )
    fake_bpy.props = SimpleNamespace(
        BoolProperty=lambda **kwargs: _prop(**kwargs),
        BoolVectorProperty=lambda **kwargs: _prop(**kwargs),
        CollectionProperty=lambda **kwargs: _prop(**kwargs),
        EnumProperty=lambda **kwargs: _prop(**kwargs),
        FloatProperty=lambda **kwargs: _prop(**kwargs),
        FloatVectorProperty=lambda **kwargs: _prop(**kwargs),
        IntProperty=lambda **kwargs: _prop(**kwargs),
        PointerProperty=lambda **kwargs: _prop(**kwargs),
        StringProperty=lambda **kwargs: _prop(**kwargs),
    )
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(addons={}),
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
    operator_domain_prefix = "fluidSim_warp_blender51.operator_domains"
    for loaded_name in tuple(sys.modules):
        if loaded_name == name or (
            name == "fluidSim_warp_blender51.operators"
            and loaded_name.startswith(operator_domain_prefix)
        ):
            sys.modules.pop(loaded_name, None)
    return importlib.import_module(name)


def _requires_live_solver(cls) -> bool:
    """True when the operator body reaches for the live session object."""
    source = inspect.getsource(cls)
    return "session()" in source or "sess.solver" in source


def test_operators_touching_the_live_session_all_have_a_poll(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")

    detected = {cls.__name__ for cls in operators.CLASSES if _requires_live_solver(cls)}
    assert _KNOWN_SESSION_OPERATORS <= detected

    monkeypatch.setattr(
        operators._operator_runtime,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: SimpleNamespace(active=False)),
    )
    context = SimpleNamespace(selected_objects=(), active_object=None)

    missing = []
    for cls in operators.CLASSES:
        if cls.__name__ in _NO_POLL_REQUIRED or not _requires_live_solver(cls):
            continue
        poll = getattr(cls, "poll", None)
        if poll is None:
            missing.append(cls.__name__)
            continue
        cls._poll_message = ""
        if poll(context) is not False or not cls._poll_message:
            missing.append(cls.__name__)
    assert missing == []

    # Legacy export initializes the solver itself but still requires a ready
    # Warp runtime.
    assert "poll" in vars(operators.FLUIDSIM_OT_export_cache_sequence)
    assert not issubclass(
        operators.FLUIDSIM_OT_export_cache_sequence,
        operators._operator_runtime._FLUIDSIM_OT_session_base,
    )


def test_facade_registers_every_operator_a_domain_module_defines(monkeypatch):
    """Operators are listed nowhere but their class statements: the facade
    discovers them, registers each once, and unregisters in reverse."""
    import ast
    from pathlib import Path

    fake_bpy = _install_bpy_stub(monkeypatch)
    registered, unregistered = [], []
    fake_bpy.utils = SimpleNamespace(
        register_class=registered.append, unregister_class=unregistered.append
    )
    operators = _reload_module("fluidSim_warp_blender51.operators")

    domains = Path(operators.__file__).parent / "operator_domains"
    defined = sorted(
        node.name
        for path in domains.glob("*.py")
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name.startswith("FLUIDSIM_OT_")
    )
    assert defined and sorted(cls.__name__ for cls in operators.CLASSES) == defined
    assert all(getattr(operators, cls.__name__) is cls for cls in operators.CLASSES)

    operators.register()
    operators.unregister()

    assert registered == list(operators.CLASSES)
    assert unregistered == list(reversed(operators.CLASSES))


def test_clear_object_role_poll_explains_why_it_is_greyed_out(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    cls = operators.FLUIDSIM_OT_clear_object_role
    cls._poll_message = ""
    untagged = SimpleNamespace(type="MESH", fluidsim=SimpleNamespace(role="NONE"))

    assert cls.poll(SimpleNamespace(selected_objects=(untagged,))) is False
    assert cls._poll_message


def test_scene_domain_panel_draw_does_not_inspect_broken_domain_geometry(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")

    class _Layout:
        def __init__(self):
            self.properties = []
            self.use_property_split = False
            self.use_property_decorate = True

        def grid_flow(self, **_kwargs):
            return self

        def prop(self, _data, name, **_kwargs):
            self.properties.append(name)

    # A domain object that no longer exposes geometry (deleted / wrong type):
    # _domain_bounds raises, and a draw callback must not propagate that.
    scene = SimpleNamespace(
        fluidsim=split_settings(SimpleNamespace(
            domain_object=SimpleNamespace(),
            voxel_size=0.05,
            resolution_divisions=64,
            particles_per_voxel=8,
            max_particles=1000,
            auto_grow_particles=True,
            topology_expand=1,
            open_boundaries=False,
        ))
    )
    layout = _Layout()
    panel = ui_scene.FLUIDSIM_PT_scene_domain()
    panel.layout = layout

    panel.draw(SimpleNamespace(scene=scene))

    assert layout.properties == [
        "domain_object",
        "resolution_divisions",
        "voxel_size",
        "particles_per_voxel",
        "max_particles",
        "auto_grow_particles",
        "topology_expand",
        "open_boundaries",
    ]


def test_released_solver_reports_no_surface_pipeline():
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    extractor = SimpleNamespace(reset_buffers=lambda: None)
    surface = SimpleNamespace(
        extractor=extractor, render_extractor=extractor, solver=object()
    )
    solver = SimpleNamespace(_host_frame_cache=None, surface=surface, sourcing=None)

    FluidSolver.release(solver)

    # io.cache._frame_payload and FluidSolver._render_surface_voxel_size both
    # gate on this; leaving a None behind turns the guard into an AttributeError
    # that masks the solver error being handled.
    assert not hasattr(solver, "surface")
    assert (
        solver.surface.metadata()
        if hasattr(solver, "surface")
        else {"voxel_size": 0.05}
    ) == {"voxel_size": 0.05}


def test_mark_world_box_rejects_a_fully_out_of_domain_box(monkeypatch):
    pytest.importorskip("warp")
    fields = importlib.import_module("fluidSim_warp_blender51.sim.fields")

    launches = []
    monkeypatch.setattr(
        fields.wp, "launch", lambda *args, **kwargs: launches.append(kwargs)
    )
    grid = SimpleNamespace(
        voxel_size=0.1,
        origin=(0.0, 0.0, 0.0),
        voxels_dim=(64, 64, 64),
        leaves_dim=(8, 8, 8),
        leaf_flags=None,
        device="cpu",
    )

    # Entirely below domain_min: clamping would glue it to the floor corner.
    with pytest.raises(ValueError):
        fields.SparseGrid.mark_world_box(grid, (-5.0, -5.0, -5.0), (-4.0, -4.0, -4.0))
    assert launches == []

    # Straddling the floor: still clamped to the overlapping part.
    fields.SparseGrid.mark_world_box(grid, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    assert len(launches) == 1
