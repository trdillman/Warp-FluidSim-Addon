"""Focused contracts for transactional Guided Setup composition."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_source_roles(monkeypatch):
    fake_bpy = types.ModuleType("bpy")

    class Operator:
        def report(self, levels, message):
            self.reports.append((set(levels), message))

    def prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(Operator=Operator)
    fake_bpy.props = SimpleNamespace(
        BoolProperty=prop,
        EnumProperty=prop,
        IntProperty=prop,
        StringProperty=prop,
    )
    fake_bpy.ops = SimpleNamespace(
        fluidsim=SimpleNamespace(preview_start=lambda: {"FINISHED"})
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_bpy.props)

    package = types.ModuleType("artist_setup_package")
    package.__path__ = []
    domains = types.ModuleType("artist_setup_package.operator_domains")
    domains.__path__ = []
    blender = types.ModuleType("artist_setup_package.blender")
    blender.__path__ = []
    common = types.ModuleType("artist_setup_package.operator_domains.common")

    class SessionBase:
        pass

    common._FLUIDSIM_OT_session_base = SessionBase
    for name in (
        "_bridge", "_configured_device", "_mutation_poll", "_owned_capacity_plan",
        "_owned_capacity_signature", "_pending", "_resolved_cache_path",
        "_runtime_poll", "_scene_identity_token", "_snippets", "_solver_settings",
    ):
        setattr(common, name, lambda *args, **kwargs: None)

    setup_module = types.ModuleType("artist_setup_package.blender.setup_wizard")
    setup_module.setup_selected = lambda *_args: SimpleNamespace(
        message="Scene configured", warnings=()
    )
    for name, module in (
        ("artist_setup_package", package),
        ("artist_setup_package.operator_domains", domains),
        ("artist_setup_package.operator_domains.common", common),
        ("artist_setup_package.blender", blender),
        ("artist_setup_package.blender.setup_wizard", setup_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    path = ROOT / "fluidSim_warp_blender51/operator_domains/source_roles.py"
    spec = importlib.util.spec_from_file_location(
        "artist_setup_package.operator_domains.source_roles", path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, fake_bpy, setup_module


def _operator_context():
    return SimpleNamespace(
        scene=SimpleNamespace(fluidsim=SimpleNamespace(status="")),
    )


def _operator(module, *, start_preview):
    operator = module.FLUIDSIM_OT_setup_scene()
    operator.mode = "QUICK_LIQUID"
    operator.preset_id = ""
    operator.start_preview = start_preview
    operator.reports = []
    return operator


def test_setup_rollback_journal_unwinds_once_in_reverse_order():
    from fluidSim_warp_blender51.blender.setup_wizard import _SetupRollbackJournal

    calls = []
    journal = _SetupRollbackJournal()
    journal.record(lambda: calls.append("first"))
    journal.record(lambda: calls.append("second"))

    journal.rollback()
    journal.rollback()

    assert calls == ["second", "first"]


def test_output_rollback_preserves_existing_non_nodes_modifier():
    from fluidSim_warp_blender51.blender.setup_wizard import (
        _record_output_state,
        _SetupRollbackJournal,
    )

    class Modifier(dict):
        def __init__(self, pointer, name, modifier_type, node_group_marker=False):
            super().__init__()
            self._pointer = pointer
            self.name = name
            self.type = modifier_type
            if node_group_marker:
                self.node_group = "group"

        def as_pointer(self):
            return self._pointer

        def items(self):
            if self.type != "NODES":
                raise TypeError("this modifier type does not support IDProperties")
            return super().items()

    class Modifiers(list):
        pass

    class Output(dict):
        pass

    bevel = Modifier(1, "Artist Bevel", "BEVEL")
    output = Output()
    output.color = [0.1, 0.2, 0.3, 1.0]
    output.display_type = "WIRE"
    output.data = SimpleNamespace(materials=["artist material"])
    output.modifiers = Modifiers([bevel])
    journal = _SetupRollbackJournal()
    capture = _record_output_state(journal, output, ("fluidsim_owned",))

    output.modifiers.append(Modifier(2, "FluidSim Nodes", "NODES", True))
    output.display_type = "TEXTURED"
    output["fluidsim_owned"] = True
    capture()
    journal.rollback()

    assert output.modifiers == [bevel]
    assert not hasattr(bevel, "node_group")
    assert output.display_type == "WIRE"
    assert "fluidsim_owned" not in output


def test_rollback_preserves_later_callback_id_and_property():
    from fluidSim_warp_blender51.blender.setup_wizard import (
        _CreatedSetupIDs,
        _record_property_keys,
        _SetupRollbackJournal,
    )

    class Datablock:
        def __init__(self, pointer):
            self.pointer = pointer
            self.name = f"ID {pointer}"

        def as_pointer(self):
            return self.pointer

    class Registry(list):
        def remove(self, item, do_unlink=False):
            super().remove(item)

    owner = {"existing": "keep"}
    registry = Registry()
    callback_deleted_id = Datablock(1)
    callback_sentinel = Datablock(2)
    setup_id = Datablock(3)
    registry.extend((callback_deleted_id, setup_id))
    journal = _SetupRollbackJournal()
    created = _CreatedSetupIDs(journal)
    created.track(registry, callback_deleted_id, 0)
    created.track(registry, setup_id, 0)
    _record_property_keys(journal, owner, ("fluidsim_setup_id",))

    owner["fluidsim_setup_id"] = "setup-owned"
    owner["callback_sentinel"] = "preserve"
    registry.remove(callback_deleted_id)
    registry.append(callback_sentinel)
    journal.rollback()

    assert registry == [callback_sentinel]
    assert owner == {"existing": "keep", "callback_sentinel": "preserve"}


def test_setup_only_does_not_start_preview(monkeypatch):
    module, fake_bpy, _setup = _load_source_roles(monkeypatch)
    declaration = module.FLUIDSIM_OT_setup_scene.__annotations__["start_preview"]
    assert "default=False" in declaration.replace(" ", "")
    calls = []
    fake_bpy.ops.fluidsim.preview_start = lambda: calls.append(True) or {"FINISHED"}
    context = _operator_context()
    operator = _operator(module, start_preview=False)

    assert operator.execute(context) == {"FINISHED"}
    assert calls == []
    assert context.scene.fluidsim.status == "Scene configured"


def test_setup_and_preview_starts_only_after_setup_commits(monkeypatch):
    module, fake_bpy, setup = _load_source_roles(monkeypatch)
    calls = []

    def configured(*_args):
        calls.append("setup")
        return SimpleNamespace(message="Scene configured", warnings=())

    setup.setup_selected = configured

    def preview():
        calls.append("preview")
        return {"FINISHED"}

    fake_bpy.ops.fluidsim.preview_start = preview
    operator = _operator(module, start_preview=True)

    assert operator.execute(_operator_context()) == {"FINISHED"}
    assert calls == ["setup", "preview"]


@pytest.mark.parametrize(
    ("preview", "reason"),
    [
        (lambda: {"CANCELLED"}, "no playable window"),
        (lambda: (_ for _ in ()).throw(RuntimeError("window disappeared")),
         "window disappeared"),
    ],
)
def test_preview_failure_preserves_committed_setup(monkeypatch, preview, reason):
    module, fake_bpy, _setup = _load_source_roles(monkeypatch)
    context = _operator_context()

    def preview_with_status():
        context.scene.fluidsim.status = "no playable window"
        return preview()

    fake_bpy.ops.fluidsim.preview_start = preview_with_status
    operator = _operator(module, start_preview=True)

    assert operator.execute(context) == {"FINISHED"}
    assert context.scene.fluidsim.status == (
        f"Scene configured; preview could not start: {reason}"
    )
    assert operator.reports[-1] == ({"WARNING"}, context.scene.fluidsim.status)
