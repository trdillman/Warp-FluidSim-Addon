"""Regression tests for custom-effect authoring review findings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    ParameterKind,
)
from fluidsim_testpkg.sim.effects.schema import KernelDefinitionV2, ParameterDefinitionV2

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "blender" / "effect_workspace.py"


@pytest.fixture()
def effects_mod(monkeypatch):
    fake_bpy = MagicMock()
    fake_bpy.types = SimpleNamespace(
        PropertyGroup=object, Operator=object, Panel=object, UIList=object
    )
    fake_props = MagicMock()
    fake_bpy.props = fake_props
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_props)
    name = "fluidsim_testpkg.blender.effects_regression"
    spec = importlib.util.spec_from_file_location(name, ROOT / "blender" / "effects.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def workspace_mod(effects_mod, monkeypatch):
    import fluidsim_testpkg.blender as blender

    monkeypatch.setattr(blender, "effects", effects_mod, raising=False)
    name = "fluidsim_testpkg.blender.workspace_regression"
    spec = importlib.util.spec_from_file_location(name, WORKSPACE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class Collection(list):
    """RNA's index-based collection operations, without Blender."""
    def add(self):
        value = SimpleNamespace(fields=Collection())
        self.append(value)
        return value

    def remove(self, index):
        self.pop(index)

    def move(self, index, target):
        self.insert(target, self.pop(index))


def test_parameter_removal_unbinds_before_rebuilding_and_keeps_applied_state(effects_mod, monkeypatch):
    pg, kernel = _remove_target("strength, radius", True)
    pg.parameters = Collection([SimpleNamespace(key="strength"), SimpleNamespace(key="radius")])
    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(applied_snapshot_json="accepted"))
    rebuilt = []
    monkeypatch.setattr(effects_mod, "rebuild_definition_json", lambda _scene, item: rebuilt.append(
        ([p.key for p in item.parameters], kernel.parameter_keys_csv)))
    effects_mod.remove_parameter(scene, pg, 0)
    assert rebuilt == [(["radius"], "radius")]
    assert scene.fluidsim_effects.applied_snapshot_json == "accepted"


def test_structural_field_edits_update_existing_binding_and_rebuild(effects_mod, monkeypatch):
    fields = Collection([SimpleNamespace(field="velocity", access="READ_CURRENT")])
    pg = SimpleNamespace(kernels=[SimpleNamespace(fields=fields)])
    rebuilt = []
    monkeypatch.setattr(effects_mod, "rebuild_definition_json", lambda scene, item: rebuilt.append(item))
    effects_mod.bind_kernel_field(None, pg, 0, "velocity", "READ_WRITE_CURRENT")
    assert len(fields) == 1 and fields[0].access == "READ_WRITE_CURRENT"
    effects_mod.bind_kernel_field(None, pg, 0, "position", "READ_CURRENT")
    effects_mod.remove_kernel_field(None, pg, 0, 0)
    assert [(f.field, f.access) for f in fields] == [("position", "READ_CURRENT")]
    assert rebuilt == [pg, pg, pg]


def test_add_kernel_owns_new_text_without_overwriting_a_named_text(effects_mod, monkeypatch):
    class Text(dict):
        def write(self, body):
            self.body = body

    foreign = Text()
    foreign.body = "user code"
    created = []

    def new(name):
        text = Text()
        text.name = name + ".001"
        created.append(text)
        return text

    effects_mod.bpy.data.texts = SimpleNamespace(new=new, get=lambda _name: foreign)
    pg = SimpleNamespace(instance_id="effect-a", kernels=Collection([
        SimpleNamespace(kernel_id="kernel_1"), SimpleNamespace(kernel_id="kernel_3")]))
    rebuilt = []
    monkeypatch.setattr(effects_mod, "rebuild_definition_json", lambda scene, item: rebuilt.append(item))
    kernel = effects_mod.add_kernel(None, pg)
    assert kernel.kernel_id == "kernel_4"
    assert kernel.source_text is created[0]
    assert kernel.source_text["fluidsim_effect_instance"] == pg.instance_id
    assert kernel.source_text["fluidsim_effect_kernel"] == "kernel_4"
    assert kernel.parameter_keys_authored
    assert foreign.body == "user code"
    assert rebuilt == [pg]


def test_remove_kernel_preserves_last_kernel_and_drops_only_owned_source(effects_mod, monkeypatch):
    class Text(dict):
        name = "owned"
        users = 1

    text = Text(fluidsim_effect_instance="effect-a")
    pg = SimpleNamespace(instance_id="effect-a", kernels=Collection([
        SimpleNamespace(source_text=text), SimpleNamespace(source_text=None)]))
    removed, rebuilt = [], []
    effects_mod.bpy.data.texts = SimpleNamespace(get=lambda _name: text, remove=removed.append)
    monkeypatch.setattr(effects_mod, "rebuild_definition_json", lambda scene, item: rebuilt.append(item))
    effects_mod.remove_kernel(None, pg, 0)
    assert removed == [text] and rebuilt == [pg]
    assert len(pg.kernels) == 1
    with pytest.raises(ValueError, match="at least one kernel"):
        effects_mod.remove_kernel(None, pg, 0)
    assert len(pg.kernels) == 1 and len(rebuilt) == 1


def test_move_effect_preserves_order_selection_and_rejects_boundary(effects_mod, monkeypatch):
    first, second = SimpleNamespace(order=0), SimpleNamespace(order=1)
    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(items=Collection([first, second]), active_index=0))
    refreshed = []
    monkeypatch.setattr(effects_mod, "refresh_stack_state", refreshed.append)
    assert not effects_mod.move_effect(scene, 0, -1)
    assert effects_mod.move_effect(scene, 0, 1)
    assert scene.fluidsim_effects.items == [second, first]
    assert second.order == 0 and first.order == 1
    assert scene.fluidsim_effects.active_index == 1
    assert refreshed == [scene]


def test_authored_save_uses_identity_policy_and_does_not_touch_trust(workspace_mod, effects_mod, tmp_path, monkeypatch):
    from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions
    from fluidsim_testpkg.sim.effects.library import unique_copy_definition

    library = workspace_mod.effect_library
    root = tmp_path / "library"
    root.mkdir()
    monkeypatch.setattr(library, "user_library_path", lambda: root)
    builtin = builtin_effect_definitions()[0]
    definition = unique_copy_definition(builtin, {builtin.effect_id})
    pg = SimpleNamespace(kernels=[])
    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(items=[pg], active_index=0))
    monkeypatch.setattr(effects_mod, "rebuild_definition_json", lambda *_: None)
    monkeypatch.setattr(effects_mod, "exportable_definition", lambda *_: definition)
    trust = MagicMock(side_effect=AssertionError("saving must not grant trust"))
    monkeypatch.setattr(effects_mod, "trust_effect", trust)
    operator = workspace_mod.FLUIDSIM_OT_effect_install_to_library()
    operator.conflict = "cancel"
    reports = []
    operator.report = lambda level, message: reports.append((level, message))
    assert operator.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert operator.execute(SimpleNamespace(scene=scene)) == {"CANCELLED"}
    assert "already installed" in reports[-1][1]
    operator.conflict = "keep_both"
    assert operator.execute(SimpleNamespace(scene=scene)) == {"FINISHED"}
    assert len(list(root.glob("*.fseffect"))) == 2
    trust.assert_not_called()


def test_effect_registration_unwinds_classes_if_pointer_creation_fails(effects_mod, monkeypatch):
    scene_type = type("Scene", (), {})
    effects_mod.bpy.types.Scene = scene_type

    def fail(**_kwargs):
        raise RuntimeError("pointer creation failed")

    monkeypatch.setattr(effects_mod, "PointerProperty", fail)
    with pytest.raises(RuntimeError, match="pointer creation failed"):
        effects_mod.register()
    assert not hasattr(scene_type, "fluidsim_effects")
    registered = [call.args[0] for call in effects_mod.bpy.utils.register_class.call_args_list]
    unregistered = [call.args[0] for call in effects_mod.bpy.utils.unregister_class.call_args_list]
    assert registered == list(effects_mod.CLASSES)
    assert unregistered == list(reversed(effects_mod.CLASSES))


def test_effect_registration_preserves_existing_scene_pointer(effects_mod):
    pointer = object()
    effects_mod.bpy.types.Scene = type("Scene", (), {"fluidsim_effects": pointer})
    with pytest.raises(RuntimeError, match="already registered"):
        effects_mod.register()
    assert effects_mod.bpy.types.Scene.fluidsim_effects is pointer
    effects_mod.bpy.utils.register_class.assert_not_called()


def test_unique_id_survives_removals(effects_mod):
    # Removing the middle item leaves {param_1, param_3}; len()+1 would reuse param_3.
    assert effects_mod.unique_id("param", ["param_1", "param_3"]) == "param_4"
    assert effects_mod.unique_id("kernel", []) == "kernel_1"
    assert effects_mod.unique_id("kernel", ["kernel_1"]) == "kernel_2"


def test_clear_effects_preserves_foreign_and_shared_source_texts(effects_mod, monkeypatch):
    class Text(dict):
        def __init__(self, name, owner=None, users=1):
            super().__init__()
            self.name, self.users = name, users
            if owner:
                self["fluidsim_effect_instance"] = owner

    foreign = Text("user source")
    owned = Text("owned", "effect-a")
    shared = Text("shared", "effect-a", users=2)
    removed = []
    texts = {t.name: t for t in (foreign, owned, shared)}
    effects_mod.bpy.data.texts = SimpleNamespace(get=texts.get, remove=removed.append)
    item = SimpleNamespace(instance_id="effect-a", kernels=[
        SimpleNamespace(source_text=t) for t in (foreign, owned, shared)
    ])
    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(items=[item], active_index=0))
    monkeypatch.setattr(effects_mod, "refresh_stack_state", lambda _scene: None)
    effects_mod.clear_effects(scene, remove_owned_texts=True)
    assert removed == [owned]
    assert scene.fluidsim_effects.items == []


def test_remove_effect_does_not_delete_a_shared_text_after_user_count_drops(effects_mod, monkeypatch):
    class Text(dict):
        name = "shared source"
        users = 2

    text = Text(fluidsim_effect_instance="effect-a")
    first = SimpleNamespace(instance_id="effect-a", order=0, kernels=[SimpleNamespace(source_text=text)])
    second = SimpleNamespace(instance_id="effect-b", order=1, kernels=[SimpleNamespace(source_text=text)])

    class Items(Collection):
        def remove(self, index):
            super().remove(index)
            text.users -= 1

    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(items=Items([first, second]), active_index=0))
    removed = []
    effects_mod.bpy.data.texts = SimpleNamespace(get=lambda _name: text, remove=removed.append)
    monkeypatch.setattr(effects_mod, "refresh_stack_state", lambda _scene: None)
    effects_mod.remove_effect(scene, 0)
    assert text.users == 1 and removed == []
    assert scene.fluidsim_effects.items == [second]
    assert second.order == 0 and scene.fluidsim_effects.active_index == 0


def test_pg_limit_clears_when_unticked_and_keeps_exact_value(effects_mod):
    assert effects_mod._pg_limit(False, 5.0, 5.0) is None
    exact = 0.1  # not representable in float32
    stored = effects_mod._as_float32(exact)
    assert stored != exact
    assert effects_mod._pg_limit(True, stored, exact) == exact
    assert effects_mod._pg_limit(True, 2.5, exact) == 2.5
    assert effects_mod._pg_limit(True, 3.0, None) == 3.0


def _parameter_pg(**overrides):
    values = dict(
        key="strength",
        label="Strength",
        kind="FLOAT",
        unit="",
        description="",
        enum_items_json="",
        has_minimum=False,
        has_maximum=False,
        minimum=0.0,
        maximum=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parameter_limits_can_be_cleared(effects_mod, monkeypatch):
    monkeypatch.setattr(effects_mod, "_parameter_value", lambda _item: 1.0)
    base = ParameterDefinitionV2(
        key="strength", label="Strength", kind=ParameterKind.FLOAT,
        default=1.0, minimum=0.0, maximum=10.0,
    )
    result = effects_mod.parameter_definition_from_pg(_parameter_pg(), base)
    assert result.minimum is None and result.maximum is None


def _kernel_pg(csv, authored):
    return SimpleNamespace(
        kernel_id="k",
        label="K",
        domain=EffectDomain.PARTICLES.value,
        stage=EffectStage.BEFORE_PRESSURE.value,
        cadence=EffectCadence.EVERY_SUBSTEP.value,
        source_text=None,
        fields=[],
        parameter_keys_csv=csv,
        parameter_keys_authored=authored,
        safe_mode=True,
        frame_start=1,
        frame_end=250,
        frame_step=1,
    )


def _kernel_base():
    return KernelDefinitionV2(
        kernel_id="k",
        label="K",
        domain=EffectDomain.PARTICLES,
        stage=EffectStage.BEFORE_PRESSURE,
        cadence=EffectCadence.EVERY_SUBSTEP,
        source="pass",
        parameter_keys=("a", "b"),
    )


def test_kernel_parameter_keys_can_be_emptied_once_authored(effects_mod):
    result = effects_mod.kernel_definition_from_pg(_kernel_pg("", True), _kernel_base())
    assert result.parameter_keys == ()


def test_kernel_parameter_keys_fall_back_for_legacy_blends(effects_mod):
    result = effects_mod.kernel_definition_from_pg(_kernel_pg("", False), _kernel_base())
    assert result.parameter_keys == ("a", "b")


def test_kernel_parameter_keys_follow_csv_edits(effects_mod):
    result = effects_mod.kernel_definition_from_pg(_kernel_pg("b", False), _kernel_base())
    assert result.parameter_keys == ("b",)


def test_workspace_operators_use_real_report_attributes():
    source = WORKSPACE.read_text(encoding="utf-8")
    assert "report.passed" in source
    assert "report.success" not in source
    assert "invoke_props_dialog" in source  # installation conflict choices are explicit


def test_edited_default_is_exported_and_unedited_default_stays_exact(effects_mod):
    # An edited value becomes the exported default.
    assert effects_mod._pg_default(2.5, 1.0) == 2.5
    assert effects_mod._pg_default((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) == (0.0, 1.0, 0.0)
    assert effects_mod._pg_default("HIGH", "LOW") == "HIGH"
    assert effects_mod._pg_default(True, False) is True
    # An untouched float32 PG value keeps the exact definition value (stable digest).
    stored = effects_mod._as_float32(0.1)
    assert effects_mod._pg_default(stored, 0.1) == 0.1
    assert effects_mod._pg_default((stored, 0.0, 1.0), (0.1, 0.0, 1.0)) == (0.1, 0.0, 1.0)
    assert effects_mod._pg_default(3.0, None) == 3.0


def test_save_and_export_bake_current_values_as_defaults():
    source = WORKSPACE.read_text(encoding="utf-8")
    ui = (ROOT / "blender" / "effect_ui.py").read_text(encoding="utf-8")
    assert "effects.exportable_definition(" in source
    assert "effects.exportable_definition(" in ui


def test_save_to_user_library_delegates_installation_policy():
    source = WORKSPACE.read_text(encoding="utf-8")
    body = source.split("class FLUIDSIM_OT_effect_install_to_library", 1)[1].split("class FLUIDSIM_OT_open_effect_authoring_workspace", 1)[0]
    assert "effect_library.install_definition(" in body
    assert "write_effect_package" not in body


def test_parameter_editor_covers_every_kind():
    source = WORKSPACE.read_text(encoding="utf-8")
    for prop in ("bool_value", "vector_value", "color_value", "enum_value", "enum_items_json"):
        assert f'p_box.prop(param, "{prop}"' in source or f'r_vec.prop(param, "{prop}"' in source, prop


def _remove_target(csv, authored, definition_json=""):
    kernel = SimpleNamespace(
        kernel_id="voxel_buoyancy", parameter_keys_csv=csv, parameter_keys_authored=authored
    )
    return SimpleNamespace(definition_json=definition_json, kernels=[kernel]), kernel


def test_unbind_parameter_key_strips_authored_csv(effects_mod):
    pg, kernel = _remove_target("strength, radius", True)
    effects_mod.unbind_parameter_key(pg, "strength")
    assert kernel.parameter_keys_csv == "radius"
    effects_mod.unbind_parameter_key(pg, "missing")
    assert kernel.parameter_keys_csv == "radius"


def test_unbind_parameter_key_handles_legacy_fallback_keys(effects_mod):
    import json

    from fluidsim_testpkg.sim.effects.templates import voxel_acceleration_template

    definition = voxel_acceleration_template()
    assert definition.kernels[0].parameter_keys == ("buoyancy",)
    pg, kernel = _remove_target("", False, json.dumps(definition.to_dict()))
    effects_mod.unbind_parameter_key(pg, "buoyancy")
    assert kernel.parameter_keys_csv == ""
    assert kernel.parameter_keys_authored is True  # emptied csv now really unbinds
    result = effects_mod.kernel_definition_from_pg(
        SimpleNamespace(**{**vars(_kernel_pg("", True)), "kernel_id": "voxel_buoyancy"}),
        definition.kernels[0],
    )
    assert result.parameter_keys == ()
