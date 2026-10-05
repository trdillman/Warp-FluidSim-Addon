"""Contracts for starter templates, example-package generation and the example-scene builder."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fluidsim_testpkg.resources.examples.custom_effects.example_effects import (
    EXAMPLE_EFFECT_DEFINITIONS,
    generate_all_example_packages,
)
from fluidsim_testpkg.sim.effects.enums import ParameterKind
from fluidsim_testpkg.sim.effects.package import read_effect_package, write_effect_package
from fluidsim_testpkg.sim.effects.schema import EffectDefinitionV2
from fluidsim_testpkg.sim.effects.templates import (
    TEMPLATES,
    get_template_definition,
    starter_template_items,
)

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parent
BUILDER = REPO / "tools" / "release" / "build_custom_effect_example_scenes.py"
EXAMPLES_DIR = ADDON / "resources" / "examples" / "custom_effects"


# --- starter templates ------------------------------------------------------


def test_template_menu_matches_registry():
    assert [key for key, _label, _desc in starter_template_items()] == list(TEMPLATES)


def test_unknown_template_key_raises():
    with pytest.raises(KeyError, match="NOPE"):
        get_template_definition("NOPE")


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_template_is_self_consistent(key):
    definition = get_template_definition(key)
    param_keys = {p.key for p in definition.parameters}
    assert len(param_keys) == len(definition.parameters), "duplicate parameter keys"
    kernel_ids = [k.kernel_id for k in definition.kernels]
    assert len(set(kernel_ids)) == len(kernel_ids), "duplicate kernel ids"
    for kernel in definition.kernels:
        assert set(kernel.parameter_keys) <= param_keys
    assert EffectDefinitionV2.from_dict(definition.to_dict()) == definition


def test_templates_return_fresh_equal_definitions():
    assert get_template_definition("BLANK") == get_template_definition("BLANK")


def test_template_effect_ids_are_unique_and_custom():
    ids = [get_template_definition(key).effect_id for key in TEMPLATES]
    assert len(set(ids)) == len(ids)
    assert all(not get_template_definition(key).built_in for key in TEMPLATES)


def test_template_package_roundtrip(tmp_path):
    definition = get_template_definition("PARTICLE_FORCE")
    path = tmp_path / "t.fseffect"
    write_effect_package(path, definition)
    assert read_effect_package(path) == definition


# --- example packages -------------------------------------------------------


def test_example_parameter_bindings_and_ids_are_consistent():
    assert len({d.effect_id for d in EXAMPLE_EFFECT_DEFINITIONS}) == len(EXAMPLE_EFFECT_DEFINITIONS)
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        param_keys = {p.key for p in definition.parameters}
        for kernel in definition.kernels:
            assert set(kernel.parameter_keys) <= param_keys, (definition.effect_id, kernel.kernel_id)


def test_generate_all_example_packages_writes_readable_packages(tmp_path):
    paths = generate_all_example_packages(tmp_path / "out")
    assert len(paths) == len(EXAMPLE_EFFECT_DEFINITIONS)
    assert all(p.parent == tmp_path / "out" for p in paths)
    by_name = {p.name: p for p in paths}
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        name = f"{definition.effect_id.split('.')[-1]}-{definition.version}.fseffect"
        assert read_effect_package(by_name[name]) == definition


def test_generate_all_example_packages_accepts_a_subset(tmp_path):
    subset = EXAMPLE_EFFECT_DEFINITIONS[:2]
    paths = generate_all_example_packages(tmp_path, subset)
    assert len(paths) == 2


def test_shipped_packages_match_the_generator():
    # Editing example_effects.py without regenerating the .fseffect files would
    # ship stale packages.
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        name = f"{definition.effect_id.split('.')[-1]}-{definition.version}.fseffect"
        assert read_effect_package(EXAMPLES_DIR / name) == definition, name


# --- example scene builder --------------------------------------------------


class _FakeTrustStore:
    def __init__(self, path):
        self.path = Path(path)


@pytest.fixture()
def builder(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", MagicMock())
    monkeypatch.setitem(sys.modules, "tools.repo_addon_loader", MagicMock())
    # main() swaps ``effects.trust_store_for_scene``; fake the add-on modules it
    # touches so the test needs neither Blender nor the import order of other tests.
    pkg = "fluidSim_warp_blender51"
    effects = types.ModuleType(f"{pkg}.blender.effects")
    effects.trust_store_for_scene = lambda _scene=None: "the user's real store"
    trust = types.ModuleType(f"{pkg}.sim.effects.trust")
    trust.TrustStore = _FakeTrustStore
    blender_pkg = types.ModuleType(f"{pkg}.blender")
    blender_pkg.effects = effects
    for mod_name, module in {
        pkg: types.ModuleType(pkg),
        f"{pkg}.blender": blender_pkg,
        f"{pkg}.blender.effects": effects,
        f"{pkg}.sim": types.ModuleType(f"{pkg}.sim"),
        f"{pkg}.sim.effects": types.ModuleType(f"{pkg}.sim.effects"),
        f"{pkg}.sim.effects.trust": trust,
    }.items():
        monkeypatch.setitem(sys.modules, mod_name, module)
    name = "custom_effect_example_scene_builder_under_test"
    spec = importlib.util.spec_from_file_location(name, BUILDER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_builders_are_complete_and_ordered(builder):
    names = list(builder.BUILDERS)
    assert len(names) == 12
    assert names == sorted(names)
    assert [n.split("_", 1)[0] for n in names] == [f"{i:02d}" for i in range(1, 13)]
    assert all(callable(fn) for fn in builder.BUILDERS.values())


def test_builder_names_match_acceptance_expectations(builder):
    source = (
        REPO / "tests" / "acceptance" / "blender_custom_effect_example_scenes_acceptance.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    expected = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "EXPECTED_EXAMPLES" for t in node.targets)
    )
    assert [f"{name}.blend" for name in builder.BUILDERS] == list(expected)


def test_parse_args_defaults_and_flags(builder, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["blender", "--python", "x.py"])
    args = builder.parse_args()
    assert (args.all, args.scene, args.device, args.dry_run) == (False, None, "cpu", False)
    assert args.output_dir == builder.DEFAULT_OUTPUT_DIR

    monkeypatch.setattr(
        sys, "argv",
        ["blender", "--", "--scene", "01_thermal_updraft_example", "--device", "cuda:0", "--dry-run"],
    )
    args = builder.parse_args()
    assert (args.scene, args.device, args.dry_run) == ("01_thermal_updraft_example", "cuda:0", True)


def test_main_builds_only_the_requested_scene(builder, monkeypatch, tmp_path):
    calls = []
    fakes = {
        "a": lambda out, dev, dry: calls.append(("a", out, dev, dry)) or out / "a.blend",
        "b": lambda out, dev, dry: calls.append(("b", out, dev, dry)) or out / "b.blend",
    }
    monkeypatch.setattr(builder, "BUILDERS", fakes)
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "--scene", "b", "--output-dir", str(tmp_path), "--dry-run"]
    )
    builder.main()
    assert calls == [("b", tmp_path.resolve(), "cpu", True)]

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--all", "--output-dir", str(tmp_path)])
    builder.main()
    assert [c[0] for c in calls] == ["a", "b"]


def test_main_builds_with_a_private_trust_store_and_restores_the_real_one(builder, monkeypatch, tmp_path):
    effects = sys.modules["fluidSim_warp_blender51.blender.effects"]
    real = effects.trust_store_for_scene
    seen = []
    monkeypatch.setattr(
        builder, "BUILDERS", {"a": lambda *_: seen.append(effects.trust_store_for_scene()) or tmp_path / "a.blend"}
    )
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--all", "--output-dir", str(tmp_path)])
    builder.main()
    (store,) = seen
    assert isinstance(store, _FakeTrustStore), "builds must trust effects in a private store"
    assert store.path.name == "trust.json"
    assert effects.trust_store_for_scene is real, "the user's trust store must be restored"


def test_main_restores_the_trust_store_when_a_build_fails(builder, monkeypatch, tmp_path):
    effects = sys.modules["fluidSim_warp_blender51.blender.effects"]
    real = effects.trust_store_for_scene

    def boom(*_):
        raise RuntimeError("build failed")

    monkeypatch.setattr(builder, "BUILDERS", {"a": boom})
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--all", "--output-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="build failed"):
        builder.main()
    assert effects.trust_store_for_scene is real


def test_main_rejects_unknown_scene(builder, monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "BUILDERS", {"a": lambda *_: None})
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--scene", "zzz", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="zzz"):
        builder.main()


def test_example_builders_use_valid_effect_parameters_and_save_under_their_key():
    """Statically check every builder against the example definitions.

    A parameter typo in an example scene would only surface in a 12-scene Blender run.
    """
    by_id = {d.effect_id: d for d in EXAMPLE_EFFECT_DEFINITIONS}
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    registry = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BUILDERS"
    )
    key_to_func = {
        ast.literal_eval(k): v.id for k, v in zip(registry.keys, registry.values, strict=True)
    }
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert len(key_to_func) == 12

    for key, func_name in key_to_func.items():
        func = funcs[func_name]
        var_to_effect: dict[str, str] = {}
        layers = []
        saved_name = None
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                for const in ast.walk(node.value):
                    if isinstance(const, ast.Constant) and const.value in by_id:
                        var_to_effect[node.targets[0].id] = const.value
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "add_effect_layer":
                layers.append(node)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "finalize_and_save":
                saved_name = ast.literal_eval(node.args[1])
        assert saved_name == f"{key}.blend", key
        assert layers, f"{key} adds no effect layer"
        for call in layers:
            definition = by_id[var_to_effect[call.args[1].id]]
            keywords = {kw.arg: kw.value for kw in call.keywords}
            params = keywords.get("parameters")
            if params is not None:
                valid = {p.key for p in definition.parameters}
                given = {ast.literal_eval(k) for k in params.keys}
                assert given <= valid, (key, sorted(given - valid))
            needs_controller = any(p.kind is ParameterKind.OBJECT for p in definition.parameters)
            # An unset OBJECT parameter fails the build.  (The reverse is not
            # required: thermal_updraft has no OBJECT parameter, so its example
            # controller empty is only a visual marker.)
            if needs_controller:
                assert "controller" in keywords, (key, definition.effect_id)
