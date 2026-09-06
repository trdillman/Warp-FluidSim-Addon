from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fluidsim_testpkg.presets import library
from fluidsim_testpkg.presets.schema import PresetV1


def _user_preset(identifier="my_water"):
    return PresetV1.from_dict({
        "schema_version": 1,
        "id": identifier,
        "name": "My Water",
        "description": "User preset",
        "expected_scale": "metres",
        "performance_cost": "Medium",
        "requires_reset": True,
        "settings": {"voxel_size": 0.08, "flip_ratio": 0.9},
        "source_version": "1.0.249",
    })


def test_all_builtins_validate_and_have_unique_ids():
    records = library.built_in_presets()
    assert {record.preset.id for record in records} == {
        "preview_water", "production_water", "splash_whitewater",
        "honey", "thick_liquid", "thin_liquid",
        "water_surface_tension", "stylized_high_tension",
    }
    assert len(records) == len({record.preset.id for record in records})


def test_pre_phase5_presets_explicitly_disable_surface_tension():
    records = {record.preset.id: record.preset for record in library.built_in_presets()}
    for identifier in (
        "preview_water", "production_water", "splash_whitewater",
        "honey", "thick_liquid", "thin_liquid",
    ):
        assert records[identifier].settings["surface_tension_enabled"] is False


def test_user_preset_atomic_save_and_load(tmp_path: Path):
    path = library.save_user_preset(_user_preset(), tmp_path)
    assert path == tmp_path / "my_water.json"
    assert library.load_preset("USER:my_water", tmp_path) == _user_preset()
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(FileExistsError):
        library.save_user_preset(_user_preset(), tmp_path)


def test_user_cannot_override_builtin(tmp_path: Path):
    with pytest.raises(ValueError, match="built-in"):
        library.save_user_preset(_user_preset("preview_water"), tmp_path)


def test_discovery_and_load_propagate_invalid_user_preset(tmp_path: Path):
    (tmp_path / "broken.json").write_text(
        '{"schema_version": 1, "id": "broken"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        library.discover_presets(tmp_path)
    with pytest.raises(ValueError):
        library.load_preset("USER:broken", tmp_path)


@pytest.mark.parametrize("error", [ImportError("no bpy"), RuntimeError("no profile")])
def test_discovery_tolerates_only_unavailable_user_profile(monkeypatch, error):
    monkeypatch.setattr(library, "user_presets", lambda _directory=None: (_ for _ in ()).throw(error))

    assert library.discover_presets() == library.built_in_presets()


def test_user_preset_dir_supports_legacy_addon_loading(monkeypatch, tmp_path):
    utils = SimpleNamespace(
        user_resource=lambda resource_type: str(tmp_path),
        extension_path_user=lambda *args, **kwargs: pytest.fail(
            "legacy add-on must not use the extension-only storage API"
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(utils=utils))

    directory = library.user_preset_dir(create=True)

    assert directory == tmp_path / "fluidsim_testpkg" / "presets"
    assert directory.is_dir()


def test_user_preset_dir_uses_extension_namespace_when_available(
        monkeypatch, tmp_path):
    calls = []
    utils = SimpleNamespace(
        extension_path_user=lambda package, **kwargs: (
            calls.append((package, kwargs)) or str(tmp_path)
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(utils=utils))
    monkeypatch.setattr(
        library, "__package__", "bl_ext.user_default.fluidsim.presets"
    )

    assert library.user_preset_dir(create=True) == tmp_path
    assert calls == [
        ("bl_ext.user_default.fluidsim", {"path": "presets", "create": True})
    ]


def test_apply_preset_writes_values_to_explicit_target():
    """Post-migration the operator passes the domain-object settings as target;
    preset values must land there, never on the stale scene.fluidsim ghost."""
    class Group:
        def __init__(self):
            self.voxel_size = 0.05
            self.flip_ratio = 0.97
            self.active_preset_id = ""

    stale = Group()
    domain = Group()
    scene = SimpleNamespace(fluidsim=stale)

    result = library.apply_preset(scene, _user_preset(), target=domain)

    # Solver values go to the explicit target (the live domain settings)...
    assert domain.voxel_size == 0.08
    assert domain.flip_ratio == 0.9
    # ...and never touch the stale scene-level group.
    assert stale.voxel_size == 0.05
    assert stale.flip_ratio == 0.97
    # "Applied:" feedback is read from the scene-level group by the panel, so
    # active_preset_id is written there, not on the value target.
    assert stale.active_preset_id == "my_water"
    assert domain.active_preset_id == ""
    assert result.requires_reset is True
    assert set(result.changed) == {"voxel_size", "flip_ratio"}


def _spec_valid_value(spec):
    """A value inside ``spec``'s allowed domain, for building complete groups."""
    if spec.value_type is bool:
        return False
    if spec.choices:
        return spec.choices[0]
    if spec.value_type is str:
        return ""
    base = spec.minimum if spec.minimum is not None else 1
    return spec.value_type(base)


def test_capture_scene_preset_snapshots_explicit_source():
    """Save-as-preset must snapshot the live domain settings, not the ghost."""
    valid = {key: _spec_valid_value(spec) for key, spec in library.SETTING_SPECS.items()}
    stale = SimpleNamespace(**{**valid, "voxel_size": 0.05, "flip_ratio": 0.97})
    domain = SimpleNamespace(**{**valid, "voxel_size": 0.08, "flip_ratio": 0.9})
    scene = SimpleNamespace(fluidsim=stale)

    preset = library.capture_scene_preset(
        scene,
        identifier="from_domain",
        name="From Domain",
        description="Captured from the live domain settings",
        expected_scale="metres",
        performance_cost="Medium",
        source=domain,
    )

    assert preset.settings["voxel_size"] == 0.08
    assert preset.settings["flip_ratio"] == 0.9


def test_apply_preset_rolls_back_on_assignment_failure():
    class Target:
        voxel_size = 0.05
        flip_ratio = 0.97
        active_preset_id = ""

        def __setattr__(self, name, value):
            if name == "flip_ratio" and value == 0.9:
                raise RuntimeError("synthetic RNA failure")
            object.__setattr__(self, name, value)

    target = Target()
    scene = SimpleNamespace(fluidsim=target)
    with pytest.raises(RuntimeError, match="synthetic"):
        library.apply_preset(scene, _user_preset())
    assert target.voxel_size == 0.05
    assert target.flip_ratio == 0.97
