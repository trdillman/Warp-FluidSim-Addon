from __future__ import annotations

from dataclasses import replace

import pytest
from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions
from fluidsim_testpkg.sim.effects.library import (
    compare_effect_versions,
    merge_effect_definitions,
    renamed_definition,
    unique_copy_definition,
)
from fluidsim_testpkg.sim.effects.package import read_effect_package, write_effect_package


@pytest.fixture(params=("authored", "imported"))
def installer(request, tmp_path, monkeypatch):
    from fluidsim_testpkg.blender import effect_library

    root = tmp_path / "library"
    root.mkdir()
    monkeypatch.setattr(effect_library, "user_library_path", lambda: root)

    def install(definition, *, conflict="cancel"):
        if request.param == "authored":
            return effect_library.install_definition(definition, conflict=conflict)
        source = tmp_path / "candidate.fseffect"
        write_effect_package(source, definition)
        return effect_library.install_package(source, conflict=conflict)

    return root, install


def test_install_intents_share_identity_conflicts_and_version_cleanup(installer):
    root, install = installer
    builtin = builtin_effect_definitions()[0]
    original = unique_copy_definition(builtin, {builtin.effect_id})
    first = install(original)
    newer = replace(original, version="2.0.0")
    with pytest.raises(FileExistsError, match="already installed"):
        install(newer)
    assert read_effect_package(first.path).version == original.version
    copied = install(newer, conflict="keep_both")
    assert read_effect_package(copied.path).effect_id != original.effect_id
    replacement = install(newer, conflict="replace")
    assert not first.path.exists()
    assert read_effect_package(replacement.path).version == "2.0.0"
    assert copied.path.exists()
    assert len(list(root.glob("*.fseffect"))) == 2


def test_install_intents_protect_builtins_and_never_grant_trust(installer):
    from fluidsim_testpkg.sim.effects.enums import TrustState
    from fluidsim_testpkg.sim.effects.trust import TrustStore, content_trust_digest

    root, install = installer
    builtin = builtin_effect_definitions()[0]
    with pytest.raises(FileExistsError):
        install(builtin)
    with pytest.raises(ValueError, match="built-in"):
        install(builtin, conflict="replace")
    result = install(builtin, conflict="keep_both")
    restored = read_effect_package(result.path)
    assert not restored.built_in
    store = TrustStore(root.parent / "trust.json")
    assert store.status(content_trust_digest(restored), built_in=restored.built_in) is TrustState.UNTRUSTED_IMPORTED
    assert not store.path.exists()


def test_failed_install_keeps_old_versions(installer, monkeypatch):
    from fluidsim_testpkg.blender import effect_library

    _root, install = installer
    builtin = builtin_effect_definitions()[0]
    original = unique_copy_definition(builtin, {builtin.effect_id})
    first = install(original)
    before = first.path.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("simulated package write failure")

    monkeypatch.setattr(effect_library, "write_effect_package", fail)
    with pytest.raises(OSError, match="write failure"):
        install(replace(original, version="2.0.0"), conflict="replace")
    assert first.path.read_bytes() == before


def test_unique_copy_definition_changes_identity_and_never_claims_builtin_trust():
    original = builtin_effect_definitions()[0]
    copied = unique_copy_definition(
        original,
        {
            original.effect_id,
            f"{original.effect_id}-copy-2",
        },
    )
    assert copied.effect_id == f"{original.effect_id}-copy-3"
    assert copied.name == f"{original.name} Copy 3"
    assert copied.built_in is False
    assert copied.package_origin == "local-copy"
    assert copied.kernels == original.kernels


def test_merge_effect_definitions_rejects_user_shadowing_and_duplicate_ids():
    original = builtin_effect_definitions()[0]
    custom = unique_copy_definition(original, {original.effect_id})
    duplicate = unique_copy_definition(original, {original.effect_id})

    result = merge_effect_definitions(
        builtin_effect_definitions(),
        (
            ("shadow.fseffect", original),
            ("custom-a.fseffect", custom),
            ("custom-b.fseffect", duplicate),
        ),
    )

    ids = {item.effect_id for item in result.library.definitions()}
    assert original.effect_id in ids
    assert custom.effect_id in ids
    assert len([item for item in result.diagnostics if item.code == "IDENTITY_CONFLICT"]) == 2
    assert any(item.source == "shadow.fseffect" for item in result.diagnostics)
    assert any(item.source == "custom-b.fseffect" for item in result.diagnostics)


def test_compare_effect_versions_distinguishes_identity_content_and_semantic_version():
    installed = unique_copy_definition(
        builtin_effect_definitions()[0],
        {builtin_effect_definitions()[0].effect_id},
    )
    identical = installed
    modified = replace(installed, summary=installed.summary + " Modified")
    newer = replace(installed, version="2.0.0")
    older = replace(installed, version="0.9.0")

    assert compare_effect_versions(installed, identical) == "IDENTICAL"
    assert compare_effect_versions(installed, modified) == "MODIFIED"
    assert compare_effect_versions(installed, newer) == "NEWER"
    assert compare_effect_versions(installed, older) == "OLDER"

    with pytest.raises(ValueError, match="same effect ID"):
        compare_effect_versions(installed, replace(installed, effect_id="different.effect"))


def test_renamed_definition_changes_public_identity_without_changing_behavior():
    original = unique_copy_definition(
        builtin_effect_definitions()[0],
        {builtin_effect_definitions()[0].effect_id},
    )
    renamed = renamed_definition(original, "studio.hero-vortex", "Hero Vortex")

    assert renamed.effect_id == "studio.hero-vortex"
    assert renamed.name == "Hero Vortex"
    assert renamed.built_in is False
    assert renamed.package_origin == "local-rename"
    assert renamed.version == original.version
    assert renamed.parameters == original.parameters
    assert renamed.kernels == original.kernels


def test_compare_package_update_reports_not_installed_identical_modified_and_newer(tmp_path, monkeypatch):
    from fluidsim_testpkg.blender import effect_library

    monkeypatch.setattr(effect_library, "user_library_path", lambda: tmp_path / "library")
    root = effect_library.user_library_path()
    root.mkdir(parents=True)
    base = unique_copy_definition(
        builtin_effect_definitions()[0],
        {builtin_effect_definitions()[0].effect_id},
    )
    installed_path = root / f"{base.effect_id}-{base.version}.fseffect"
    write_effect_package(installed_path, base)

    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    identical_path = candidate_dir / "identical.fseffect"
    write_effect_package(identical_path, base)
    assert effect_library.compare_package_update(identical_path).status == "IDENTICAL"

    modified = replace(base, summary=base.summary + " Modified")
    modified_path = candidate_dir / "modified.fseffect"
    write_effect_package(modified_path, modified)
    assert effect_library.compare_package_update(modified_path).status == "MODIFIED"

    newer = replace(base, version="2.0.0")
    newer_path = candidate_dir / "newer.fseffect"
    write_effect_package(newer_path, newer)
    comparison = effect_library.compare_package_update(newer_path)
    assert comparison.status == "NEWER"
    assert comparison.installed_version == base.version
    assert comparison.candidate_version == "2.0.0"
    assert comparison.installed_source == str(installed_path)

    absent = replace(base, effect_id="studio.absent-effect")
    absent_path = candidate_dir / "absent.fseffect"
    write_effect_package(absent_path, absent)
    assert effect_library.compare_package_update(absent_path).status == "NOT_INSTALLED"


def test_compare_package_update_rejects_builtin_identity_and_reports_ambiguous_install(tmp_path, monkeypatch):
    from fluidsim_testpkg.blender import effect_library

    monkeypatch.setattr(effect_library, "user_library_path", lambda: tmp_path / "library")
    root = effect_library.user_library_path()
    root.mkdir(parents=True)
    builtin = builtin_effect_definitions()[0]
    builtin_path = tmp_path / "builtin.fseffect"
    write_effect_package(builtin_path, builtin)
    assert effect_library.compare_package_update(builtin_path).status == "BUILTIN_CONFLICT"

    custom = unique_copy_definition(builtin, {builtin.effect_id})
    write_effect_package(root / "one.fseffect", custom)
    write_effect_package(root / "two.fseffect", replace(custom, version="2.0.0"))
    candidate_path = tmp_path / "candidate.fseffect"
    write_effect_package(candidate_path, replace(custom, version="3.0.0"))
    assert effect_library.compare_package_update(candidate_path).status == "AMBIGUOUS"


def test_rename_installed_package_is_atomic_preserves_code_and_removes_old_path(tmp_path, monkeypatch):
    from fluidsim_testpkg.blender import effect_library

    monkeypatch.setattr(effect_library, "user_library_path", lambda: tmp_path / "library")
    root = effect_library.user_library_path()
    root.mkdir(parents=True)
    original = unique_copy_definition(
        builtin_effect_definitions()[0],
        {builtin_effect_definitions()[0].effect_id},
    )
    old_path = root / f"{original.effect_id}-{original.version}.fseffect"
    write_effect_package(old_path, original)

    installed = effect_library.rename_installed_package(
        old_path,
        new_effect_id="studio.renamed-vortex",
        new_name="Renamed Vortex",
    )

    assert not old_path.exists()
    assert installed.path == root / f"studio.renamed-vortex-{original.version}.fseffect"
    restored = read_effect_package(installed.path)
    assert restored.effect_id == "studio.renamed-vortex"
    assert restored.name == "Renamed Vortex"
    assert restored.kernels == original.kernels
    assert restored.parameters == original.parameters


def test_rename_installed_package_rejects_outside_paths_and_identity_collisions(tmp_path, monkeypatch):
    from fluidsim_testpkg.blender import effect_library

    monkeypatch.setattr(effect_library, "user_library_path", lambda: tmp_path / "library")
    root = effect_library.user_library_path()
    root.mkdir(parents=True)
    builtin = builtin_effect_definitions()[0]
    first = unique_copy_definition(builtin, {builtin.effect_id})
    second = unique_copy_definition(builtin, {builtin.effect_id, first.effect_id})
    first_path = root / "first.fseffect"
    second_path = root / "second.fseffect"
    write_effect_package(first_path, first)
    write_effect_package(second_path, second)

    outside = tmp_path / "outside.fseffect"
    write_effect_package(outside, first)
    with pytest.raises(ValueError, match="installed"):
        effect_library.rename_installed_package(
            outside, new_effect_id="studio.outside", new_name="Outside"
        )
    with pytest.raises(ValueError, match="built-in"):
        effect_library.rename_installed_package(
            first_path, new_effect_id=builtin.effect_id, new_name="Collision"
        )
    with pytest.raises(FileExistsError, match="already installed"):
        effect_library.rename_installed_package(
            first_path, new_effect_id=second.effect_id, new_name="Collision"
        )
