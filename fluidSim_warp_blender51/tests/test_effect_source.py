from __future__ import annotations

import hashlib
import os
import stat
import tempfile

import pytest
from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    ParameterKind,
)
from fluidsim_testpkg.sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)
from fluidsim_testpkg.sim.effects.source import (
    default_module_directory,  # bound at collection, before conftest redirects it
    ensure_private_directory,
    generate_kernel_source,
)


def _instance():
    definition = EffectDefinitionV2(
        effect_id="test.source",
        version="1.0.0",
        name="Source",
        summary="Source",
        description="Source",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        parameters=(
            ParameterDefinitionV2("strength", "Strength", ParameterKind.FLOAT, 2.0),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="particle_kernel",
                label="Particle Kernel",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="value = strength\nvalue = value * 2.0",
                parameter_keys=("strength",),
            ),
        ),
    )
    return EffectInstanceV2("source-1", definition)


def test_generated_source_is_content_addressed_and_maps_user_lines():
    generated = generate_kernel_source(_instance(), 0)
    assert generated.module_name.startswith("fluidsim_effect_")
    assert len(generated.source_digest) == 64
    assert "@wp.kernel" in generated.source
    assert "strength: float" in generated.source
    assert generated.original_line_for_generated(generated.user_line_start) == 1
    assert generated.original_line_for_generated(generated.user_line_start + 1) == 2


def test_equivalent_generation_is_deterministic():
    first = generate_kernel_source(_instance(), 0)
    second = generate_kernel_source(_instance(), 0)
    assert first.source == second.source
    assert first.module_name == second.module_name
    assert first.kernel_name == second.kernel_name


def _symlink_or_skip(target, link, *, directory=False):
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_default_module_directory_is_per_user_not_shared_temp(monkeypatch, tmp_path):
    shared_temp = tmp_path / "shared-temp"
    user_cache = tmp_path / "user-cache"
    monkeypatch.setattr(tempfile, "tempdir", str(shared_temp))
    monkeypatch.setenv("LOCALAPPDATA", str(user_cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(user_cache))
    directory = default_module_directory()
    assert directory == user_cache / "fluidsim_warp" / "effects_v2"
    assert not directory.is_relative_to(shared_temp)


def test_write_cache_file_replaces_planted_module(tmp_path):
    generated = generate_kernel_source(_instance(), 0)
    planted = tmp_path / f"{generated.module_name}.py"
    planted.write_text("raise SystemExit('planted')\n", encoding="utf-8")
    path = generated.write_cache_file(tmp_path)
    assert path == planted
    assert hashlib.sha256(path.read_bytes()).hexdigest() == generated.source_digest
    assert sorted(item.name for item in tmp_path.iterdir()) == [planted.name]


def test_write_cache_file_does_not_write_through_planted_symlink(tmp_path):
    generated = generate_kernel_source(_instance(), 0)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    modules = tmp_path / "modules"
    modules.mkdir()
    link = modules / f"{generated.module_name}.py"
    _symlink_or_skip(victim, link)
    path = generated.write_cache_file(modules)
    assert not path.is_symlink()
    assert path.read_text(encoding="utf-8") == generated.source
    assert victim.read_text(encoding="utf-8") == "untouched"


def test_symlinked_module_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    _symlink_or_skip(real, link, directory=True)
    with pytest.raises(PermissionError, match="not a real directory"):
        ensure_private_directory(link)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_private_directory_removes_group_and_world_access(tmp_path):
    directory = tmp_path / "modules"
    directory.mkdir()
    directory.chmod(0o777)
    ensure_private_directory(directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership")
def test_private_directory_owned_by_another_user_is_refused(monkeypatch, tmp_path):
    owner = tmp_path.stat().st_uid
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    with pytest.raises(PermissionError, match="owned by another user"):
        ensure_private_directory(tmp_path)
