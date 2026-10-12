"""Headless bake exit codes name the right remediation category.

Regression for review item B31x: a corrupt cache manifest and a resume
refusal surfaced as exit 13 (invalid scene) instead of 14 (resume/cache), so
farm automation was told to fix the scene when the cache was the problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import command_line
from fluidSim_warp_blender51.io import cache_schema

PKG = "fluidSim_warp_blender51"


@dataclass(frozen=True)
class _Policy:
    allowed: bool = True
    code: str = "OK"
    message: str = ""


@dataclass(frozen=True)
class _Device:
    alias: str = "cuda:0"


class ResumeRefused(ValueError):
    pass


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run ``command_line.run`` against stubbed Blender, runtime and bake owners."""
    domain = SimpleNamespace(fluidsim_domain=SimpleNamespace(height_enabled=True))
    scene = SimpleNamespace(
        name="Scene",
        objects=[domain],
        fluidsim=SimpleNamespace(
            domain_object=domain, cache_path="", bake_start=1, bake_end=3,
            checkpoint_interval=1, cache_compress=False),
    )
    bpy = _module(
        "bpy",
        context=SimpleNamespace(scene=scene, window=None),
        data=SimpleNamespace(scenes={}, filepath=""),
        app=SimpleNamespace(version_string="5.2.0"),
    )
    state = SimpleNamespace(
        resume_decision=lambda _scene: SimpleNamespace(
            allowed=True, code="OK", message=""),
        bake_run=None,
        bake_calls=[],
    )

    class _Backend:
        def __init__(self, **_kwargs):
            pass

        def resume_decision(self, scene, **kwargs):
            return state.resume_decision(scene, **kwargs)

    def _bake_run(*_args, **kwargs):
        state.bake_calls.append(kwargs)
        if state.bake_run is not None:
            state.bake_run()
        raise AssertionError("the bake loop is not under test")

    stubs = {
        "bpy": bpy,
        f"{PKG}.blender.effects": _module(
            f"{PKG}.blender.effects",
            scene_effect_instances=lambda _scene: (),
            trust_store_for_scene=lambda _scene: None),
        f"{PKG}.sim.effects.policy": _module(
            f"{PKG}.sim.effects.policy",
            evaluate_custom_code_policy=lambda *_args: _Policy()),
        f"{PKG}.runtime.warp_runtime": _module(
            f"{PKG}.runtime.warp_runtime",
            probe_runtime=lambda force=False: SimpleNamespace(
                warp_version="1.17.0", module_file="", cuda_driver_version="",
                cuda_toolkit_version="", cuda_devices=(_Device(),), message="ok"),
            warp_version_support=lambda _version: ("supported", "")),
        f"{PKG}.blender.cache_bake": _module(
            f"{PKG}.blender.cache_bake", BakeRun=_bake_run),
        f"{PKG}.blender.production_bake": _module(
            f"{PKG}.blender.production_bake",
            BakeStatus=SimpleNamespace(), ProductionBakeBackend=_Backend,
            SignalCancellation=object, ResumeRefused=ResumeRefused),
        f"{PKG}.preferences": _module(
            f"{PKG}.preferences", set_configured_device=lambda _device: None),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(command_line, "_is_legacy_2x_cache", lambda _cache: False)
    monkeypatch.setattr(cache_schema, "load_manifest", lambda _path: object())

    def run(**overrides):
        options = command_line.CliOptions(
            scene=overrides.pop("scene", ""), frame_start=None,
            frame_end=overrides.pop("frame_end", None),
            cache=tmp_path / "cache", device="", resume="auto",
            checkpoint_interval=1, save_after_bake=False,
            report_json=tmp_path / "report.json")
        code = command_line.run(options)
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        return code, report

    run.state = state
    return run


def test_corrupt_manifest_is_a_cache_error(cli, monkeypatch):
    def corrupt(_path):
        raise ValueError("manifest is not valid JSON")

    monkeypatch.setattr(cache_schema, "load_manifest", corrupt)
    code, report = cli()
    assert code == command_line.EXIT_RESUME_INCOMPATIBLE == report["exit_code"]
    assert "manifest" in report["message"]


def test_unverifiable_cache_on_resume_is_a_cache_error(cli):
    def broken(_scene):
        raise ValueError("checkpoint hash mismatch")

    cli.state.resume_decision = broken
    code, report = cli()
    assert code == command_line.EXIT_RESUME_INCOMPATIBLE
    assert "checkpoint hash mismatch" in report["message"]


def test_resume_refused_by_the_bake_is_a_cache_error(cli):
    def refuse():
        raise ResumeRefused("SOURCE_CHANGED: a source animation changed")

    cli.state.bake_run = refuse
    code, report = cli()
    assert code == command_line.EXIT_RESUME_INCOMPATIBLE
    assert "SOURCE_CHANGED" in report["message"]


def test_a_missing_scene_stays_a_scene_error(cli):
    code, report = cli(scene="Nope")
    assert code == command_line.EXIT_SCENE_INVALID
    assert "does not exist" in report["message"]


def test_resume_extends_a_complete_cache_to_a_later_frame_end(cli):
    """Review finding (f): the command line asked for the resume decision
    without the end frame, so a complete cache was refused ALREADY_COMPLETE
    where the GUI's Resume extends it (owner decision D4)."""
    asked = []

    def decide(_scene, **kwargs):
        asked.append(kwargs)
        if kwargs.get("end", 0) > 3:
            return SimpleNamespace(allowed=True, code="RESUME_READY", message="")
        return SimpleNamespace(
            allowed=False, code="ALREADY_COMPLETE", message="complete through 3")

    cli.state.resume_decision = decide
    code, report = cli(frame_end=10)

    assert "ALREADY_COMPLETE" not in report["message"]
    assert code != command_line.EXIT_RESUME_INCOMPATIBLE
    assert asked[-1] == {"end": 10}
    assert cli.state.bake_calls[-1]["resume"] is True
    assert cli.state.bake_calls[-1]["end"] == 10


def test_a_complete_cache_without_a_later_end_is_still_refused(cli):
    cli.state.resume_decision = lambda _scene, **_kwargs: SimpleNamespace(
        allowed=False, code="ALREADY_COMPLETE", message="complete through 3")

    code, report = cli()

    assert code == command_line.EXIT_RESUME_INCOMPATIBLE
    assert "ALREADY_COMPLETE" in report["message"]
    assert cli.state.bake_calls == []
