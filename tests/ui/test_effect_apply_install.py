"""Compile & Apply publishes APPLIED only after the live solver installed it.

``apply_stack`` used to wrap the live install in ``except AttributeError``,
so an ``AttributeError`` raised *inside* installation was swallowed and the
stack was still published APPLIED although the solver never installed it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def _setup(monkeypatch, solver):
    _install_bpy_stub(monkeypatch)
    effects = _reload_module("fluidSim_warp_blender51.blender.effects")
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(
        applied_signature="", applied_inventory_json="", applied_snapshot_json=""))
    candidate = SimpleNamespace(success=True, issues=())
    session = SimpleNamespace(
        solver=solver, active=solver is not None, owns_scene=lambda _scene: True)
    published = []
    monkeypatch.setattr(effects, "test_draft", lambda _scene, *, device: candidate)
    monkeypatch.setattr(effects, "scene_effect_instances", lambda _scene: ())
    monkeypatch.setattr(effects, "_publish_applied_state",
                        lambda _scene, instances: published.append(instances))
    monkeypatch.setattr(scene_bridge, "session", lambda: session)
    return effects, scene, candidate, published


def test_attribute_error_inside_live_install_is_not_published_applied(monkeypatch):
    def install(_candidate):
        raise AttributeError("'NoneType' object has no attribute 'launch'")

    effects, scene, _candidate, published = _setup(
        monkeypatch, SimpleNamespace(set_custom_effect_candidate=install))

    with pytest.raises(AttributeError, match="launch"):
        effects.apply_stack(scene, device="cuda:0")
    assert published == []


def test_installed_candidate_is_published_applied(monkeypatch):
    installed = []

    def install(candidate):
        installed.append(candidate)
        return True

    effects, scene, candidate, published = _setup(
        monkeypatch, SimpleNamespace(set_custom_effect_candidate=install))

    assert effects.apply_stack(scene, device="cuda:0") is candidate
    assert installed == [candidate]
    assert published == [()]


def test_solver_without_live_install_still_publishes(monkeypatch):
    effects, scene, candidate, published = _setup(monkeypatch, SimpleNamespace())

    assert effects.apply_stack(scene, device="cuda:0") is candidate
    assert published == [()]


def test_rejected_candidate_is_not_published(monkeypatch):
    effects, scene, _candidate, published = _setup(
        monkeypatch, SimpleNamespace(set_custom_effect_candidate=lambda _c: False))

    with pytest.raises(RuntimeError, match="rejected"):
        effects.apply_stack(scene, device="cuda:0")
    assert published == []
