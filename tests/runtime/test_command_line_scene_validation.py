"""Headless bake scene validation accepts every liquid input the solver uses.

Regression for review item B17: seed-driven demos (crown_splash, paddle_stir)
initialize liquid only from role-NONE objects with ``auto_inject_seed``; the
command line rejected them with exit 13 although the GUI path seeds them.
"""
from __future__ import annotations

from types import SimpleNamespace

from fluidSim_warp_blender51.blender.command_line import _scene_validation


def _obj(role="NONE", enabled=True, auto_inject_seed=False):
    return SimpleNamespace(fluidsim=SimpleNamespace(
        role=role, enabled=enabled, auto_inject_seed=auto_inject_seed))


def _scene(*objects, height_enabled=False):
    domain = SimpleNamespace(fluidsim_domain=SimpleNamespace(height_enabled=height_enabled))
    return SimpleNamespace(
        fluidsim=SimpleNamespace(domain_object=domain),
        objects=[domain, *objects],
    )


def test_seed_only_scene_is_valid():
    valid, message = _scene_validation(_scene(_obj(auto_inject_seed=True)))
    assert valid, message


def test_disabled_seed_does_not_count_as_liquid():
    valid, _ = _scene_validation(_scene(_obj(enabled=False, auto_inject_seed=True)))
    assert not valid


def test_source_and_height_scenes_stay_valid_and_empty_scene_is_refused():
    assert _scene_validation(_scene(_obj(role="SOURCE")))[0]
    assert _scene_validation(_scene(height_enabled=True))[0]
    assert not _scene_validation(_scene(_obj()))[0]
