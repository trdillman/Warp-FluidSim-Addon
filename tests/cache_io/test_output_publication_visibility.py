"""Output visibility contracts for persistent FluidSim outputs.

Render and viewport visibility share one rule, and publication must not
rewrite visibility flags that already hold the wanted value (each RNA write
re-syncs collections and rebuilds depsgraph relations in Blender).
"""
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import output_publication as output


class _Output(dict):
    """Output object stand-in: ID properties plus visibility flags."""

    def __init__(self, name, *, hide_viewport=False, hide_render=False):
        super().__init__()
        self.name = name
        self.type = "POINTCLOUD"
        self.hide_viewport = hide_viewport
        self.hide_render = hide_render


def _scene(*, surface_enabled, whitewater_enabled, mode="COMBINED"):
    domain_settings = SimpleNamespace(
        surface_enabled=surface_enabled,
        whitewater_enabled=whitewater_enabled,
        voxel_size=0.1,
    )
    fs = SimpleNamespace(
        show_surface=True,
        show_particles=True,
        show_whitewater=True,
        whitewater_output_mode=mode,
        domain_object=SimpleNamespace(fluidsim_domain=domain_settings),
    )
    return SimpleNamespace(fluidsim=fs)


def _outputs(monkeypatch):
    objects = {name: _Output(name) for name in (
        output.SURFACE_NAME, output.PARTICLES_NAME, *output.WHITEWATER_OUTPUT_NAMES)}
    monkeypatch.setattr(output, "output_object", lambda _scene, name: objects.get(name))
    return objects


@pytest.mark.parametrize("mode", ["COMBINED", "SEPARATE", "BOTH"])
def test_render_visibility_honours_domain_surface_and_whitewater_switches(
        monkeypatch, mode):
    """B75: outputs the viewport hides (domain switch off) must not render."""
    objects = _outputs(monkeypatch)
    objects[output.WHITEWATER_NAME][output.LEGACY_UNCLASSIFIED_COUNT_TAG] = 5
    output.ensure_render_visibility(
        _scene(surface_enabled=False, whitewater_enabled=False, mode=mode))

    assert objects[output.SURFACE_NAME].hide_render is True
    for name in output.WHITEWATER_OUTPUT_NAMES:
        assert objects[name].hide_render is True, name
    assert objects[output.PARTICLES_NAME].hide_render is False


def test_render_visibility_still_shows_enabled_outputs(monkeypatch):
    objects = _outputs(monkeypatch)
    output.ensure_render_visibility(
        _scene(surface_enabled=True, whitewater_enabled=True, mode="SEPARATE"))

    assert objects[output.SURFACE_NAME].hide_render is False
    assert objects[output.WHITEWATER_NAME].hide_render is True
    for name in (output.FOAM_NAME, output.BUBBLES_NAME, output.SPRAY_NAME):
        assert objects[name].hide_render is False, name
