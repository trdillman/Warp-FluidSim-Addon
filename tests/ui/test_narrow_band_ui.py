"""Setup stage: the Narrow-Band toggle, its band field and its greyed reasons."""

from __future__ import annotations

from test_operator_panel_behavior import (
    _draw_stage,
    _install_bpy_stub,
    _reload_module,
    _stage_context,
)

from fluidSim_warp_blender51.sim import narrow_band


def _setup_layout(monkeypatch, objects=(), **domain_overrides):
    _install_bpy_stub(monkeypatch)
    ui_stages = _reload_module("fluidSim_warp_blender51.blender.ui_stages")
    values = dict(
        narrow_band_enabled=False, narrow_band_voxels=3.0, species_enabled=False)
    values.update(domain_overrides)
    context, _scene, domain = _stage_context(**values)
    context.scene.objects = list(objects)
    return _draw_stage(ui_stages.FLUIDSIM_STAGE_PT_setup, context), domain


def _label_texts(layout):
    return [label["text"] for label in layout.labels]


def test_toggle_is_drawn_on_the_domain_and_the_band_field_is_hidden_while_off(
        monkeypatch):
    layout, domain = _setup_layout(monkeypatch)

    assert "narrow_band_enabled" in layout.props
    assert "narrow_band_voxels" not in layout.props
    drawn = {call["name"]: call["data"] for call in layout.prop_calls}
    assert drawn["narrow_band_enabled"] is domain
    assert layout.drawn_enabled("narrow_band_enabled") is True
    assert not set(_label_texts(layout)) & {
        narrow_band.REASON_COLOR_MIXING, narrow_band.REASON_EXPORT_ATTRIBUTES}


def test_band_field_appears_once_enabled(monkeypatch):
    layout, domain = _setup_layout(monkeypatch, narrow_band_enabled=True)

    assert "narrow_band_voxels" in layout.props
    drawn = {call["name"]: call["data"] for call in layout.prop_calls}
    assert drawn["narrow_band_voxels"] is domain
    assert layout.drawn_enabled("narrow_band_enabled") is True


def test_blocked_toggle_is_greyed_and_lists_each_reason(monkeypatch):
    layout, _domain = _setup_layout(
        monkeypatch, species_enabled=True, export_attributes=True)

    assert layout.drawn_enabled("narrow_band_enabled") is False
    assert "narrow_band_voxels" not in layout.props
    texts = _label_texts(layout)
    assert narrow_band.REASON_COLOR_MIXING in texts
    assert narrow_band.REASON_EXPORT_ATTRIBUTES in texts


def test_scalar_field_source_object_greys_the_toggle(monkeypatch):
    from types import SimpleNamespace

    field = SimpleNamespace(
        fluidsim=SimpleNamespace(role="FIELD", field_operation="SCALAR"))
    layout, _domain = _setup_layout(monkeypatch, objects=[field])

    assert layout.drawn_enabled("narrow_band_enabled") is False
    assert narrow_band.REASON_SCALAR_FIELD_SOURCE in _label_texts(layout)


def test_an_already_enabled_blocked_toggle_stays_usable_so_it_can_be_turned_off(
        monkeypatch):
    layout, _domain = _setup_layout(
        monkeypatch, narrow_band_enabled=True, species_enabled=True)

    assert layout.drawn_enabled("narrow_band_enabled") is True
    assert narrow_band.REASON_COLOR_MIXING in _label_texts(layout)
