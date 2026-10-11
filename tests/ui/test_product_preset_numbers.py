"""The preset enum keeps the numbers older .blend files saved for each preset.

Blender stores a dynamic enum as an integer. Before the "Choose a preset…"
placeholder existed, a preset's number was its position in the discovered list,
so presets must keep 0..n-1 and the placeholder must sit outside that range.
"""

from __future__ import annotations

from test_artist_mode_callback_ownership import _load_properties


def test_presets_keep_their_list_position_and_the_placeholder_is_the_default(
        monkeypatch, tmp_path):
    fake_bpy, properties = _load_properties(monkeypatch)
    fake_bpy.utils.user_resource = lambda _kind: str(tmp_path)
    from fluidSim_warp_blender51.presets.library import discover_presets

    items = properties._product_preset_items(None, None)
    placeholder, presets = items[0], items[1:]

    assert placeholder[0] == properties.PRODUCT_PRESET_NONE
    assert placeholder[4] == properties.PRODUCT_PRESET_NONE_NUMBER
    assert [(item[0], item[4]) for item in presets] == [
        (f"{record.source}:{record.preset.id}", number)
        for number, record in enumerate(discover_presets())
    ]
    assert properties.PRODUCT_PRESET_NONE_NUMBER >= len(presets)


def test_scan_failure_fallback_keeps_preview_water_at_its_old_number(monkeypatch):
    _fake_bpy, properties = _load_properties(monkeypatch)
    from fluidSim_warp_blender51.presets import library

    def broken():
        raise OSError("unreadable preset folder")

    monkeypatch.setattr(library, "discover_presets", broken)
    items = properties._product_preset_items(None, None)

    assert [(item[0], item[4]) for item in items] == [
        (properties.PRODUCT_PRESET_NONE, properties.PRODUCT_PRESET_NONE_NUMBER),
        ("BUILTIN:preview_water", 0),
    ]
