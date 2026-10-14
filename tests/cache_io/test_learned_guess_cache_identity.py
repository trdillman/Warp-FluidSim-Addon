"""The hidden learned-guess setting leaves cache identity alone while it is off.

``pressure_learned_guess`` (4.0 lane N) changes results when on, so it must
be part of cache identity then. Off (the default, and the only state the UI
can reach) it must not change any signature: every pre-4.0 cache, and every
4.0 cache baked before the field existed, has to keep matching.
"""

from __future__ import annotations

from fluidSim_warp_blender51.io import cache

BASE = {"voxel_size": 0.05, "pressure_tolerance": 1e-4, "device": "cuda:0",
        "pressure_preconditioner": "MULTIGRID"}


def test_off_hashes_like_a_record_without_the_field():
    off = dict(BASE, pressure_learned_guess=False)
    assert cache.settings_record(off)["signature"] == cache.settings_record(BASE)["signature"]
    assert cache.legacy_settings_signatures(off) == cache.legacy_settings_signatures(BASE)
    assert cache.scene_key_from_records(off, ()) == cache.scene_key_from_records(BASE, ())
    assert "pressure_learned_guess" not in cache.settings_record(off)["settings"]


def test_on_is_part_of_cache_identity():
    on = dict(BASE, pressure_learned_guess=True)
    assert cache.settings_record(on)["signature"] != cache.settings_record(BASE)["signature"]
    assert cache.settings_record(on)["settings"]["pressure_learned_guess"] is True
    assert cache.scene_key_from_records(on, ()) != cache.scene_key_from_records(BASE, ())
    assert not cache.settings_signature_matches(on, cache.settings_record(BASE)["signature"])
