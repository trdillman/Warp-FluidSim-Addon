"""Geometry-dependent paused-cache edits bypass timeline cursor suppression."""
import numpy as np
from test_output_fallback_attributes import world  # noqa: F401 - shared fixture


def test_paused_cache_edits_republish_only_when_geometry_changes(world):  # noqa: F811 - fixture
    w = world
    playback = w.bridge.playback
    playback.handle_frame_change(w.bridge.session(), w.scene, None)
    assert not w.bridge.session().active
    assert w.reads == [("test-cache", 7)]
    w.scene.fluidsim.liquid_display_mode = "SURFACE"
    w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
    particles = w.objects[w.output.PARTICLES_NAME]
    assert len(particles.data.vertices) == 6 and not particles.hidden
    assert len(w.reads) == 1  # retained cached geometry: no second disk read

    for mode in ("SEPARATE", "BOTH"):
        w.scene.fluidsim.whitewater_output_mode = mode
        w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
        for name in (w.output.FOAM_NAME, w.output.BUBBLES_NAME, w.output.SPRAY_NAME):
            assert name in w.objects and len(w.objects[name].data.vertices) == 1
        assert w.objects[w.output.WHITEWATER_NAME].hide_viewport == (mode == "SEPARATE")
    w.scene.fluidsim.particle_display_fraction = .5
    w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
    assert len(particles.data.vertices) == 3
    w.scene.fluidsim.particle_attribute_mode = "HEIGHT"
    w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
    np.testing.assert_array_equal(particles.attributes["__visualization__"], [2, 8, 14])
    reads = len(w.reads)
    w.scene.fluidsim.particle_display_scale = 2
    w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
    assert len(w.reads) == reads  # radius/material edits stay metadata-only
    assert not w.bridge.session().active and w.scene.frame_current == 7




def test_corrupt_frame_does_not_consume_a_paused_refresh(world, monkeypatch):  # noqa: F811
    import pytest

    from fluidSim_warp_blender51.io import cache

    w = world
    w.bridge.playback.handle_frame_change(w.bridge.session(), w.scene, None)
    particles = w.objects[w.output.PARTICLES_NAME]
    original = cache.read_frame
    w.scene.fluidsim.particle_display_fraction = .5

    def corrupt(*_args):
        raise ValueError("frame checksum mismatch")

    monkeypatch.setattr(cache, "read_frame", corrupt)
    with pytest.raises(ValueError, match="checksum"):
        w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
    assert len(particles.data.vertices) == 6
    monkeypatch.setattr(cache, "read_frame", original)
    w.pending.request_scene_update(w.scene, w.pending.SceneUpdate.OUTPUT)
    assert len(particles.data.vertices) == 3
    assert not w.bridge.session().active
