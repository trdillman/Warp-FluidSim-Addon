from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_material_response_demo_is_a_rebuildable_250_frame_showcase():
    builder = (ROOT / "tools" / "build_demo_scenes.py").read_text()

    assert "def build_material_response():" in builder
    assert '"material_response": build_material_response' in builder
    assert "scene.frame_end = 250" in builder
    assert 'run_scene("material_response", 250)' in builder
    assert 'rotation=(0.0, math.radians(11.0), 0.0)' in builder
    assert "fs.live = True" in builder
    assert "max_particles = 400_000" in builder
    assert 'source.fluidsim.initial_velocity = (5.5, 0.0, -0.15)' in builder
    assert 'f"{label} Impact Wall F{friction:.2f} R{restitution:.2f}"' in builder


def test_material_response_demo_covers_four_distinct_combinations():
    builder = (ROOT / "tools" / "build_demo_scenes.py").read_text()

    for values in (
        '0.0, 0.0, "SLIDE"',
        '0.9, 0.0, "GRIP"',
        '0.0, 0.9, "BOUNCE"',
        '0.75, 0.75, "GRIP + BOUNCE"',
    ):
        assert values in builder


def test_saved_demo_verifier_requires_particle_scale_and_material_colliders():
    verifier = (ROOT / "tools" / "check_demo_blends.py").read_text()

    assert 'if name == "material_response":' in verifier
    assert '"minimum_particle_scale"' in verifier
    assert '"four_material_combinations"' in verifier
    assert '"generated_through_frame_250"' in verifier
    assert '"non_overlapping_sources"' in verifier
    assert '"emitters_over_high_end"' in verifier
    assert '"timeline_live_enabled"' in verifier


def test_material_response_preview_has_visible_value_labels():
    builder = (ROOT / "tools" / "build_demo_scenes.py").read_text()

    assert 'f"{label}\\nF {friction:.2f}  R {restitution:.2f}"' in builder
    assert '"Material Response Legend"' in builder
