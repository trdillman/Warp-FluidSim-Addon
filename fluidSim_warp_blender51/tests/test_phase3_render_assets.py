from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
ASSET = ROOT / "resources" / "geometry_nodes" / "fluidsim_render_nodes.blend"


def test_bundled_geometry_nodes_asset_is_generated_blender_content():
    assert ASSET.is_file()
    assert ASSET.stat().st_size > 50_000
    magic = ASSET.read_bytes()[:7]
    assert magic == b"BLENDER" or magic[:4] == b"\x28\xb5\x2f\xfd"
    assert not list(ASSET.parent.glob("*.blend1"))


def test_asset_generator_preserves_shader_attributes_on_point_instances():
    source = (TOOLS_RELEASE / "generate_geometry_nodes.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "FluidSim Surface Render [P4 v1]",
        "FluidSim Liquid Particles [P4 v1]",
        "FluidSim Whitewater [P4 v1]",
        "FluidSim Foam [P4 v1]",
        "FluidSim Bubbles [P4 v1]",
        "FluidSim Spray [P4 v1]",
    ):
        assert name in source
    assert "GeometryNodeSetMaterial" in source
    assert "GeometryNodeSetPointRadius" in source
    assert "GeometryNodeInstanceOnPoints" in source
    assert "GeometryNodeRealizeInstances" in source
    assert "fluidsim-render-p4-v1" in source
    assert "use_fake_user = True" in source


def test_default_materials_use_simulation_attributes_not_debug_display_color():
    source = (ROOT / "blender" / "materials.py").read_text(encoding="utf-8")
    assert 'SIMULATION_COLOR_ATTRIBUTE = "fluidsim_color"' in source
    assert 'SIMULATION_VISCOSITY_ATTRIBUTE = "fluidsim_viscosity"' in source
    assert 'attr.attribute_name = "display_color"' in source
    assert "Legacy realtime debug material" in source


def test_installer_is_revisioned_idempotent_and_user_safe():
    source = (ROOT / "blender" / "geometry_nodes.py").read_text(encoding="utf-8")
    assert 'ASSET_REVISION = "fluidsim-render-p4-v1"' in source
    assert "modifier.properties.inputs" in source
    assert "is user-owned" in source
    assert "force=bool(force)" in source
    assert "current_revision != ASSET_REVISION" in source


def test_phase3_runtime_asset_is_packaged_and_generators_are_tools_owned():
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    assert "resources/geometry_nodes/fluidsim_render_nodes.blend" not in manifest
    for tool in (
        "generate_geometry_nodes.py",
        "validate_phase3_blender.py",
        "validate_phase3_warp.py",
        "validate_phase4_blender.py",
        "validate_phase4_warp.py",
    ):
        assert (TOOLS_RELEASE / tool).is_file()
        assert f"release/{tool}" not in manifest
