"""Contracts for the live demo launcher: every demo is reachable and runs the installed add-on live."""

import ast
import json
import os
from pathlib import Path

import pytest

from tools.demos import live_demo as ld
from tools.demos import nets_showcase as ns
from tools.release import build_extension as be

REPO = Path(__file__).resolve().parents[2]


def test_every_shipped_demo_is_launchable_and_on_disk():
    assert set(ld.SHIPPED) == set(be.DEMO_IDS)
    for demo in ld.SHIPPED:
        assert ld.demo_blend(demo).is_file(), demo


def test_opt_in_demos_are_read_from_where_their_builder_saves():
    builder = (REPO / "tools" / "demos" / "build_demo_scenes.py").read_text(encoding="utf-8")
    assert ld.demo_blend("tsunami_parthenon").parent == REPO / "demos"
    for demo in ("upres_cove", "nets_showcase"):
        assert ld.demo_blend(demo).parent == REPO / "artifacts" / "review_demos"
    # tsunami's builder saves with the default directory (demos/); the others pass REVIEW_DEMOS.
    start = builder.index("def build_tsunami_parthenon")
    end = builder.index("UPRES_COVE_PREVIEW_FRAME")
    assert "REVIEW_DEMOS" not in builder[start:end]


def test_builder_finds_the_provisioned_warp_on_pythonpath(tmp_path, monkeypatch):
    from tools.release import build_extension as be

    seen = []
    monkeypatch.setattr(be, "_provision_smoke_warp", lambda target: seen.append(target))
    env = ld.builder_environment(tmp_path / "b", {"PYTHONPATH": "C:/dev"})
    assert seen == [tmp_path / "b" / "warp-site"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path / "b" / "warp-site")
    assert "C:/dev" in env["PYTHONPATH"]


def test_unknown_demo_is_refused():
    with pytest.raises(ValueError):
        ld.demo_blend("not_a_demo")


def test_blender_starts_normally_so_the_installed_extension_is_on():
    command = ld.blender_command(Path("blender"), Path("x.blend"), Path("boot.py"))
    assert "--factory-startup" not in command
    assert "--background" not in command and "-b" not in command
    assert command[-2:] == ["--python", "boot.py"]


def test_builder_command_lets_blender_see_pythonpath():
    command = ld.build_builder_command(Path("blender"), "upres_cove")
    assert "--python-use-system-env" in command
    assert command.index("--python-use-system-env") < command.index("--python")


def test_nets_showcase_switches_the_surface_backend_on():
    assert ld.DEMO_ENV["nets_showcase"] == {ns.BACKEND_ENV: ns.BACKEND}


def test_environment_is_isolated_and_drops_developer_imports(tmp_path):
    base = {"PYTHONPATH": "C:/dev/warp", "PYTHONHOME": "x", "HOME": "h",
            "PROCESSOR_ARCHITECTURE": "AMD64"}
    env = ld.launch_environment("nets_showcase", tmp_path, tmp_path / "c.json", base=base)
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert Path(env["BLENDER_USER_EXTENSIONS"]).is_relative_to(tmp_path)
    assert Path(env["BLENDER_USER_CONFIG"]).is_relative_to(tmp_path)
    assert env[ns.BACKEND_ENV] == ns.BACKEND
    assert env["FLUIDSIM_LIVE_DEMO_CONFIG"] == str(tmp_path / "c.json")
    plain = ld.launch_environment("dam_break", tmp_path, tmp_path / "c.json", base=base)
    assert ns.BACKEND_ENV not in plain


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    ld.write_config(path, demo="dam_break", report=tmp_path / "r.json", profile=tmp_path,
                    shading="MATERIAL", camera_view=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["demo"] == "dam_break" and data["camera_view"] is True
    assert data["expected_env"] == {}


def test_arguments():
    assert ld.parse_args(["--list"]).list
    with pytest.raises(SystemExit):
        ld.parse_args(["--demo", "dam_break"])  # no --blender
    with pytest.raises(SystemExit):
        ld.parse_args(["--blender", "b", "--demo", "dam_break", "--preset", "draft"])
    args = ld.parse_args(["--blender", "b", "--demo", "nets_showcase", "--preset", "draft"])
    assert args.preset == "draft"


def test_boot_script_forces_live_realtime_playback_with_the_deck_on():
    source = (REPO / "tools" / "demos" / "live_demo_boot.py").read_text(encoding="utf-8")
    ast.parse(source)
    for needle in ("fs.live = True", "cache_playback_enabled = False",
                   "cache_enabled = False", "deck_shown = True", "animation_play",
                   "addon_in_temp_profile", "live_stopped", "os.replace",
                   "SURFACE_NETS_AVAILABLE", "Live Simulation did not start"):
        assert needle in source, needle


def test_boot_script_properties_exist_on_the_add_on():
    props = (REPO / "fluidSim_warp_blender51" / "properties.py").read_text(encoding="utf-8")
    for name in ("live:", "cache_playback_enabled:", "cache_enabled:", "deck_shown:",
                 "stat_particles:", "stat_step_ms:"):
        assert name in props, name
    assert os.path.isfile(ld.BOOT_SCRIPT)
