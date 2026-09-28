"""Host-side checks for tools/release/install_local.py (no Blender needed)."""

from __future__ import annotations

import json

from tools.release import install_local


def _report(**overrides):
    report = {
        "modules": ["bl_ext.user_default.fluidsim_warp"],
        "fluidsim_version": install_local.release_identity()["VERSION_STRING"],
        "runtime_available": True,
        "runtime_message": "ok",
        "warp_path": "site-packages/warp/__init__.py",
    }
    report.update(overrides)
    return report


def test_default_zip_matches_build_extension_naming():
    identity = install_local.release_identity()
    assert install_local.default_zip().name == (
        f"{identity['EXTENSION_ID']}-{identity['VERSION_STRING']}-windows_x64.zip"
    )


def test_parse_probe_finds_marker_line_amid_blender_noise():
    output = "Blender 5.2\nRead prefs\n" + install_local.PROBE_MARKER + json.dumps({"a": 1}) + "\nquit\n"
    assert install_local.parse_probe(output) == {"a": 1}
    assert install_local.parse_probe("no marker here") is None


def test_ready_report_has_no_problems():
    expected = install_local.release_identity()["VERSION_STRING"]
    assert install_local.check_probe(_report(), expected) == []


def test_stale_version_and_foreign_warp_are_reported():
    expected = install_local.release_identity()["VERSION_STRING"]
    problems = install_local.check_probe(
        _report(fluidsim_version="1.0.255", runtime_available=False,
                warp_path="scripts/warp_foundry_runtime/warp/__init__.py"),
        expected,
    )
    assert any("1.0.255" in p for p in problems)
    assert any("warp_foundry_runtime" in p for p in problems)


def test_missing_or_duplicate_extension_is_not_ready():
    expected = install_local.release_identity()["VERSION_STRING"]
    assert install_local.check_probe(None, expected)
    assert install_local.check_probe(_report(modules=[]), expected)
    assert install_local.check_probe(_report(modules=["a.fluidsim_warp", "b.fluidsim_warp"]), expected)


def test_legacy_copies_found_in_addons_and_extensions(tmp_path):
    (tmp_path / "5.1" / "scripts" / "addons" / "fluidSim_warp_blender51").mkdir(parents=True)
    (tmp_path / "5.1" / "extensions" / "user_default" / "fluidSim_warp_blender51").mkdir(parents=True)
    # The current extension id is not legacy and must not be flagged.
    (tmp_path / "5.2" / "extensions" / "user_default" / "fluidsim_warp").mkdir(parents=True)
    found = install_local.legacy_copies(tmp_path)
    assert sorted(p.relative_to(tmp_path).parts[1] for p in found) == ["extensions", "scripts"]
