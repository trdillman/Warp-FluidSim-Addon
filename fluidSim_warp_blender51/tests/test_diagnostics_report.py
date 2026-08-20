from __future__ import annotations

from fluidsim_testpkg.diagnostics import report
from fluidsim_testpkg.runtime.warp_runtime import DeviceRecord, RuntimeProbe


def _probe() -> RuntimeProbe:
    return RuntimeProbe(
        available=True,
        message="Detected 1 CUDA device(s).",
        warp_version="1.15.0",
        module_file="/home/alice/.config/blender/site-packages/warp/__init__.py",
        cuda_driver_version=12090,
        cuda_toolkit_version=12090,
        cuda_devices=(DeviceRecord("cuda:0", "RTX Test", 8 * 1024**3),),
    )


def test_redact_path_replaces_home_prefix():
    assert report.redact_path("/home/alice/cache/run", home="/home/alice") == "~/cache/run"
    assert report.redact_path(
        r"C:\Users\Alice\cache\run", home=r"C:\Users\Alice"
    ) == "~/cache/run"
    assert report.redact_path(
        "/Users/alice/cache/run", home="/Users/alice"
    ) == "~/cache/run"
    assert report.redact_path("relative/cache", home="/home/alice") == "relative/cache"


def test_support_report_contains_release_runtime_and_scene_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "probe_runtime", lambda: _probe())
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    payload = report.build_report(
        package_name="bl_ext.local.fluidsim_warp",
        blender_version="5.1.2",
        selected_device="cuda:0",
        cache_path=str(cache_path),
        scene_signature="scene-abc",
        settings_signature="settings-def",
    )
    assert payload["fluidsim"]["version"] == "1.0.252"
    assert payload["fluidsim"]["package"] == "bl_ext.local.fluidsim_warp"
    assert payload["host"]["blender"] == "5.1.2"
    assert payload["runtime"]["warp_version"] == "1.15.0"
    assert payload["runtime"]["selected_device"] == "cuda:0"
    assert payload["runtime"]["devices"][0]["memory_bytes"] == 8 * 1024**3
    assert payload["cache"]["writable"] is True
    assert payload["scene"]["scene_signature"] == "scene-abc"
    assert payload["scene"]["settings_signature"] == "settings-def"
    assert "/home/alice" not in str(payload)


def test_support_report_includes_custom_effect_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "probe_runtime", lambda: _probe())
    state = {
        "draft_signature": "a" * 64,
        "tested_signature": "b" * 64,
        "applied_signature": "c" * 64,
        "policy": "BUILTINS_ONLY",
        "inventory": [{"effect_id": "fluidsim.vortex_force"}],
        "items": [{"effect_id": "fluidsim.vortex_force", "trust": "BUILT_IN_TRUSTED"}],
    }
    payload = report.build_report(
        cache_path=str(tmp_path), effect_state=state
    )
    assert payload["custom_effects"] == state
