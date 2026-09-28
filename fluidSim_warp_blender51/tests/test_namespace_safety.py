from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEGACY = "fluidSim_warp_blender51"


def test_no_runtime_file_hardcodes_legacy_package_name():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if any(part in {"tests", "release", ".worktrees"} for part in path.parts):
            continue
        if LEGACY in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_entrypoint_documents_bundled_warp_runtime_policy():
    source = " ".join(
        (ROOT / "__init__.py").read_text(encoding="utf-8").split()
    )

    assert "neither probes nor installs runtime dependencies" in source
    assert "bundled, hash-locked NVIDIA Warp wheel" in source
    assert "runtime/warp_runtime.py" in source


def test_effect_source_package_root_handles_extension_namespace():
    from fluidsim_testpkg.sim.effects.source import addon_root_package

    assert addon_root_package("bl_ext.local.fluidsim_warp.sim.effects") == "bl_ext.local.fluidsim_warp"
    assert addon_root_package("fluidsim_warp.sim.effects") == "fluidsim_warp"


def test_effect_source_package_root_rejects_non_effects_package():
    from fluidsim_testpkg.sim.effects.source import addon_root_package

    try:
        addon_root_package("fluidsim_warp.runtime")
    except ValueError as exc:
        assert "expected a .sim.effects package" in str(exc)
    else:
        raise AssertionError("non-.sim.effects package was accepted")


def test_render_unregister_removes_only_owned_handler_functions():
    source = (ROOT / "blender" / "render.py").read_text(encoding="utf-8")
    assert "def _remove_all(" in source
    assert "startswith(" not in source
    assert "_remove_all(render_init, _on_render_init)" in source
    assert "_remove_all(handlers, _on_render_end)" in source

