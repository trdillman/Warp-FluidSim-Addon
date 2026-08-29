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


def test_entrypoint_documents_external_warp_runtime_policy():
    source = " ".join(
        (ROOT / "__init__.py").read_text(encoding="utf-8").split()
    )

    assert "neither probes nor installs runtime dependencies" in source
    assert "certified external Warp runtime" in source
    assert "runtime/warp_runtime.py" in source
    assert "package-declared wheels" not in source


def test_custom_package_root_handles_extension_namespace():
    from fluidsim_testpkg.sim.custom import addon_root_package

    assert addon_root_package("bl_ext.local.fluidsim_warp.sim") == "bl_ext.local.fluidsim_warp"
    assert addon_root_package("fluidsim_warp.sim") == "fluidsim_warp"


def test_custom_package_root_rejects_non_sim_package():
    from fluidsim_testpkg.sim.custom import addon_root_package

    try:
        addon_root_package("fluidsim_warp.runtime")
    except ValueError as exc:
        assert "expected a .sim package" in str(exc)
    else:
        raise AssertionError("non-.sim package was accepted")


def test_render_unregister_removes_only_owned_handler_functions():
    source = (ROOT / "blender" / "render.py").read_text(encoding="utf-8")
    assert "def _remove_all(" in source
    assert "startswith(" not in source
    assert "_remove_all(render_init, _on_render_init)" in source
    assert "_remove_all(handlers, _on_render_end)" in source


def test_snippet_storage_uses_extension_namespace_api():
    source = (ROOT / "properties.py").read_text(encoding="utf-8")
    assert 'bpy.utils.extension_path_user(__package__, path="snippets", create=create)' in source
    assert 'user_resource("CONFIG", path="fluidsim_snippets"' not in source
