"""Unit tests for the snippet library: presets, validation, auto-fix,
shortcuts, and user-preset persistence.

Pure-python module tests (no bpy, no warp) plus a small warp-optional
section covering the sim-side frame gating.
"""

import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fluidSim_warp_blender51.sim import snippet_library as lib  # noqa: E402


def _preset_arg_names(preset):
    names = []
    for arg in preset.arguments:
        names.append(arg["field"] if arg["mode"] == "FIELD" else arg["name"])
    return names


# ---------------------------------------------------------------------------
# Example presets
# ---------------------------------------------------------------------------


def test_ten_example_presets_with_requested_categories():
    assert len(lib.EXAMPLE_PRESETS) == 10
    counts = {}
    for preset in lib.EXAMPLE_PRESETS:
        counts[preset.category] = counts.get(preset.category, 0) + 1
    assert counts["Impressive"] == 2
    assert counts["Creative"] == 2
    assert counts["Learning"] == 2
    assert sum(counts.values()) == 10


def test_preset_ids_unique_and_resolvable():
    ids = [p.preset_id for p in lib.EXAMPLE_PRESETS]
    assert len(ids) == len(set(ids))
    for pid in ids:
        assert lib.preset_by_id(pid) is not None
    assert lib.preset_by_id("nope") is None


def test_every_example_preset_passes_validation():
    for preset in lib.EXAMPLE_PRESETS:
        issues = lib.validate_snippet(preset.code, preset.run_over,
                                      _preset_arg_names(preset))
        errors = [i for i in issues if i.level == "ERROR"]
        assert not errors, f"{preset.preset_id}: {errors}"


def test_preset_fields_and_enums_are_known():
    orders = {"FIRST_STEP", "BEFORE_SOURCING", "BEFORE_FORCES",
              "BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION", "FINAL_STEP"}
    run_overs = {"VOXEL_TILE", "VOXEL_LINEAR", "PARTICLES"}
    for preset in lib.EXAMPLE_PRESETS:
        assert preset.order in orders
        assert preset.run_over in run_overs
        for arg in preset.arguments:
            if arg["mode"] == "FIELD":
                assert arg["field"] in lib.KNOWN_FIELDS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_snippet_is_invalid():
    assert not lib.is_valid("")
    assert not lib.is_valid("   \n  ")


def test_syntax_error_reported_with_line():
    issues = lib.validate_snippet("x = 1\ny = = 2", "PARTICLES")
    assert any(i.level == "ERROR" and "syntax" in i.message for i in issues)


def test_unknown_name_rejected_with_suggestion():
    issues = lib.validate_snippet("x = velocty", "PARTICLES")
    err = [i for i in issues if i.level == "ERROR"][0]
    assert "velocty" in err.message
    assert "velocity" in err.message  # suggestion


def test_known_field_requires_declared_argument():
    issues = lib.validate_snippet(
        "temperature[flat_voxel_index] = 1.0", "VOXEL_LINEAR")
    err = [i for i in issues if i.level == "ERROR"][0]
    assert "FIELD argument" in err.message
    # Declaring the argument makes it valid.
    assert lib.is_valid("temperature[flat_voxel_index] = 1.0",
                        "VOXEL_LINEAR", ("temperature",))


def test_constants_count_as_arguments():
    assert lib.is_valid("x = strength * 2.0", "PARTICLES", ("strength",))
    assert not lib.is_valid("x = strength * 2.0", "PARTICLES")


def test_imports_and_defs_rejected():
    assert not lib.is_valid("import os", "PARTICLES")
    assert not lib.is_valid("from os import path", "PARTICLES")
    assert not lib.is_valid("def f():\n    pass", "PARTICLES")


def test_vector_attribute_access_rejected():
    issues = lib.validate_snippet("x = position.x", "PARTICLES")
    assert any("Auto Fix" in i.message for i in issues
               if i.level == "ERROR")
    # wp.* and C.* attribute access stays allowed.
    assert lib.is_valid("x = wp.length(velocity)", "PARTICLES")
    assert lib.is_valid("v = C.LEAF_VOLUME", "VOXEL_LINEAR")


def test_particle_copy_write_warns():
    issues = lib.validate_snippet("velocity = velocity * 0.5", "PARTICLES")
    assert any(i.level == "WARNING" and "particle_velocity" in i.message
               for i in issues)


def test_voxel_context_not_leaked_into_particles():
    assert lib.is_valid("x = world_position", "VOXEL_LINEAR")
    assert not lib.is_valid("x = world_position", "PARTICLES")
    assert lib.is_valid("x = position", "PARTICLES")
    assert not lib.is_valid("x = position", "VOXEL_LINEAR")


def test_assigned_locals_are_usable():
    assert lib.is_valid("a = 1.0\nb = a * 2.0", "PARTICLES")


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def test_autofix_c_style_habits():
    code = "float_speed = wp.length(velocity);\nif true && 1.0f > 0.5f:\n    x = 1.0;"
    fixed, fixes = lib.autofix_snippet(code)
    assert ";" not in fixed
    assert "&&" not in fixed
    assert "True" in fixed
    assert "1.0f" not in fixed
    assert fixes


def test_autofix_vector_components_and_comments():
    code = "// push up\nv = velocity.x + position.z"
    fixed, _ = lib.autofix_snippet(code)
    assert "# push up" in fixed
    assert "velocity[0]" in fixed
    assert "position[2]" in fixed
    assert lib.is_valid(fixed, "PARTICLES")


def test_autofix_math_and_numpy_to_wp():
    fixed, _ = lib.autofix_snippet("x = math.sin(time) + np.cos(time)")
    assert "wp.sin" in fixed and "wp.cos" in fixed


def test_autofix_smart_quotes_and_tabs():
    fixed, fixes = lib.autofix_snippet("\tx = 1.0 # \u201cnote\u201d")
    assert "\t" not in fixed
    assert "\u201c" not in fixed


def test_autofix_empty_becomes_pass():
    fixed, _ = lib.autofix_snippet("")
    assert fixed == "pass"


def test_autofix_preserves_valid_code():
    code = "speed = wp.length(velocity)\nif speed > 2.0:\n    x = speed"
    fixed, fixes = lib.autofix_snippet(code)
    assert fixed == code
    assert fixes == []


# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------


def test_shortcut_ids_unique_and_resolvable():
    ids = [s.shortcut_id for s in lib.SHORTCUTS]
    assert len(ids) == len(set(ids))
    assert lib.shortcut_by_id(ids[0]) is not None
    assert lib.shortcut_by_id("nope") is None


def test_shortcuts_filtered_by_domain():
    particle = {s.shortcut_id for s in lib.shortcuts_for("PARTICLES")}
    voxel = {s.shortcut_id for s in lib.shortcuts_for("VOXEL_LINEAR")}
    assert "ctx_particle_speed" in particle
    assert "ctx_particle_speed" not in voxel
    assert "ctx_world_pos" in voxel
    assert "ctx_world_pos" not in particle
    assert "time_dt" in particle and "time_dt" in voxel


def test_every_shortcut_code_validates_in_its_domain():
    for shortcut in lib.SHORTCUTS:
        run_overs = (("PARTICLES", "VOXEL_LINEAR")
                     if shortcut.run_over == "ANY"
                     else (("PARTICLES",) if shortcut.run_over == "PARTICLES"
                           else ("VOXEL_LINEAR",)))
        for run_over in run_overs:
            issues = lib.validate_snippet(shortcut.code, run_over)
            errors = [i for i in issues if i.level == "ERROR"]
            assert not errors, f"{shortcut.shortcut_id}/{run_over}: {errors}"


# ---------------------------------------------------------------------------
# User preset persistence
# ---------------------------------------------------------------------------


def test_user_preset_roundtrip(tmp_path):
    data = lib.preset_to_dict(
        "My Swirl", "test", "BEFORE_FORCES", "PARTICLES",
        "x = wp.length(velocity)",
        [{"mode": "CONSTANT", "name": "strength", "type": "FLOAT",
          "value": 2.0}],
        frame_start=10, frame_end=50, frame_step=2)
    path = lib.save_user_preset(str(tmp_path), data)
    assert os.path.isfile(path)
    loaded = lib.load_user_presets(str(tmp_path))
    assert len(loaded) == 1
    got = loaded[0]
    assert got["label"] == "My Swirl"
    assert got["order"] == "BEFORE_FORCES"
    assert got["frame_start"] == 10 and got["frame_end"] == 50
    assert got["arguments"][0]["name"] == "strength"
    assert got["_path"] == path


def test_load_user_presets_skips_garbage(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "nocode.json").write_text("{\"label\": \"x\"}",
                                          encoding="utf-8")
    (tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")
    assert lib.load_user_presets(str(tmp_path)) == []
    assert lib.load_user_presets(str(tmp_path / "missing")) == []


def test_safe_filename():
    path_a = lib.save_user_preset.__module__  # silence linters
    assert lib._safe_filename("My / Cool * Snip") == "My_Cool_Snip.json"
    assert lib._safe_filename("***") == "snippet.json"


def test_code_hash_stability():
    assert lib.code_hash("abc") == lib.code_hash("abc")
    assert lib.code_hash("abc") != lib.code_hash("abd")


# ---------------------------------------------------------------------------
# Sim-side integration (needs warp import; skipped where unavailable)
# ---------------------------------------------------------------------------


def test_frame_window_gating():
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.custom import CustomSnippet

    sn = CustomSnippet(frame_start=10, frame_end=20, frame_step=1)
    assert not sn.active_on_frame(9)
    assert sn.active_on_frame(10)
    assert sn.active_on_frame(20)
    assert not sn.active_on_frame(21)

    every_4 = CustomSnippet(frame_step=4)
    assert every_4.active_on_frame(0)
    assert not every_4.active_on_frame(1)
    assert every_4.active_on_frame(8)

    windowed_step = CustomSnippet(frame_start=5, frame_step=3)
    assert windowed_step.active_on_frame(5)
    assert not windowed_step.active_on_frame(6)
    assert windowed_step.active_on_frame(8)

    open_ended = CustomSnippet()
    assert open_ended.active_on_frame(1)
    assert open_ended.active_on_frame(100000)


def test_snippet_manager_rejects_invalid_before_compile():
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.custom import (
        CustomArgument, CustomSnippet, SnippetManager,
    )
    from fluidSim_warp_blender51.sim.enums import (
        CustomKernelArgumentMode, CustomKernelOrder, CustomKernelRunOver,
    )

    manager = SnippetManager()
    bad = CustomSnippet(
        kernel_name="bad_import", code="import os",
        order=CustomKernelOrder.FINAL_STEP,
        run_over=CustomKernelRunOver.PARTICLES)
    good = CustomSnippet(
        kernel_name="good_damp",
        code="particle_velocity[particle_index] = velocity * 0.5",
        order=CustomKernelOrder.FINAL_STEP,
        run_over=CustomKernelRunOver.PARTICLES,
        arguments=[CustomArgument(
            mode=CustomKernelArgumentMode.FIELD, field="particle_velocity")])
    manager.set_snippets([bad, good])
    assert any(name == "bad_import" for name, _ in manager.errors)
    assert manager.has_any()


def _fake_solver_for_snippet_launch():
    particles = SimpleNamespace(
        capacity=1000,
        position=object(),
        velocity=object(),
        alive=object(),
        count=object(),
    )
    grid = SimpleNamespace(
        voxel_capacity=2000,
        voxel_size=0.1,
        leaf_coords=object(),
        num_leaves=object(),
        origin=(0.0, 0.0, 0.0),
    )
    return SimpleNamespace(
        particles=particles,
        grid=grid,
        voxel_type=SimpleNamespace(data=object()),
        device="cuda:0",
        _nvox=23,
        _np_dim=lambda: 17,
    )


def _compiled_launch_probe(run_over):
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.custom import (
        CompiledSnippet, CustomSnippet,
    )
    from fluidSim_warp_blender51.sim.enums import CustomKernelRunOver

    compiled = object.__new__(CompiledSnippet)
    compiled.kernel = object()
    compiled.error = None
    compiled._binders = []
    compiled.snippet = CustomSnippet(run_over=CustomKernelRunOver[run_over])
    return compiled


def test_particle_snippet_launch_uses_active_particle_dimension(monkeypatch):
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim import custom

    seen = {}

    def fake_launch(_kernel, *, dim, inputs, device):
        seen["dim"] = dim
        seen["device"] = device

    monkeypatch.setattr(custom.wp, "launch", fake_launch)
    compiled = _compiled_launch_probe("PARTICLES")
    compiled.launch(
        _fake_solver_for_snippet_launch(),
        {"frame": 1, "time": 0.0, "real_time_step": 1 / 24,
         "solver_time_step": 1 / 24, "time_scale": 1.0, "substeps": 1},
    )

    assert seen == {"dim": 17, "device": "cuda:0"}


def test_voxel_snippet_launch_uses_active_voxel_dimension(monkeypatch):
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim import custom

    seen = {}

    def fake_launch(_kernel, *, dim, inputs, device):
        seen["dim"] = dim
        seen["device"] = device

    monkeypatch.setattr(custom.wp, "launch", fake_launch)
    compiled = _compiled_launch_probe("VOXEL_LINEAR")
    compiled.launch(
        _fake_solver_for_snippet_launch(),
        {"frame": 1, "time": 0.0, "real_time_step": 1 / 24,
         "solver_time_step": 1 / 24, "time_scale": 1.0, "substeps": 1},
    )

    assert seen == {"dim": 23, "device": "cuda:0"}


def test_all_presets_compile_with_warp():
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.custom import (
        CompiledSnippet, CustomArgument, CustomSnippet,
    )
    from fluidSim_warp_blender51.sim.enums import (
        CustomKernelArgumentMode, CustomKernelConstantType,
        CustomKernelOrder, CustomKernelRunOver,
    )

    for preset in lib.EXAMPLE_PRESETS:
        args = []
        for arg in preset.arguments:
            if arg["mode"] == "FIELD":
                args.append(CustomArgument(
                    mode=CustomKernelArgumentMode.FIELD, field=arg["field"]))
            else:
                ctype = (CustomKernelConstantType.INT
                         if arg.get("type") == "INT"
                         else CustomKernelConstantType.FLOAT)
                args.append(CustomArgument(
                    mode=CustomKernelArgumentMode.CONSTANT,
                    constant_type=ctype,
                    constant_name=arg["name"],
                    constant_value=float(arg["value"])))
        sn = CustomSnippet(
            kernel_name=f"preset_{preset.preset_id}",
            code=preset.code,
            order=CustomKernelOrder[preset.order],
            run_over=CustomKernelRunOver[preset.run_over],
            arguments=args)
        compiled = CompiledSnippet(sn)
        assert compiled.error is None, (
            f"{preset.preset_id}: {compiled.error}")
