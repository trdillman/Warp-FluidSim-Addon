from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from fluidsim_testpkg.blender import solver_settings
from fluidsim_testpkg.presets import library
from fluidsim_testpkg.sim.surface_tension_math import capillary_dt

ROOT = Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"


def _assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _class_annotations(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }
    raise AssertionError(f"{class_name} not found in {path}")


def test_phase5_solver_settings_and_projection_are_declared():
    fields = {
        "surface_tension_enabled",
        "surface_tension_coefficient",
        "surface_tension_density",
        "surface_tension_band_voxels",
        "surface_tension_safety",
        "surface_tension_smoothing_iterations",
    }
    assert fields <= _class_annotations(ROOT / "sim" / "solver_settings_model.py", "SolverSettings")
    assert fields <= set(solver_settings.DIRECT_SETTING_FIELDS)
    tree = ast.parse((ROOT / "blender" / "pending_settings.py").read_text(encoding="utf-8"))
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "_LIVE_REGISTRY_FIELDS", "_RESET_REGISTRY_FIELDS"
            }:
                assert isinstance(node.value, ast.Call) and node.value.args
                assignments[target.id] = set(ast.literal_eval(node.value.args[0]))
    assert "surface_tension_enabled" in assignments["_RESET_REGISTRY_FIELDS"]
    assert fields - {"surface_tension_enabled"} <= assignments["_LIVE_REGISTRY_FIELDS"]


def test_phase5_presets_are_strict_and_physically_labelled(tmp_path):
    physical = library.load_preset(
        "BUILTIN:water_surface_tension", directory=tmp_path)
    stylized = library.load_preset(
        "BUILTIN:stylized_high_tension", directory=tmp_path)
    assert physical.settings["surface_tension_enabled"] is True
    assert physical.settings["surface_tension_coefficient"] == 0.072
    assert physical.settings["surface_tension_density"] == 1000.0
    assert "N/m" in physical.description
    assert stylized.settings["surface_tension_coefficient"] > physical.settings[
        "surface_tension_coefficient"
    ]
    assert "nonphysical" in stylized.description
    assert physical.source_version == stylized.source_version == "1.0.250"


def test_frame_stats_source_guards_json_against_infinity():
    source = (ROOT / "sim" / "frame_runtime.py").read_text(encoding="utf-8")
    assert '"capillaryDt": (' in source
    assert "math.isfinite(self.capillary_dt)" in source
    # The pure default is still representable in strict JSON after the guard.
    payload = {"capillaryDt": 0.0, "capillarySubstepsRequired": 1}
    assert math.isfinite(payload["capillaryDt"])
    json.dumps(payload, allow_nan=False)


def test_validation_inventory_includes_all_five_phase5_gates():
    names = _assignment_literal(ROOT / "sim" / "validation.py", "VALIDATION_TESTS")
    assert len(names) == 19
    phase5 = names[-5:]
    assert all("surface-tension" in name for name in phase5)
    assert any("static droplet" in name for name in phase5)
    assert any("zero-coefficient" in name for name in phase5)
    assert any("capillary timestep" in name for name in phase5)


def test_capillary_constraint_uses_authored_units_and_safety():
    expected = 0.5 * math.sqrt(1000.0 * 0.05**3 / (4.0 * math.pi * 0.072))
    assert capillary_dt(1000.0, 0.05, 0.072, 0.5) == expected


def test_capillary_limit_is_not_skipped_before_first_source_emission():
    tree = ast.parse((ROOT / "sim" / "frame_runtime.py").read_text(
        encoding="utf-8"))
    selector = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "select_substeps"
    )
    particle_branch = next(
        node
        for node in selector.body
        if isinstance(node, ast.If)
        and "has_particles" in ast.unparse(node.test)
    )
    calls_inside_particle_branch = {
        node.func.id
        for node in ast.walk(particle_branch)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    calls_in_method = {
        node.func.id
        for node in ast.walk(selector)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "capillary_dt" in calls_in_method
    assert "capillary_dt" not in calls_inside_particle_branch


def test_zero_time_scale_particle_sanitizer_uses_a_safe_denominator():
    source = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    method_start = source.index("def _sanitize_particles")
    method_end = source.index("def _g2p_advect", method_start)
    method = source[method_start:method_end]
    assert "safe_dt = max(float(dt), 1.0e-8)" in method
    assert "/ safe_dt" in method


def test_release_workflow_and_package_include_phase5_gates():
    local_gate = (TOOLS_RELEASE / "run_local_gate.py").read_text(encoding="utf-8")
    gate = (ROOT.parent / "tools/run_production_gate.py").read_text(encoding="utf-8")
    manifest = (ROOT / "blender_manifest.toml").read_text(encoding="utf-8")
    build = (TOOLS_RELEASE / "build_extension.py").read_text(encoding="utf-8")
    smoke = (TOOLS_RELEASE / "smoke_test_blender.py").read_text(encoding="utf-8")
    # The local gate runs Warp CPU; the production gate runs CUDA.
    assert '"phase5"' in local_gate
    assert '"--device", "cpu"' in local_gate
    assert '"phase5"' in gate[gate.index("RELEASE_WARP_VALIDATORS = "):].split("\n", 1)[0]
    assert '"--device", "cuda:0"' in gate
    assert (TOOLS_RELEASE / "validate_phase5_blender.py").is_file()
    assert "validate_phase5_blender.py" not in manifest
    assert '"sim/surface_tension.py"' in build
    assert '"sim/surface_tension_math.py"' in build
    assert "water_surface_tension.json" in smoke


def test_surface_tension_invalidates_pre_advection_sdf_before_whitewater():
    solver_source = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    surface_source = (ROOT / "sim" / "surface.py").read_text(encoding="utf-8")
    assert "self.surface.invalidate_solver_sdf()" in solver_source
    assert "def invalidate_solver_sdf" in surface_source


def test_source_emission_invalidates_same_key_surface_sdf_before_capillary_force():
    source = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    emission = source.index("if emitted:")
    force = source.index('with self._phase("p2g"):', emission)
    block = source[emission:force]
    assert "self.surface.invalidate_solver_sdf()" in block


def test_direct_particle_injection_invalidates_surface_and_render_products():
    execution = (ROOT / "sim" / "source_execution.py").read_text(encoding="utf-8")
    solver = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    surface = (ROOT / "sim" / "surface.py").read_text(encoding="utf-8")
    assert "self.runtime.invalidate_particle_surface()" in execution
    assert "self.surface.invalidate_particle_state()" in solver
    assert "def invalidate_particle_state" in surface
    method = surface[surface.index("def invalidate_particle_state") :]
    method_end = method.find("\n    def ", 1)
    body = method if method_end < 0 else method[:method_end]
    assert "self.built_key = None" in body
    assert "self.solver_sdf_key = None" in body
    assert "self.invalidate()" in body


def test_optional_scratch_accounting_is_reversible():
    source = (ROOT / "sim" / "surface_tension.py").read_text(encoding="utf-8")
    assert "profiler.tally_buffer(delta)" in source
    assert "self._release_resources()" in source
    disabled_branch = source[source.index("if not self._enabled():") :]
    assert "self._release_resources()" in disabled_branch.split("return False", 1)[0]


def test_phase3_warp_regression_is_not_bound_to_an_old_checkout_name():
    source = (TOOLS_RELEASE / "validate_phase3_warp.py").read_text(
        encoding="utf-8"
    )
    assert "fluidsim_phase3_work.sim" not in source
    assert 'parser.add_argument("--module"' in source
    assert 'importlib.import_module(f"{module_name}.sim.' in source


def test_solver_sdf_expands_for_surface_tension_support():
    source = (ROOT / "sim" / "surface.py").read_text(encoding="utf-8")
    method_start = source.index("def build_sdf")
    method_end = source.index("def rebuild_topology", method_start)
    method = source[method_start:method_end]
    assert "required_sdf_splat_radius_voxels" in method
    assert "normal_support_voxels" in method
    assert "sdf_filter_iterations" in method
    assert "far_distance = max" in method


def test_frame_topology_expands_for_the_capillary_sdf_support():
    source = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    assert "def _surface_tension_topology_expand" in source
    assert "capillary_expand = self._surface_tension_topology_expand()" in source
    frame_start = source.index("frame_topology_expand = max")
    frame_end = source.index("self.whitewater.reset_frame_diagnostics", frame_start)
    block = source[frame_start:frame_end]
    # Both the frame-level and the per-substep expansion carry the halo (B37).
    assert "capillary_expand," in block
    assert "substep_topology_expand = max(int(s.topology_expand), capillary_expand)" in block


def test_solver_capillary_halo_respects_authored_surface_splat_radius():
    source = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    method_start = source.index("def _surface_tension_topology_expand")
    method_end = source.index("def _whitewater_topology_expand", method_start)
    method = source[method_start:method_end]
    assert "int(s.surface_splat_radius)" in method
    assert "splat_radius = max(" in method


def test_phase5_blender_validator_tracks_native_properties_panel_identity():
    source = (TOOLS_RELEASE / "validate_phase5_blender.py").read_text(
        encoding="utf-8"
    )
    # The Scene tab registers no domain panels, so the validator must require the
    # Scene id to be absent and the Physics mirror to be the registered home.
    assert '"FLUIDSIM_PT_sim_surface_tension", None) is not None' in source
    assert '"FLUIDSIM_PT_physics_domain_sim_surface_tension"' in source
    assert '"FLUIDSIM_PT_surface_tension"' not in source
    assert 'bl_space_type != "PROPERTIES"' in source
    assert 'bl_context != "physics"' in source
