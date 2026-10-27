"""Bounded CUDA allocation proof; run through an isolated Blender launcher.

This measures array payload separately from CUDA pool usage/free-memory samples.
It does not advance a frame or qualify simulation output/performance.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import bpy
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "support"))
from _test_device import script_device
from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings


def inventory(*owners):
    """Inspect real reachable Warp arrays; aliases count once by device pointer."""
    arrays, seen = {}, set()

    def visit(value, path):
        if isinstance(value, wp.array):
            if value.ptr:
                key = (str(value.device), int(value.ptr))
                arrays.setdefault(key, {
                    "path": path, "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "payload_bytes": int(value.size * wp.types.type_size_in_bytes(value.dtype)),
                })
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, (tuple, list)):
            items = enumerate(value)
        elif type(value).__module__.startswith("fluidSim_warp_blender51.sim"):
            items = vars(value).items() if hasattr(value, "__dict__") else ()
        else:
            return
        for name, item in items:
            visit(item, f"{path}.{name}")

    for index, owner in enumerate(owners):
        visit(owner, f"owner{index}")
    return {"payload_bytes": sum(item["payload_bytes"] for item in arrays.values()),
            "array_count": len(arrays), "arrays": list(arrays.values())}


def memory_sample(device):
    wp.synchronize_device(device)
    return {"free_bytes": int(device.free_memory),
            "pool_used_bytes": int(wp.get_mempool_used_mem_current(device)),
            "pool_used_high_bytes": int(wp.get_mempool_used_mem_high(device))}


def run_case(name, settings, device):
    solver = None
    record = {"name": name, "status": "FAIL", "settings": {
        key: getattr(settings, key) for key in (
            "sample_type", "max_particles", "max_whitewater", "leaf_capacity",
            "surface_enabled", "surface_resolution_mode", "narrow_band_enabled",
            "surface_tension_enabled", "export_attributes")}}
    try:
        record["before"] = memory_sample(device)
        solver = FluidSolver(settings)
        if settings.surface_enabled:
            solver.surface.ensure_resources()
        if settings.surface_tension_enabled:
            solver.surface_tension.ensure_resources()
            solver.surface._ensure_band_buffers(solver.grid.voxel_capacity, device)
        record["initial"] = inventory(solver)
        record["initial_memory"] = memory_sample(device)
        record["initial_allowance_bytes"] = estimate_solver_bytes(settings)
        assert record["initial_allowance_bytes"] >= record["initial"]["payload_bytes"]
        assert bool(solver.narrow_band is not None) == settings.narrow_band_enabled
        if settings.surface_enabled and settings.surface_resolution_mode == 1:
            assert solver.surface.grid is not solver.grid
        old_particles, old_bins = solver.particles, solver.bins
        old_candidates = (solver.whitewater.candidates, solver.whitewater.candidate_offsets,
                          solver.whitewater.candidate_types)
        old_payload = inventory(old_particles, old_bins, old_candidates)
        original_allocate = solver.whitewater.allocate_candidate_workspace

        def observe_candidate(capacity):
            workspace = original_allocate(capacity)
            # This is the final allocation before commit; all old arrays and
            # all replacement arrays are simultaneously alive here.
            record["replacement_boundary_memory"] = memory_sample(device)
            return workspace

        solver.whitewater.allocate_candidate_workspace = observe_candidate
        result = solver.ensure_particle_capacity(settings.max_particles + 1)
        solver.whitewater.allocate_candidate_workspace = original_allocate
        assert result.grew, result
        fresh_candidates = (solver.whitewater.candidates, solver.whitewater.candidate_offsets,
                            solver.whitewater.candidate_types)
        fresh = inventory(solver.particles, solver.bins, fresh_candidates)
        retained = inventory(old_particles, old_bins, old_candidates,
                             solver.particles, solver.bins, fresh_candidates)
        assert retained["payload_bytes"] == old_payload["payload_bytes"] + fresh["payload_bytes"]
        assert result.peak_allocation_bytes >= fresh["payload_bytes"]
        record["growth"] = {"current": result.current, "target": result.target,
                            "additional_allocation_allowance_bytes": result.peak_allocation_bytes,
                            "old_payload_bytes": old_payload["payload_bytes"],
                            "fresh": fresh, "retained": retained,
                            "memory_with_retained_old_arrays": memory_sample(device)}
        # Release test-owned retention before solver teardown.
        del old_particles, old_bins, old_candidates, fresh_candidates, original_allocate
        record["status"] = "PASS"
        return record
    except Exception:
        record["error"] = traceback.format_exc()
        return record
    finally:
        if solver is not None:
            solver.release()
        solver = old_particles = old_bins = old_candidates = fresh_candidates = None
        original_allocate = None
        gc.collect()
        record["after_release_memory"] = memory_sample(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "FAIL", "cases": [],
              "scope": "Real CUDA allocation and replacement retention; no frame advancement",
              "measurement_limits": [
                  "Payload is unique reachable Warp-array element bytes, excluding allocator overhead.",
                  "Pool current/high and free VRAM are separate observations; high-water may predate a case.",
                  "Free VRAM may reflect other processes; samples are not isolated allocator-reservation proof.",
                  "Dense marching-cubes backend internals and runtime-only lazy allocations are not materialized."]}
    try:
        assert bpy.app.background, "use a background isolated Blender process"
        private_extensions = os.environ.get("BLENDER_USER_EXTENSIONS", "")
        assert private_extensions and Path(private_extensions).is_absolute(), "private BLENDER_USER_EXTENSIONS required"
        assert "--factory-startup" in sys.argv, "factory startup required"
        device = wp.get_device(script_device())
        assert device.is_cuda, "CUDA allocation proof requires CUDA"
        digest = hashlib.sha256()
        for path in sorted((ROOT / "fluidSim_warp_blender51" / "sim").rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
        report.update({"source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sim_source_sha256": digest.hexdigest(), "blender_version": bpy.app.version_string,
            "warp_version": wp.__version__, "device": str(device), "device_name": device.name})
        base = SolverSettings(domain_min=(0.0, 0.0, 0.0), domain_max=(0.8, 0.8, 0.8),
                              voxel_size=0.1, leaf_capacity=1, max_particles=32,
                              max_whitewater=16, surface_enabled=False, device=str(device))
        scenarios = [
            ("base", base),
            ("apic_optional", replace(base, sample_type=2, export_attributes=True,
                                      particles_attribute_color=True, particles_attribute_source_id=True,
                                      particles_attribute_uid=True)),
            ("capillarity_narrow_band", replace(base, narrow_band_enabled=True,
                                               surface_tension_enabled=True)),
            ("secondary_surface", replace(base, surface_enabled=True, surface_resolution_mode=1,
                                          surface_resolution_multiplier=2.0, surface_leaf_capacity_scale=8.0)),
        ]
        for name, settings in scenarios:
            record = run_case(name, settings, device)
            report["cases"].append(record)
            if record["status"] != "PASS":
                raise AssertionError(f"{name}: {record.get('error', 'failed')}")
        report["status"] = "PASS"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ALLOCATION-ACCOUNTING-ACCEPTANCE-{report['status']}: {output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
