"""Diagnostic: log the full PCG residual trajectory of the viscosity solve.

Measurement-only. Production code is untouched; ``ViscositySolver._solve_component``
is monkeypatched in-process with a replica of the shipped loop that records the
residual after *every* iteration instead of only at ``it % 10``.

Writes raw JSON under ``artifacts/visc_pcg_residual/``.

Usage:
    python tools/diagnose_viscosity_pcg_convergence.py --device cuda:0 \
        --frames 3 --warmup 1 --scene-preset dam_break --max-iterations 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS)
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import profile_gpu_solver as pgs  # noqa: E402


def _utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def install_probe(records, max_iterations_override, record_limit, skip_solves=0):
    import warp as wp

    from fluidSim_warp_blender51.sim import viscosity as V
    from fluidSim_warp_blender51.sim.pressure import (
        k_apply_preconditioner, k_compute_alpha, k_compute_beta, k_update_p,
        k_update_xr,
    )

    original = V.ViscositySolver._solve_component
    state = {"solve_index": 0, "component": 0}

    def instrumented(self, comp, weight, viscosity_field, voxel_type, dt,
                     use_pcg, iterations, tolerance, check_every=1):
        g = self.grid
        n = self._active_count()
        if n <= 0 or not use_pcg:
            return original(self, comp, weight, viscosity_field, voxel_type, dt,
                            use_pcg, iterations, tolerance, check_every)

        idx = state["solve_index"]
        comp_idx = state["component"]
        state["component"] = (comp_idx + 1) % 3
        if state["component"] == 0:
            state["solve_index"] = idx + 1
        component_solves_seen = idx * 3 + comp_idx
        recording = (component_solves_seen >= skip_solves
                     and len(records) < record_limit)

        budget = int(iterations)
        if recording and max_iterations_override:
            budget = int(max_iterations_override)

        stencil = [g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                   g.num_leaves]
        wp.launch(V.k_viscosity_counter_increment, dim=1, inputs=[self.counter],
                  device=self.device)
        wp.launch(V.k_viscosity_build_rhs, dim=n,
                  inputs=[comp, weight, voxel_type, g.num_leaves, self.rhs, self.x],
                  device=self.device)
        wp.launch(V.k_viscosity_compute_inverse_diagonal, dim=n,
                  inputs=[viscosity_field, weight, voxel_type, *stencil,
                          g.voxel_size, dt, self.inv_diag],
                  device=self.device)
        wp.launch(V.k_viscosity_apply_matrix, dim=n,
                  inputs=[self.x, viscosity_field, weight, voxel_type,
                          *stencil, g.voxel_size, dt, self.ap],
                  device=self.device)
        wp.launch(V.k_viscosity_initialize_pcg, dim=n,
                  inputs=[self.rhs, self.ap, self.inv_diag, g.num_leaves, self.r],
                  device=self.device)
        wp.launch(V.k_viscosity_finalize_initial_residual, dim=n,
                  inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                  device=self.device)
        wp.launch(V.k_viscosity_initialize_direction, dim=n,
                  inputs=[self.z, g.num_leaves, self.p_dir],
                  device=self.device)
        wp.utils.array_inner(self.r, self.z, out=self.rz, count=n)

        rec = None
        if recording:
            wp.utils.array_inner(self.rhs, self.rhs, out=self.residual, count=n)
            b_sq = float(self.residual.numpy()[0])
            wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
            r0_sq = float(self.residual.numpy()[0])
            inv_diag_host = self.inv_diag.numpy()[:n]
            nonzero = inv_diag_host[inv_diag_host > 0.0]
            rec = {
                "solveIndex": idx,
                "component": comp_idx,
                "activeVoxels": int(n),
                "dt": float(dt),
                "voxelSize": float(g.voxel_size),
                "iterationBudget": int(budget),
                "shippedIterationBudget": int(iterations),
                "tolerance": float(tolerance),
                "thresholdResidualSq": float(tolerance * tolerance * max(1, n)),
                "rhsNormSq": b_sq,
                "initialResidualSq": r0_sq,
                "rzInitial": float(self.rz.numpy()[0]),
                "invDiagMin": float(nonzero.min()) if nonzero.size else 0.0,
                "invDiagMax": float(nonzero.max()) if nonzero.size else 0.0,
                "invDiagMean": float(nonzero.mean()) if nonzero.size else 0.0,
                "activeFaceCount": int(nonzero.size),
                "residualSq": [],
                "rz": [],
            }
            records.append(rec)

        it = 0
        stop_it = None
        while it < budget:
            wp.launch(V.k_viscosity_apply_matrix, dim=n,
                      inputs=[self.p_dir, viscosity_field, weight, voxel_type,
                              *stencil, g.voxel_size, dt, self.ap],
                      device=self.device)
            wp.utils.array_inner(self.p_dir, self.ap, out=self.p_ap, count=n)
            wp.launch(k_compute_alpha, dim=1,
                      inputs=[self.rz, self.p_ap, self.alpha], device=self.device)
            wp.launch(k_update_xr, dim=n,
                      inputs=[self.p_dir, self.ap, self.alpha, g.num_leaves,
                              self.x, self.r], device=self.device)
            wp.launch(k_apply_preconditioner, dim=n,
                      inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                      device=self.device)
            wp.utils.array_inner(self.r, self.z, out=self.rz_new, count=n)
            wp.launch(k_compute_beta, dim=1,
                      inputs=[self.rz_new, self.rz, self.beta], device=self.device)
            wp.launch(k_update_p, dim=n,
                      inputs=[self.z, self.beta, g.num_leaves, self.p_dir],
                      device=self.device)
            rz_host = float(self.rz_new.numpy()[0]) if recording else None
            wp.copy(self.rz, self.rz_new, count=1)
            it += 1
            if recording:
                wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
                res = float(self.residual.numpy()[0])
                rec["residualSq"].append(res)
                rec["rz"].append(rz_host)
                if stop_it is None and res < tolerance * tolerance * max(1, n):
                    stop_it = it
            elif it % max(1, int(check_every)) == 0 or it == budget:
                wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
                if float(self.residual.numpy()[0]) < tolerance * tolerance * max(1, n):
                    break
        if recording:
            rec["firstIterationBelowThreshold"] = stop_it
            shipped_stop = None
            for k, res in enumerate(rec["residualSq"][:int(iterations)], start=1):
                if ((k % 10 == 0 or k == int(iterations))
                        and res < rec["thresholdResidualSq"]):
                    shipped_stop = k
                    break
            rec["shippedCadenceStop"] = shipped_stop
        self.last_iterations = it

        wp.launch(V.k_viscosity_apply_collision, dim=n,
                  inputs=[self.x, comp, weight, voxel_type, g.num_leaves,
                          self.tmp_u],
                  device=self.device)
        wp.copy(comp, self.tmp_u, count=n)

    V.ViscositySolver._solve_component = instrumented
    return original


def install_status_probe(records, record_limit):
    """Record what the production solver *reports* for every component solve.

    Unlike ``install_probe`` this does not replace the solve; it runs the
    shipped code path and captures the status it publishes, so it measures the
    production convergence check rather than a replica of it.
    """
    import math

    from fluidSim_warp_blender51.sim import viscosity as V

    original = V.ViscositySolver._solve_component
    state = {"solve_index": 0, "component": 0}

    def instrumented(self, comp, weight, viscosity_field, voxel_type, dt,
                     use_pcg, iterations, tolerance, check_every=1):
        idx = state["solve_index"]
        comp_idx = state["component"]
        state["component"] = (comp_idx + 1) % 3
        if state["component"] == 0:
            state["solve_index"] = idx + 1
        n = self._active_count()
        status = original(self, comp, weight, viscosity_field, voxel_type, dt,
                          use_pcg, iterations, tolerance, check_every)
        if len(records) < record_limit:
            threshold = float(tolerance) * math.sqrt(max(1, n))
            residual = float(self.last_residual)
            records.append({
                "solveIndex": idx,
                "component": comp_idx,
                "activeVoxels": int(n),
                "dt": float(dt),
                "iterationBudget": int(iterations),
                "checkEvery": int(check_every),
                "tolerance": float(tolerance),
                "thresholdResidual": threshold,
                "residual": residual,
                "residualOverThreshold": (
                    residual / threshold if threshold > 0.0 else float("inf")),
                "iterations": int(self.last_iterations),
                "status": str(self.last_viscosity_status),
                "statusCode": int(self.last_viscosity_status_code),
                "failure": str(self.last_viscosity_failure),
                "executionMode": str(self.last_pcg_execution_mode),
                "converged": bool(status["converged"]) if status else False,
            })
        return status

    V.ViscositySolver._solve_component = instrumented
    return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--emit-frames", type=int, default=2)
    parser.add_argument("--scene-preset", default="dam_break",
                        choices=("benchmark", "dam_break"))
    parser.add_argument("--viscosity-constant", type=float, default=1.0)
    parser.add_argument("--viscosity-tolerance", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Run recorded solves past the shipped budget to see "
                             "whether the residual is still decreasing (0 = shipped).")
    parser.add_argument("--record-solves", type=int, default=6,
                        help="How many component solves to record in full.")
    parser.add_argument("--skip-solves", type=int, default=0,
                        help="Component solves to run on the shipped path before "
                             "recording starts (lets recording land in a "
                             "steady-state frame).")
    parser.add_argument("--probe", default="trajectory",
                        choices=("trajectory", "status"),
                        help="trajectory: replay a replica of the loop and log "
                             "every residual. status: run the production solve "
                             "and log the status it reports.")
    parser.add_argument("--check-every", type=int, default=None,
                        help="Override the production convergence cadence "
                             "(status probe only; default = shipped).")
    parser.add_argument("--out-dir", default="artifacts/visc_pcg_residual")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    records = []
    if args.probe == "status":
        install_status_probe(records, args.record_solves)
        if args.check_every is not None:
            from fluidSim_warp_blender51.sim import viscosity as _V

            _V.DEFAULT_VISCOSITY_CHECK_EVERY = int(args.check_every)
            _V.ViscositySolver.solve.__defaults__ = (
                True, 40, 1.0e-4, int(args.check_every))
    else:
        install_probe(records, args.max_iterations, args.record_solves,
                      skip_solves=args.skip_solves)

    profile_args = argparse.Namespace(
        device=args.device, frames=args.frames, warmup=args.warmup,
        emit_frames=args.emit_frames,
        voxel_size=pgs._DEFAULTS["voxel_size"],
        max_particles=pgs._DEFAULTS["max_particles"],
        max_whitewater=200000,
        min_substeps=pgs._DEFAULTS["min_substeps"],
        max_substeps=pgs._DEFAULTS["max_substeps"],
        whitewater=False, no_surface=False, sync=False, no_phase_timers=False,
        scene_preset=args.scene_preset, no_whitewater=False,
        viscosity=True, viscosity_constant=args.viscosity_constant,
        viscosity_iterations=None, viscosity_tolerance=args.viscosity_tolerance,
        surface_tension=False, surface_tension_coefficient=0.072,
        json_out=None,
    )
    t0 = time.perf_counter()
    summary, _raw = pgs.run_profile(profile_args)
    wall = time.perf_counter() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(
        args.out_dir, f"visc_pcg_residual{tag}_{_utc_stamp()}.json")
    payload = {
        "schema": ("visc_pcg_status/1" if args.probe == "status"
                   else "visc_pcg_residual/1"),
        "command": " ".join(sys.argv),
        "device": summary.get("device"),
        "warpVersion": summary.get("warpVersion"),
        "wallSeconds": wall,
        "profileSummary": summary,
        "args": vars(args),
        "solves": records,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print(f"wrote {path}")
    if args.probe == "status":
        for rec in records:
            print(
                f"solve {rec['solveIndex']} comp {rec['component']}: "
                f"n={rec['activeVoxels']} dt={rec['dt']:.6f} "
                f"it={rec['iterations']}/{rec['iterationBudget']} "
                f"status={rec['status']} mode={rec['executionMode']} "
                f"r={rec['residual']:.6g} thr={rec['thresholdResidual']:.6g} "
                f"r/thr={rec['residualOverThreshold']:.4f}")
        return 0
    for rec in records:
        traj = rec["residualSq"]
        thr = rec["thresholdResidualSq"]
        r40 = traj[39] if len(traj) > 39 else float("nan")
        print(
            f"solve {rec['solveIndex']} comp {rec['component']}: "
            f"n={rec['activeVoxels']} dt={rec['dt']:.6f} "
            f"|b|^2={rec['rhsNormSq']:.6g} r0^2={rec['initialResidualSq']:.6g} "
            f"thr={thr:.6g} r40^2={r40:.6g} rEnd^2={traj[-1]:.6g} "
            f"firstBelow={rec['firstIterationBelowThreshold']} "
            f"shippedStop={rec['shippedCadenceStop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
