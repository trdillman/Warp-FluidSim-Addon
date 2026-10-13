"""Diagnostic controls only: no production changes, settings/defaults or speed claims.

Run with the certified Warp interpreter. Uses the existing 16x16x32 tank.
The single-check control changes stopping/check frequency, not the PDE or dt.
"""
from pathlib import Path
import argparse
from collections import Counter
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "tests" / "solver"))
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from _narrow_band_harness import settings, lattice, H  # noqa: E402
from fluidSim_warp_blender51.sim.solver import FluidSolver  # noqa: E402
from fluidSim_warp_blender51.sim.pressure import k_apply_matrix  # noqa: E402


def run_case(device, name, frames=30):
    options = {"narrow_band":False}
    if name == "multigrid":
        options["pressure_preconditioner"] = "MULTIGRID"
    if name == "jacobi_800":
        options["pressure_iterations"] = 800
    if name == "jacobi_narrow":
        options["narrow_band"] = True
    solver = FluidSolver(settings(device, **options))
    records, measurements = [], []
    start = time.perf_counter()
    try:
        solver.inject_particles(lattice())
        ps = solver.pressure_solver
        original = ps.solve_pcg
        def observed(voxel_type,max_iterations,tolerance,use_coarse_guess=False,check_every=None):
            supported = wp.is_conditional_graph_supported
            if name == "jacobi_host":
                wp.is_conditional_graph_supported = lambda: False
            if name == "jacobi_single_check":
                check_every = max_iterations
            try:
                result = original(voxel_type,max_iterations,tolerance,use_coarse_guess,check_every)
            finally:
                wp.is_conditional_graph_supported = supported
            item = {"frame":int(solver.frame),"status":str(ps.last_status.get("reason", "unknown")),
                    "iterations":int(ps.last_iterations),"limit":int(max_iterations),
                    "residual":float(ps.last_residual),"check_every":int(ps.last_pcg_check_every),
                    "mode":ps.last_pcg_execution_mode}
            if not records or solver.frame >= frames - 1:
                full_n = ps._dim()
                count = int(ps._pressure_count.numpy()[0])
                work_n = min(full_n,1 << (max(1,count)-1).bit_length())
                ap = wp.zeros(full_n,dtype=float,device=device)
                cp = wp.zeros(work_n,dtype=float,device=device)
                cap = wp.zeros(work_n,dtype=float,device=device)
                wp.launch(k_apply_matrix,dim=work_n,inputs=[ps.pressure,ps.pressure_indices,
                    ps._pressure_count,voxel_type,ps.inv_diag,*ps._stencil_args(),ap,cp,cap],device=device)
                indices = ps.pressure_indices.numpy()[:count]
                rhs = ps.rhs.numpy()[indices]
                residual = rhs - ap.numpy()[indices]
                item.update(true_residual=float(np.linalg.norm(residual)),
                    rhs_norm=float(np.linalg.norm(rhs)),pressure_cells=count,
                    acceptance_norm=float(tolerance*np.sqrt(max(1,full_n))))
            records.append(item)
            return result
        ps.solve_pcg = observed
        for frame in range(frames+1):
            if frame:
                solver.step_frame()
            positions = solver.particle_positions()
            occupied = len(np.unique(np.floor(positions/H).astype(np.int32),axis=0))
            measurements.append({"frame":frame,"particles":len(positions),
                "height99":float(np.percentile(positions[:,2],99)),
                "height_max":float(positions[:,2].max()),
                "occupied_particle_cells":occupied,
                "occupied_volume_proxy":float(occupied*H**3)})
        result = {"case":name,"frames":measurements,"pressure_solves":records,
                  "statuses":dict(Counter(row["status"] for row in records)),
                  "instrumented_wall_seconds":time.perf_counter()-start}
        print(json.dumps({"case":name,"initial":measurements[0],"last":measurements[-1],
                          "statuses":result["statuses"],"first_pressure":records[0]}),flush=True)
        return result
    finally:
        solver.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--frames",type=int,default=30)
    parser.add_argument("--cases",nargs="+",default=["jacobi","jacobi_800","jacobi_host","jacobi_single_check","multigrid","jacobi_narrow"])
    args = parser.parse_args()
    wp.init()
    device = wp.get_device(args.device)
    if args.device.startswith("cuda") and not device.is_cuda:
        raise RuntimeError("Requested CUDA device was not resolved")
    report = {"warp":wp.__version__,"warp_file":wp.__file__,"device":str(device),
              "device_name":device.name,"source_root":str(ROOT),"cases":[],
              "scope":"Instrumented correctness diagnosis, not a performance benchmark"}
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    for name in args.cases:
        report["cases"].append(run_case(device,name,args.frames))
        out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("RESTING-COLUMN-DIAGNOSIS-COMPLETE",flush=True)


if __name__ == "__main__":
    main()
