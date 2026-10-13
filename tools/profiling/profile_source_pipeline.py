#!/usr/bin/env python3
"""Measured Phase 2 source compiler profiler with approved Balanced gates."""
from __future__ import annotations
import argparse
import json
import statistics
import time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for _name in list(sys.modules):
 if _name == "fluidSim_warp_blender51" or _name.startswith("fluidSim_warp_blender51."):
  del sys.modules[_name]
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from fluidSim_warp_blender51.sim.enums import (  # noqa: E402
    SourceEvaluationCadence, SourceLifecycle, SourceOperation, SourcePayloadKind,
    SourceShapeShape)
from fluidSim_warp_blender51.sim.source_regions import (  # noqa: E402
    AnalyticRegionPlan, CompileContext, MeshRegionPlan, PayloadPlan, PointRegionPlan,
    SourceCompiler, SourceKey, SourcePlan, VolumeRegionPlan)

def parse():
 p=argparse.ArgumentParser()
 p.add_argument("--frames",type=int,default=30)
 p.add_argument("--json-out",required=True)
 argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
 return p.parse_args(argv)
def ctx(frame): return CompileContext(frame,0,1,"cuda:0",(0,0,0),.25,(0,0,0),(2,2,2),upload=True)
def source(region,key,cadence): return SourcePlan(SourceKey(key),SourceOperation.LIQUID,SourceLifecycle.TOP_UP,cadence,region,PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
def measure(name,plans,frames):
 c=SourceCompiler()
 samples=[]
 peak=0
 peak_live=0
 peak_entries=0
 diagnostics=set()
 for frame in range(frames+2):
  t=time.perf_counter()
  b=c.compile(plans,ctx(frame))
  wp.synchronize_device("cuda:0")
  elapsed=(time.perf_counter()-t)*1000
  diagnostics.update(d.code for d in b.diagnostics if d.severity=="ERROR")
  peak=max(peak,b.device_bytes)
  peak_live=max(peak_live,c.live_cache_bytes)
  peak_entries=max(peak_entries,len(c._cache))
  if frame>=2:
      samples.append(elapsed)
 live_before_reset=c.live_cache_bytes
 c.release()
 retained=c.live_cache_bytes
 return {"name":name,"median_ms":statistics.median(samples),"p95_ms":sorted(samples)[max(0,int(len(samples)*.95)-1)],"peak_bytes":peak,"peak_live_cache_bytes":peak_live,"peak_cache_entries":peak_entries,"live_before_reset_bytes":live_before_reset,"compile_count":c.compile_count,"cache_hits":c.cache_hits,"retained_after_reset_bytes":retained,"error_diagnostics":sorted(diagnostics),"samples_ms":samples}

def _make_solver(compiled,with_source):
 from fluidSim_warp_blender51.sim.solver import FluidSolver,SolverSettings
 from fluidSim_warp_blender51.sim.sourcing import SourceShapeDesc
 s=SolverSettings()
 s.domain_min=(0,0,0)
 s.domain_max=(1.6,1.6,1.6)
 s.voxel_size=.2
 s.max_particles=20000
 s.whitewater_enabled=False
 s.surface_enabled=False
 s.min_substeps=1
 s.max_substeps=1
 s.pressure_iterations=20
 s.device="cuda:0"
 solver=FluidSolver(s)
 if with_source:
  if compiled:
      solver.set_source_batch(native_plans=(source(AnalyticRegionPlan(shape=SourceShapeShape.BOX,position=(.8,.8,.8),half_extents=(.3,.3,.3)),"ab",SourceEvaluationCadence.STATIC),))
  else:
      solver.set_source_batch((SourceShapeDesc(shape=SourceShapeShape.BOX,position=(.8,.8,.8),half_extents=(.3,.3,.3),radius=.3,height=.3,emit_rate=8.0,source_id="ab"),))
 return solver

def measure_solver_pair(with_source,frames):
 legacy,compiled=_make_solver(False,with_source),_make_solver(True,with_source)
 samples=[[],[]]
 for frame in range(frames+2):
  order=(0,1) if frame%2==0 else (1,0)
  for index in order:
   solver=legacy if index==0 else compiled
   elapsed=float(solver.step_frame().step_time*1000.0)
   if frame>=2:
       samples[index].append(elapsed)
 names=("legacy_analytic","compiled_analytic") if with_source else ("legacy_no_source","compiled_no_source")
 result=[]
 for name,values in zip(names,samples):
     result.append({"name":name,"median_ms":statistics.median(values),"p95_ms":sorted(values)[max(0,int(len(values)*.95)-1)],"samples_ms":values})
 legacy.release()
 compiled.release()
 return result

def main():
 a=parse()
 frames=max(3,a.frames)
 wp.init()
 v=np.array([[.5,.5,.5],[1.5,.5,.5],[1.5,1.5,.5],[.5,1.5,.5],[.5,.5,1.5],[1.5,.5,1.5],[1.5,1.5,1.5],[.5,1.5,1.5]],np.float32)
 t=np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)],np.int32)
 cases=[measure("no_source",(),frames),measure("analytic",(source(AnalyticRegionPlan(shape=SourceShapeShape.SPHERE,position=(1,1,1),radius=.5),"a",SourceEvaluationCadence.STATIC),),frames),measure("static_solid_mesh",(source(MeshRegionPlan(v,t,solid=True),"m",SourceEvaluationCadence.STATIC),),frames),measure("animated_shell_mesh",(source(MeshRegionPlan(v,t,solid=False,shell_half_width=.2),"s",SourceEvaluationCadence.EVERY_SIM_FRAME),),frames),measure("point_rasterization",(source(PointRegionPlan([[1,1,1]],rasterize_radius=.3),"p",SourceEvaluationCadence.EVERY_SIM_FRAME),),frames)]
 try:
  import openvdb
  import tempfile
  volume_path=str(Path(tempfile.gettempdir())/"fluidsim_source_profile.vdb")
  grid=openvdb.FloatGrid()
  grid.name="density"
  acc=grid.getAccessor()
  for x in range(4):
   for y in range(4):
    for z in range(4):
        acc.setValueOn((x,y,z),1.0)
  openvdb.write(volume_path,grids=[grid])
  cases.append(measure("volume_grid_sampling",(source(VolumeRegionPlan(volume_path,mask_grid="density"),"v",SourceEvaluationCadence.EVERY_SIM_FRAME),),frames))
 except ImportError:
  raise SystemExit("SOURCE-PIPELINE-PROFILE-FAIL bundled openvdb unavailable")
 ab=measure_solver_pair(False,frames)+measure_solver_pair(True,frames)
 def regression(new,old,key): return (new[key]-old[key])/max(old[key],.001)
 no_source_median=regression(ab[1],ab[0],"median_ms")
 no_source_p95=regression(ab[1],ab[0],"p95_ms")
 analytic_reg=regression(ab[3],ab[2],"median_ms")
 analytic_p95=regression(ab[3],ab[2],"p95_ms")
 # Exercise the measured dynamic work envelope immediately below the
 # pre-approved 33.3 ms p95 bound, not merely a rejection code.
 near_triangles=np.tile(t,(6,1))[:72]
 near_case=measure("animated_mesh_near_budget",(source(MeshRegionPlan(v,near_triangles,solid=False,shell_half_width=.2),"near",SourceEvaluationCadence.EVERY_SIM_FRAME),),frames)
 cases.append(near_case)
 dynamic=[c for c in cases if c["name"] in {"animated_shell_mesh","animated_mesh_near_budget","point_rasterization","volume_grid_sampling"}]
 # A production-scale 50k-triangle STATIC source at 0.02 m uses Warp's CUDA
 # BVH. The same input is rejected only for EVERY_SIM_FRAME when it cannot meet
 # the approved interactive budget.
 huge_triangles=np.tile(t,(4167,1))[:50000]
 huge_static=source(MeshRegionPlan(v,huge_triangles,solid=False,shell_half_width=.2),"huge-static",SourceEvaluationCadence.STATIC)
 scaling_start=time.perf_counter()
 huge_static_batch=SourceCompiler().compile((huge_static,),CompileContext(1,0,1,"cuda:0",(0,0,0),.02,(0,0,0),(2,2,2),upload=True))
 wp.synchronize_device("cuda:0")
 scaling_ms=(time.perf_counter()-scaling_start)*1000.0
 huge_dynamic=source(MeshRegionPlan(v,huge_triangles,solid=False,shell_half_width=.2),"huge-dynamic",SourceEvaluationCadence.EVERY_SIM_FRAME)
 huge_dynamic_batch=SourceCompiler().compile((huge_dynamic,),CompileContext(1,0,1,"cpu",(0,0,0),.02,(0,0,0),(2,2,2),upload=False))
 scaling={"near_budget_triangle_count":int(len(near_triangles)),"near_budget_comparisons":int(512*len(near_triangles)),"near_budget_median_ms":near_case["median_ms"],"near_budget_p95_ms":near_case["p95_ms"],"realistic_triangle_count":int(len(huge_triangles)),"realistic_voxel_size":.02,"static_cuda_bvh_compile_ms":scaling_ms,"static_cuda_bvh_diagnostics":[d.code for d in huge_static_batch.diagnostics],"dynamic_budget_diagnostics":[d.code for d in huge_dynamic_batch.diagnostics]}
 checks={"all_profile_cases_compile_without_error":all(not c["error_diagnostics"] for c in cases),"no_source_median_regression_le_10pct":no_source_median<=.10,"no_source_p95_regression_le_15pct":no_source_p95<=.15,"analytic_median_regression_le_10pct":analytic_reg<=.10,"analytic_p95_regression_le_15pct":analytic_p95<=.15,"static_no_rebuild_after_warmup":all(c["compile_count"]==1 for c in cases if c["name"] in {"analytic","static_solid_mesh"}),"dynamic_cache_bounded_to_one_generation":all(c["peak_cache_entries"]<=1 for c in dynamic),"dynamic_median_le_16_7ms":all(c["median_ms"]<=16.7 for c in dynamic),"dynamic_p95_le_33_3ms":all(c["p95_ms"]<=33.3 for c in dynamic),"peak_staging_le_512mib":all(max(c["peak_bytes"],c["peak_live_cache_bytes"])<=512*1024*1024 for c in cases),"realistic_static_mesh_cuda_bvh":not any(d.severity=="ERROR" for d in huge_static_batch.diagnostics) and huge_static_batch.sources[0].region.covered_voxel_count>0,"oversized_dynamic_mesh_rejected":"SOURCE_DYNAMIC_MESH_BUDGET" in scaling["dynamic_budget_diagnostics"],"reset_within_8mib":all(c["retained_after_reset_bytes"]<=8*1024*1024 for c in cases)}
 report={"schema":"fluidsim-source-profile-v1","warmup_frames":2,"measured_frames":frames,"device":"cuda:0","device_name":wp.get_device("cuda:0").name,"methodology":"same-run same-device production FluidSolver.step_frame A/B: preserved legacy descriptors versus immutable compiled SourcePlan; identical settings and fixtures. Mesh scaling is evidence-only; no latency threshold was invented after measurement.","thresholds":{"existing_step_median_regression":.10,"existing_step_p95_regression":.15,"dynamic_median_ms":16.7,"dynamic_p95_ms":33.3,"peak_source_bytes":512*1024*1024,"reset_retained_bytes":8*1024*1024},"ab_solver_steps":ab,"regressions":{"no_source_median":no_source_median,"no_source_p95":no_source_p95,"analytic_median":analytic_reg,"analytic_p95":analytic_p95},"mesh_scaling_evidence":scaling,"cases":cases,"checks":checks,"ok":all(checks.values())}
 out=Path(a.json_out)
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(report,indent=2),encoding="utf8")
 if not report["ok"]:
     raise SystemExit("SOURCE-PIPELINE-PROFILE-FAIL "+str({k:v for k,v in checks.items() if not v}))
 print("SOURCE-PIPELINE-PROFILE-PASS")
if __name__=="__main__":
    main()
