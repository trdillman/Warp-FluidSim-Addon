import numpy as np
from fluidSim_warp_blender51.sim import constants as C, sourcing
from fluidSim_warp_blender51.sim.enums import (
    BlendMode, SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind, SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan, PayloadPlan, PointRegionPlan, SourceKey, SourcePlan,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings


def arrays(device):
 def f():
  return sourcing.wp.zeros(C.LEAF_VOLUME,dtype=float,device=device)
 return {"voxel_type":sourcing.wp.zeros(C.LEAF_VOLUME,dtype=sourcing.wp.int32,device=device),"pressure":f(),"temperature":f(),"viscosity":f(),"density":f(),"u":f(),"v":f(),"w":f()}

def launch(operation,coverage,values):
 device=sourcing.wp.get_preferred_device()
 out=arrays(device)
 leaves=sourcing.wp.array(np.array([[0,0,0]],np.int32),dtype=sourcing.wp.vec3i,device=device)
 sourcing.wp.launch(sourcing.k_apply_compiled_source,dim=C.LEAF_VOLUME,inputs=[int(operation),sourcing.wp.array(np.full(C.LEAF_VOLUME,coverage,np.float32),dtype=float,device=device),sourcing.wp.empty(0,dtype=float,device=device),sourcing.wp.empty(0,dtype=sourcing.wp.vec3,device=device),leaves,1,sourcing.wp.array(np.array([0],np.int32),dtype=sourcing.wp.int32,device=device),sourcing.wp.vec3i(8,8,8),sourcing.wp.vec3i(1,1,1),2.0,sourcing.wp.vec3(1,2,3),1,1.0,0.5],outputs=list(out.values()),device=device)
 return {k:v.numpy() for k,v in out.items()}

def test_pressure_field_writes_only_pressure_and_never_liquid_bits():
 out=launch(SourceOperation.PRESSURE,1.0,None)
 assert np.allclose(out["pressure"],2.0)
 assert np.count_nonzero(out["voxel_type"])==0
 for key in ("temperature","viscosity","density","u","v","w"):
     assert np.count_nonzero(out[key])==0

def test_collision_precedence_blocks_later_liquid_classification():
 out=launch(SourceOperation.COLLISION,1.0,None)
 assert np.all((out["voxel_type"] & C.VOXEL_COLLISION)!=0)
 assert np.all((out["voxel_type"] & (C.VOXEL_FLUID|C.VOXEL_SINK))==0)


def test_pressure_source_is_applied_after_solve_before_projection_without_liquid(cuda_device):
 settings=SolverSettings()
 settings.domain_min=(0,0,0)
 settings.domain_max=(2,2,2)
 settings.voxel_size=.25
 settings.max_particles=64
 settings.surface_enabled=False
 settings.whitewater_enabled=False
 settings.pressure_iterations=20
 settings.device=cuda_device
 solver=FluidSolver(settings)
 pressure=SourcePlan(SourceKey("pressure"),SourceOperation.PRESSURE,
  SourceLifecycle.EVERY_FRAME,SourceEvaluationCadence.STATIC,
  AnalyticRegionPlan(shape=SourceShapeShape.BOX,position=(1,1,1),half_extents=(.5,.5,.5)),
  PayloadPlan(kind=SourcePayloadKind.PRESSURE,pressure=5.0))
 solver.set_source_batch(native_plans=(pressure,))
 solver.step_frame()
 assert solver.particles.host_count==0
 assert np.count_nonzero(solver.voxel_type.data.numpy() & C.VOXEL_FLUID)==0
 assert np.max(np.abs(solver.pressure_solver.pressure.numpy()))>=4.99
 assert solver.sourcing.last_pressure_apply_count>0
 solver.release()


def test_pressure_source_changes_projected_liquid_velocity(cuda_device):
 def run(with_pressure):
  settings=SolverSettings()
  settings.domain_min=(0,0,0)
  settings.domain_max=(2,2,2)
  settings.voxel_size=.25
  settings.max_particles=128
  settings.surface_enabled=False
  settings.whitewater_enabled=False
  settings.pressure_iterations=40
  settings.device=cuda_device
  solver=FluidSolver(settings)
  points=PointRegionPlan([[.55,.75,.75],[.55,1.0,.75],[.55,.75,1.0],[.55,1.0,1.0]])
  liquid=SourcePlan(SourceKey("liquid"),SourceOperation.LIQUID,
   SourceLifecycle.DIRECT_APPEND,SourceEvaluationCadence.STATIC,points,
   PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
  plans=[liquid]
  if with_pressure:
   plans.append(SourcePlan(SourceKey("pressure"),SourceOperation.PRESSURE,
    SourceLifecycle.EVERY_FRAME,SourceEvaluationCadence.STATIC,
    AnalyticRegionPlan(shape=SourceShapeShape.BOX,position=(.75,.875,.875),half_extents=(.25,.5,.5)),
    PayloadPlan(kind=SourcePayloadKind.PRESSURE,pressure=8.0)))
  solver.set_source_batch(native_plans=plans)
  solver.step_frame()
  result=solver.particles.velocity.numpy()[:solver.particles.host_count].copy()
  applied=solver.sourcing.last_pressure_apply_count
  solver.release()
  return result,applied
 baseline,_=run(False)
 influenced,applied=run(True)
 assert applied>0
 assert baseline.shape==influenced.shape
 assert np.max(np.linalg.norm(influenced-baseline,axis=1))>1.0e-4


def test_staggered_source_survives_p2g_ordering_in_real_solver(cuda_device):
 settings=SolverSettings()
 settings.domain_min=(0,0,0)
 settings.domain_max=(2,2,2)
 settings.voxel_size=.25
 settings.max_particles=32
 settings.surface_enabled=False
 settings.whitewater_enabled=False
 settings.gravity=(0,0,0)
 settings.device=cuda_device
 solver=FluidSolver(settings)
 plan=SourcePlan(SourceKey("staggered"),SourceOperation.STAGGERED_VELOCITY,
  SourceLifecycle.EVERY_FRAME,SourceEvaluationCadence.STATIC,
  AnalyticRegionPlan(shape=SourceShapeShape.BOX,position=(1,1,1),half_extents=(.5,.5,.5)),
  PayloadPlan(kind=SourcePayloadKind.STAGGERED_VECTOR,vector=(1,2,3)))
 solver.set_source_batch(native_plans=(plan,))
 solver.step_frame()
 assert solver.sourcing.last_post_p2g_apply_count>0
 assert np.max(np.abs(solver.vel.u.numpy()))>=.99
 assert np.max(np.abs(solver.vel.v.numpy()))>=1.99
 assert np.max(np.abs(solver.vel.w.numpy()))>=2.99
 solver.release()


def test_staggered_source_applies_independent_face_component_samples():
 device=sourcing.wp.get_preferred_device()
 leaves=sourcing.wp.array(
  np.array([[0,0,0]],np.int32),dtype=sourcing.wp.vec3i,device=device)
 coverage=sourcing.wp.array(np.ones(C.LEAF_VOLUME,np.float32),dtype=float,device=device)
 u_values=sourcing.wp.array(np.full(C.LEAF_VOLUME,1,np.float32),dtype=float,device=device)
 v_values=sourcing.wp.array(np.full(C.LEAF_VOLUME,2,np.float32),dtype=float,device=device)
 w_values=sourcing.wp.array(np.full(C.LEAF_VOLUME,3,np.float32),dtype=float,device=device)
 table=sourcing.wp.array(np.array([0],np.int32),dtype=sourcing.wp.int32,device=device)
 u=sourcing.wp.zeros(C.LEAF_VOLUME,dtype=float,device=device)
 v=sourcing.wp.zeros(C.LEAF_VOLUME,dtype=float,device=device)
 w=sourcing.wp.zeros(C.LEAF_VOLUME,dtype=float,device=device)
 sourcing.wp.launch(sourcing.k_apply_compiled_staggered_source,dim=C.LEAF_VOLUME,
  inputs=[coverage,u_values,v_values,w_values,leaves,1,table,
          sourcing.wp.vec3i(8,8,8),sourcing.wp.vec3i(1,1,1),
          sourcing.wp.vec3(0,0,0),1,int(BlendMode.REPLACE),1.0,1.0],
  outputs=[u,v,w],device=device)
 np.testing.assert_allclose(u.numpy(),1.0)
 np.testing.assert_allclose(v.numpy(),2.0)
 np.testing.assert_allclose(w.numpy(),3.0)
