"""Actual Blender/CUDA ownership and failure controls; no frame certification."""
import argparse,gc,hashlib,json,runpy,sys,traceback,weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import bpy
import warp as wp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fluidSim_warp_blender51.sim import memory_safety,solver as solver_module
from fluidSim_warp_blender51.sim.solver import FluidSolver,SolverSettings

ns=runpy.run_path(str(ROOT/'tests/acceptance/blender_allocation_accounting_acceptance.py'))
run_case=ns['run_case']; inventory=ns['inventory']; memory_sample=ns['memory_sample']
device=wp.get_device('cuda:0')
base=SolverSettings(domain_min=(0.,0.,0.),domain_max=(.8,.8,.8),voxel_size=.1,
    leaf_capacity=1,max_particles=32,max_whitewater=16,surface_enabled=False,device='cuda:0')

def particle_digest(owner):
    values=[]
    for key,value in vars(owner).items():
        if isinstance(value,dict):
            values.extend((key+'.'+str(k),v) for k,v in value.items() if isinstance(v,wp.array))
        elif isinstance(value,wp.array): values.append((key,value))
    digest=hashlib.sha256()
    for key,value in sorted(values):
        digest.update(key.encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def transactional_controls():
    settings=replace(base,sample_type=2,export_attributes=True,particles_attribute_color=True,
                     particles_attribute_source_id=True,particles_attribute_uid=True)
    owner=FluidSolver(settings)
    refs=[]
    result={}
    try:
        owner.particles.position.fill_(wp.vec3(1.,2.,3.))
        owner.particles.count.fill_(2)
        owner.particles.host_count=2
        original=(owner.particles,owner.bins,owner.whitewater.candidates,
                  owner.whitewater.candidate_offsets,owner.whitewater.candidate_types)
        before=particle_digest(owner.particles)
        result['before_memory']=memory_sample(device)
        with patch.object(memory_safety,'vram_snapshot',return_value={
                'gpu_total_bytes':16*1024**3,'gpu_free_bytes':4096}), \
             patch.object(owner.particles,'resized_copy',side_effect=AssertionError('forbidden particle allocation')) as p, \
             patch.object(solver_module,'ParticleBins',side_effect=AssertionError('forbidden bin allocation')) as b, \
             patch.object(owner.whitewater,'allocate_candidate_workspace',side_effect=AssertionError('forbidden candidate allocation')) as c:
            refused=owner.ensure_particle_capacity(33)
            assert refused.status=='BLOCKED',refused
            p.assert_not_called();b.assert_not_called();c.assert_not_called()
        assert all(a is b for a,b in zip(original,(owner.particles,owner.bins,owner.whitewater.candidates,
            owner.whitewater.candidate_offsets,owner.whitewater.candidate_types)))
        assert settings.max_particles==32 and particle_digest(owner.particles)==before
        result['budget_refusal']={'status':'PASS','result':vars(refused),'allocation_calls':0}
        allocate=owner.whitewater.allocate_candidate_workspace
        resize=owner.particles.resized_copy
        bins_type=solver_module.ParticleBins
        def observe_resize(capacity):
            value=resize(capacity);inventory(value,array_refs=refs);return value
        def observe_bins(*args,**kwargs):
            value=bins_type(*args,**kwargs);inventory(value,array_refs=refs);return value
        def fail_after_allocations(capacity):
            value=allocate(capacity);inventory(value,array_refs=refs)
            result['precommit_memory']=memory_sample(device)
            raise RuntimeError('cuda-precommit-control-sentinel')
        with patch.object(owner.particles,'resized_copy',observe_resize), \
             patch.object(solver_module,'ParticleBins',observe_bins), \
             patch.object(owner.whitewater,'allocate_candidate_workspace',fail_after_allocations):
            failed=owner.ensure_particle_capacity(33)
        gc.collect();wp.synchronize_device(device)
        assert failed.status=='FAILED' and 'cuda-precommit-control-sentinel' in failed.reason,failed
        assert refs and not any(ref() is not None for _,ref in refs),'temporary replacement wrappers retained'
        assert all(a is b for a,b in zip(original,(owner.particles,owner.bins,owner.whitewater.candidates,
            owner.whitewater.candidate_offsets,owner.whitewater.candidate_types)))
        assert settings.max_particles==32 and particle_digest(owner.particles)==before
        result['precommit_rollback']={'status':'PASS','result':vars(failed),
                                    'observed_replacement_wrappers':len(refs),'live_after_failure':0}
        result['after_failure_memory']=memory_sample(device)
        retried=owner.ensure_particle_capacity(33)
        assert retried.grew and owner.particles.host_count==2
        assert owner.particles.count.numpy().tolist()==[2]
        assert owner.particles.position.numpy()[:2].tolist()==[[1.,2.,3.],[1.,2.,3.]]
        result['retry']={'status':'PASS','result':vars(retried),'live_count':2,'copied_positions':True}
    finally:
        owner.release()
        owner=original=allocate=resize=None
        gc.collect()
        result['after_release_memory']=memory_sample(device)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--json-out',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    report={'status':'FAIL','device':device.name,'warp_version':wp.__version__,
        'blender_version':bpy.app.version_string,'controls':{},'limits':[
        'Budget refusal and precommit failure are injected faults over real CUDA arrays, not real VRAM exhaustion.',
        'Weakrefs prove wrapper lifetime; pool reservation/free VRAM are separate observations.',
        'No frame advancement or performance acceptance.']}
    retained=[]
    try:
        assert device.is_cuda and '4080 SUPER' in device.name
        report['controls']['transaction']=transactional_controls()
        with patch.dict(run_case.__globals__,estimate_solver_bytes=lambda _:0):
            bad=run_case('initial-undercount-control',replace(base),device)
        assert bad['status']=='FAIL' and 'AssertionError' in bad['error']
        report['controls']['initial_undercount']={'status':'PASS','observed':bad}
        from fluidSim_warp_blender51.sim import particle_capacity
        original_plan=particle_capacity.plan_particle_growth
        def undercount(*a,**kw):return replace(original_plan(*a,**kw),peak_allocation_bytes=0)
        with patch.object(particle_capacity,'plan_particle_growth',undercount):
            bad=run_case('growth-undercount-control',replace(base),device)
        assert bad['status']=='FAIL' and 'AssertionError' in bad['error']
        report['controls']['growth_undercount']={'status':'PASS','observed':bad}
        release=FluidSolver.release
        def leak(owner):retained.append(owner.particles.position);release(owner)
        with patch.object(FluidSolver,'release',leak):
            bad=run_case('owned-leak-control',replace(base),device)
        assert retained and bad['status']=='FAIL' and 'owned Warp arrays retained' in bad['error']
        report['controls']['owned_leak']={'status':'PASS','observed':bad}
        retained.clear();gc.collect()
        report['controls']['warmed_repeats']=[]
        for i in range(3):
            positive=run_case('repeat-'+str(i),replace(base),device)
            report['controls']['warmed_repeats'].append(positive)
            assert positive['status']=='PASS',positive.get('error')
        report['status']='PASS'
    except Exception:report['error']=traceback.format_exc()
    finally:
        retained.clear();gc.collect()
        Path(args.json_out).write_text(json.dumps(report,indent=2))
    print('CUDA-ALLOCATION-CONTROLS-'+report['status'])
    if report['status']!='PASS': print(report.get('error','failed'))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
