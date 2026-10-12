"""Engineering probe: wave forcing in native Liquid Warp, same settings in control."""
from pathlib import Path
import bpy, numpy as np, math, os, json, time, traceback, hashlib
ROOT=Path(r'F:\LiquidWarp-CoastalCascade-20261011\ocean_swell_v2')
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'runtime'/'warp_cache'))
import sys
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,r'C:\Users\Tyler\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages')
from tools.repo_addon_loader import enable_repo_addon
enable_repo_addon(str(ROOT/'source'),default_set=False,persistent=False)
from tools.demos.build_demo_scenes import fresh_scene,make_domain,set_resolution,make_particle_seed
from fluidSim_warp_blender51.blender import sources,colliders
from fluidSim_warp_blender51.blender import scene_bridge,playback
SCENARIO=os.environ.get('OCEAN_PROBE_FORCE','1')
OUT=ROOT/'evidence'/('wave_probe_'+('on' if SCENARIO=='1' else 'off')+'.json')
def bed(y,x):
    t=np.clip((y+6)/17,0,1)
    smooth=t*t*(3-2*t)
    return -2.7+1.8*smooth + .13*np.sin(1.1*x+.21*y)*np.exp(-((y-4)/5)**2)
def build_bed(scene):
    xs=np.linspace(-11.75,11.75,55)
    ys=np.linspace(-12,13,64)
    v=[(float(x),float(y),float(bed(y,x))) for y in ys for x in xs]
    f=[]; nx=len(xs);ny=len(ys)
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;f.append((a,a+1,a+1+nx,a+nx))
    boundary=list(range(nx))+[j*nx+nx-1 for j in range(1,ny)]+list(range(ny*nx-2,(ny-1)*nx-1,-1))+[j*nx for j in range(ny-2,0,-1)]
    lower=[]
    for a in boundary:
        lower.append(len(v));v.append((v[a][0],v[a][1],-3.7))
    for i,a in enumerate(boundary):
        j=(i+1)%len(boundary);f.append((a,boundary[j],lower[j],lower[i]))
    f.append(tuple(reversed(lower)))
    m=bpy.data.meshes.new('watertight seabed');m.from_pydata(v,[],f);m.update()
    ob=bpy.data.objects.new('COLLIDER | smooth shoal bathymetry',m);scene.collection.objects.link(ob)
    colliders.tag_collider(ob,friction=.08,restitution=0,transfer_velocity=False)
    return ob
def water_seed(dx):
    xs=np.arange(-10.65,10.65,dx,dtype=np.float32)
    ys=np.arange(-10.65,11.70,dx,dtype=np.float32)
    pts=[]
    for y in ys:
        base=bed(y,xs)
        # Initially prepared cross-bay crest, not a volume jet or river source.
        eta=.25*np.cos((y+5)*2*math.pi/7.5+.10*np.sin(xs*.35))*np.exp(-((y+3)/7)**2)
        low=base+.12
        for z in np.arange(-2.65, .35,dx,dtype=np.float32):
            mask=(z>low)&(z<eta-.07)
            if mask.any():
                a=np.empty((int(mask.sum()),3),dtype=np.float32);a[:,0]=xs[mask];a[:,1]=y;a[:,2]=z
                pts.append(a)
    return np.vstack(pts)
def force(scene):
    bpy.ops.mesh.primitive_cube_add(size=1)
    ob=bpy.context.object;ob.name='WAVEMAKER | full-width zero-mean oscillatory offshore acceleration'
    ob.location=(0,-8.4,-1.3);ob.dimensions=(21.6,2.8,3.4)
    sources.tag_force(ob,shape='BOX')
    pg=ob.fluidsim;pg.force_type='DIRECTIONAL';pg.force_direction=(0,1,0)
    pg.force_strength=0;pg.enabled=(SCENARIO=='1');pg.feather=.2
    for frame in range(1,121):
        pg.force_strength=3.5*math.sin((frame-1)*2*math.pi/(24*1.65))+0.65*math.sin((frame-1)*2*math.pi/(24*2.45)+.4)
        ob.keyframe_insert(data_path='fluidsim.force_strength',frame=frame)
    # Blender 5.2 layered Actions do not expose legacy Action.fcurves.
    # Cubic Bezier interpolation gives a smooth physical acceleration signal.
    pg.force_strength=0
    ob.hide_render=True;ob.display_type='WIRE';ob.hide_set(True)
    return ob
def metrics(scene):
    sess=scene_bridge.session()
    n=int(sess.solver.particles.refresh_host_count())
    pos=sess.solver.particles.position.numpy()[:n]
    bins={}
    # 98th percentile of particle z in a 1.4-m longitudinal bin
    for y in (-8,-5,-2,1,4,7,10):
        m=(abs(pos[:,1]-y)<.7)&(abs(pos[:,0])<6.5)
        z=pos[m,2]
        bins[str(y)]={'n':int(len(z)),'p98':round(float(np.quantile(z,.98)),4) if len(z)>15 else None}
    return {'frame':scene.frame_current,'particles':n,'surface_bins':bins,
            'surface_vertices':len(bpy.data.objects.get('FluidSim Surface').data.vertices) if bpy.data.objects.get('FluidSim Surface') else 0}
def run():
    report={'ok':False,'scenario':SCENARIO,'samples':[],'blender':bpy.app.version_string,'task':str(ROOT)}
    try:
        bpy.context.preferences.use_preferences_save=False
        scene=fresh_scene();scene.name='Engineering | isolated ocean wave driver '+SCENARIO
        scene.gravity=(0,0,-9.81)
        lo=(-12,-12,-3.9);hi=(12,13,4.3)
        dom=make_domain(scene,lo,hi,'DOMAIN | ocean engineering test');ds=dom.fluidsim_domain
        set_resolution(ds,lo,hi,.25)
        ds.max_particles=650000;ds.leaf_capacity=8192;ds.min_substeps=2;ds.max_substeps=12
        ds.pressure_preconditioner='MULTIGRID';ds.pressure_iterations=90
        ds.whitewater_enabled=False;ds.surface_smooth_iterations=0
        ds.open_boundaries=False;ds.domain_sdf_walls=True
        build_bed(scene)
        dx=.25
        pts=water_seed(dx*.5)
        src=make_particle_seed(scene,pts,'SOURCE | non-river initial ocean seed')
        src.fluidsim.role='SOURCE';src.fluidsim.source_ui_schema_version=4;src.fluidsim.enabled=True;src.fluidsim.auto_inject_seed=False;src.fluidsim.source_region_kind='POINTS';src.fluidsim.point_source_representation='DIRECT_APPEND';src.fluidsim.emission_mode='INITIAL_FILL'
        src.fluidsim.initial_velocity=(0,0,0);src.hide_render=True;src.hide_set(True)
        wav=force(scene)
        scene.frame_start=1;scene.frame_end=120;scene.frame_set(1)
        bpy.context.view_layer.update()
        report['initial_particles']=len(pts);report['force_keyed']=int(bool(wav.animation_data))
        report['force_samples']=[]
        for fr in (1,8,16,24,32,40,48):
            scene.frame_set(fr)
            report['force_samples'].append([fr,round(wav.fluidsim.force_strength,4)])
        scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'evidence'/('probe_scene_'+SCENARIO+'.blend')),check_existing=False,compress=True)
        assert bpy.ops.fluidsim.initialize()=={'FINISHED'},'GPU solver admission blocked'
        sess=scene_bridge.session()
        assert 'cuda' in str(sess.solver.device)
        scene.fluidsim.live=True
        for frame in range(2,90):
            scene.frame_set(frame)
            assert sess.active,'solver lost'
            if frame%8==2:
                data=metrics(scene)
                data['wavemaker_strength']=round(wav.fluidsim.force_strength,3)
                report['samples'].append(data)
                print('OCEAN_PROBE '+json.dumps(data),flush=True)
                OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
        report['ok']=True;report['device']=str(sess.solver.device)
    except BaseException as e:
        report['error']=repr(e);report['traceback']=traceback.format_exc();traceback.print_exc()
    finally:
        OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
        if os.environ.get('FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT'):Path(os.environ['FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT']).write_text(json.dumps({'ok':report['ok'],'report':str(OUT)}))
        print('OCEAN-WAVE-PROBE-'+('PASS' if report['ok'] else 'FAIL'),flush=True)
        bpy.context.preferences.use_preferences_save=False
        bpy.context.preferences.view.use_save_prompt=False
        bpy.ops.wm.quit_blender()
if bpy.app.background:
    run()
else:
    bpy.app.timers.register(run,first_interval=.8)
