"""Independent cold-open test for wave scene in a fresh isolated Blender 5.2 process."""
from pathlib import Path
import bpy,os,sys,json,traceback,time,hashlib
import numpy as np
ROOT=Path(r'F:\LiquidWarp-CoastalCascade-20261011\ocean_swell_v2')
REPO=ROOT/'source'
sys.path.insert(0,str(REPO))
sys.path.insert(0,r'C:\Users\Tyler\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages')
bpy.context.preferences.use_preferences_save=False
from tools.repo_addon_loader import enable_repo_addon
enable_repo_addon(str(REPO),default_set=False,persistent=False)
from fluidSim_warp_blender51.blender import scene_bridge
LIVE=ROOT/'delivery'/'Ocean_Swell_LIVE.blend'
REPORT={'ok':False,'checks':{},'sample_frames':[],'errors':[]}
def check(key,condition,details=None):
    REPORT['checks'][key]=bool(condition)
    if not condition:raise AssertionError(key+(': '+str(details) if details is not None else ''))
def stage(frame):
    scene=bpy.context.scene;scene.frame_set(frame)
    sess=scene_bridge.session()
    check('cuda_device',sess.active and 'cuda' in str(sess.solver.device))
    count=int(sess.solver.particles.refresh_host_count())
    check('nonempty_ocean',count>1000,count)
    pts=sess.solver.particles.position.numpy()[:count]
    check('finite_positions',bool(np.isfinite(pts).all()))
    surf=bpy.data.objects.get('FluidSim Surface')
    check('reconstructed_surface',bool(surf and len(surf.data.vertices)>3000))
    stats=dict(sess.frame_stats[-1]) if sess.frame_stats else {}
    measure={'frame':frame,'count':count,'foam':int(stats.get('whitewaterParticles') or 0),
             'surface_vertices':len(surf.data.vertices),'region_crest':{}}
    for y in (-12,-8,-4,0,4,8):
        mask=(abs(pts[:,1]-y)<.9)&(abs(pts[:,0])<10)
        heights=pts[mask,2]
        measure['region_crest'][str(y)]=float(np.quantile(heights,.98)) if len(heights)>100 else None
    REPORT['sample_frames'].append(measure)
    print('OCEAN-COLD-METRIC '+json.dumps(measure),flush=True)
    return measure
def run():
    started=time.time()
    before=hashlib.sha256(LIVE.read_bytes()).hexdigest()
    try:
        check('correct_blend',Path(bpy.data.filepath).resolve()==LIVE.resolve())
        check('version_52',bpy.app.version>=(5,2,0))
        fs=bpy.context.scene.fluidsim;scene=bpy.context.scene
        check('authored_ocean_tag',scene.get('ocean_swell_v2') is True)
        check('saved_frame_one',scene.frame_current==1)
        check('saved_live',fs.live)
        dom=fs.domain_object
        check('real_domain',bool(dom and dom.name=='DOMAIN | full-width ocean nearshore'))
        seabed=bpy.data.objects.get('COLLIDER | watertight ocean bathymetry')
        check('seabed_collider',bool(seabed and seabed.fluidsim.role=='COLLIDER' and seabed.fluidsim.enabled))
        stack=bpy.data.objects.get('COLLIDER | pine-topped center sea stack')
        check('hero_collision',bool(stack and stack.fluidsim.role=='COLLIDER' and stack.fluidsim.enabled))
        inject=[o for o in scene.objects if getattr(o,'fluidsim',None) and o.fluidsim.role=='SOURCE' and o.fluidsim.enabled]
        check('no_river_or_steady_source',len(inject)==1,[o.name for o in inject])
        if inject:
            p=inject[0].fluidsim
            check('owned_one_shot_direct_point_seed',p.source_ui_schema_version>=4 and p.source_region_kind=='POINTS' and p.point_source_representation=='DIRECT_APPEND' and p.emission_mode=='INITIAL_FILL' and not p.auto_inject_seed)
        maker=bpy.data.objects.get('WAVEMAKER | 37m oscillatory deep-ocean swell driver')
        check('zero_mean_animated_force_region',bool(maker and maker.fluidsim.role=='FORCE' and maker.fluidsim.enabled and maker.animation_data and maker.fluidsim.force_type=='DIRECTIONAL'))
        if maker:
            fs.live=False
            strengths=[]
            for f in (8,20,32,44):
                scene.frame_set(f);strengths.append(round(float(maker.fluidsim.force_strength),3))
            check('oscillating_positive_negative_strength',max(strengths)>2 and min(strengths)<-2,strengths)
            scene.frame_set(1);fs.live=True
            REPORT['driver_values']=strengths
        check('whitewater_enabled',dom.fluidsim_domain.whitewater_enabled)
        check('no_external_simulated_ocean_mask',all('ocean' not in o.name.lower() or not (o.fluidsim.role=='SOURCE' and 'initial' not in o.name.lower()) for o in inject))
        images={n.image for ob in scene.objects if getattr(ob.data,'materials',None)
                for m in ob.data.materials if m and m.use_nodes for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image}
        check('used_scan_images_packed',all(img.packed_file for img in images))
        for k in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES','BLENDER_USER_EXTENSIONS'):
            check('private_'+k,Path(os.environ[k]).resolve().is_relative_to(ROOT.resolve()))
        points=[]
        for frame in range(2,87):
            scene.frame_set(frame)
            if frame in (2,18,34,50,66,86):
                points.append(stage(frame))
        check('particle_mass_no_continuous_growth',max(p['count'] for p in points)<=min(p['count'] for p in points)*1.015,
              [p['count'] for p in points])
        crests=[]
        for entry in points:
            crests.append(entry['region_crest']['-8'])
        check('changing_wave_elevation',max(crests)-min(crests)>.09,crests)
        check('whitewater_breaking',max(p['foam'] for p in points)>0,[p['foam'] for p in points])
        background=[o for o in scene.objects if o.name.startswith('SET | distant horizon ocean')]
        offshore_max_y=max((o.matrix_world @ v.co).y for o in background for v in o.data.vertices) if background else -1e6
        check('scenic_ocean_far_outside_hero',bool(background and offshore_max_y<=-16.50),offshore_max_y)
        physical=bpy.data.objects.get('FluidSim Surface')
        check('foreground_solver_surface_renderable',bool(physical and not physical.hide_render))
        fs.live=True;scene.frame_set(1);scene.frame_set(2)
        check('frame1_reset_reseeds',int(scene_bridge.session().solver.particles.refresh_host_count())>=points[0]['count']*.90)
        check('cold_file_unchanged',hashlib.sha256(LIVE.read_bytes()).hexdigest()==before)
        check('scene_not_river',not any(o.name.startswith('SOURCE | continuous') for o in scene.objects))
        REPORT['live_sha256']=before
        REPORT['ok']=all(REPORT['checks'].values())
    except BaseException as exc:
        REPORT['errors'].append(repr(exc));REPORT['traceback']=traceback.format_exc();traceback.print_exc()
    finally:
        REPORT['elapsed_s']=round(time.time()-started,1)
        (ROOT/'evidence'/'ocean_cold_reopen.json').write_text(json.dumps(REPORT,indent=2),encoding='utf8')
        if REPORT['ok']:print('OCEAN-SWELL-COLD-PASS',flush=True)
        else:print('OCEAN-SWELL-COLD-FAIL',flush=True)
        bpy.context.preferences.use_preferences_save=False
        bpy.context.preferences.view.use_save_prompt=False
        bpy.ops.wm.quit_blender()
if bpy.app.background:run()
else:bpy.app.timers.register(run,first_interval=.7)
