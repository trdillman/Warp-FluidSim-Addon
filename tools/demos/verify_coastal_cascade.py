"""Independent cold-process, real-frame-handler, reset and GUI-playback qualification.
Run through tools/gate/launch_profile_safe_addon_review.py, not inside the builder.
This script does not import the scene builder or save any changes to the live file.
"""
from __future__ import annotations
import bpy,os,sys,json,time,hashlib,traceback,math
from pathlib import Path
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# Keep all verification UI preferences session-only, including clean-exit behavior.
bpy.context.preferences.use_preferences_save=False

ROOT=Path(os.environ.get('COASTAL_ROOT',r'F:\LiquidWarp-CoastalCascade-20261011'))
OUT=ROOT/'evidence'/'cold_reopen_verification.json'
LIVE=ROOT/'delivery'/'Coastal_Cascade_LIVE.blend'
# Read the already installed certified Warp dependency without altering its directory.
site=Path(r'C:\Users\Tyler\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages')
if str(site) not in sys.path:sys.path.insert(0,str(site))
from fluidSim_warp_blender51.blender import scene_bridge
import warp as wp
REPORT={'schema':'coastal-independent-reopen-v1','ok':False,'pid':os.getpid(),'blender':bpy.app.version_string,'started_at':time.time(),'checks':{},'frames':[],'reset_frames':[],'errors':[]}
STATE={}

def record():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    temp=OUT.with_suffix('.tmp');temp.write_text(json.dumps(REPORT,indent=2),encoding='utf-8');temp.replace(OUT)

def check(name,condition,detail=''):
    REPORT['checks'][name]=bool(condition)
    if not condition:raise AssertionError(name+(': '+str(detail) if detail else ''))

def frame_state(scene):
    sess=scene_bridge.session();check('solver_active',sess.active,scene_bridge.current_error(scene))
    check('simulation_cuda',bool(wp.get_device(sess.solver.device).is_cuda and sess.solver.particles.position.device.is_cuda))
    count=int(sess.solver.particles.refresh_host_count());check('nonempty_solver',count>0)
    positions=sess.solver.particles.position[:count].numpy()
    check('finite_liquid',bool(np.isfinite(positions).all()))
    surf=bpy.data.objects.get('FluidSim Surface')
    check('published_surface_present',surf is not None and surf.type=='MESH' and len(surf.data.vertices)>0)
    verts=np.empty(len(surf.data.vertices)*3,dtype=np.float32);surf.data.vertices.foreach_get('co',verts)
    check('finite_surface',bool(np.isfinite(verts).all()))
    stats=dict(sess.frame_stats[-1]) if sess.frame_stats else {}
    return {'frame':int(scene.frame_current),'particles':count,'whitewater':int(stats.get('whitewaterParticles') or 0),'whitewater_dropped':int(stats.get('whitewaterDropped') or 0),'position_sha256':hashlib.sha256(positions.tobytes()).hexdigest(),'surface_sha256':hashlib.sha256(verts.tobytes()).hexdigest(),'surface_vertices':len(surf.data.vertices),'bounds':[positions.min(axis=0).tolist(),positions.max(axis=0).tolist()]}

def finish(error=None):
    if error is not None:
        REPORT['errors'].append(repr(error));REPORT['traceback']=traceback.format_exc()
    REPORT['finished_at']=time.time()
    REPORT['ok']=bool(REPORT['checks']) and all(REPORT['checks'].values()) and not REPORT['errors']
    if 'live_sha256_before' in REPORT:
        REPORT['live_sha256_after']=hashlib.sha256(LIVE.read_bytes()).hexdigest()
        REPORT['checks']['live_file_not_modified']=REPORT['live_sha256_before']==REPORT['live_sha256_after']
        REPORT['ok'] &= REPORT['checks']['live_file_not_modified']
    record()
    marker=os.environ.get('FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT')
    if marker:Path(marker).write_text(json.dumps({'ok':REPORT['ok'],'report':str(OUT),'errors':REPORT['errors']}),encoding='utf-8')
    print('COASTAL-COLD-REOPEN-'+('PASS' if REPORT['ok'] else 'FAIL')+' '+str(OUT),flush=True)
    try:
        if bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
        bpy.context.scene.fluidsim.live=False
        scene_bridge.release_session(bpy.context.scene)
    except Exception as exc:
        REPORT['cleanup_notice']=repr(exc);record()
    bpy.context.preferences.view.use_save_prompt=False
    bpy.ops.wm.quit_blender('EXEC_DEFAULT')
    return None

def gui_poll():
    try:
        scene=bpy.context.scene
        STATE['gui_seen_frames'].add(int(scene.frame_current))
        if scene.frame_current>=8:
            bpy.ops.screen.animation_cancel(restore_frame=False)
            final=frame_state(scene);REPORT['gui_final']=final;REPORT['gui_elapsed_seconds']=time.monotonic()-STATE['gui_start']
            REPORT['gui_observed_frames']=sorted(STATE['gui_seen_frames'])
            check('real_gui_playback_advanced',final['frame']>=8 and final['position_sha256']!=STATE['gui_initial_hash'])
            return finish()
        if time.monotonic()-STATE['gui_start']>90:raise RuntimeError('GUI playback did not reach frame 8 within 90 seconds')
        return .15
    except BaseException as exc:return finish(exc)

def run_checks():
    try:
        scene=bpy.context.scene;fs=scene.fluidsim
        check('correct_file',Path(bpy.data.filepath).resolve()==LIVE.resolve())
        REPORT['live_sha256_before']=hashlib.sha256(LIVE.read_bytes()).hexdigest()
        REPORT['module_file']=scene_bridge.__file__;REPORT['warp']=wp.__version__;REPORT['warp_file']=wp.__file__
        check('certified_blender',bpy.app.version[:2]==(5,2));check('certified_warp',wp.__version__=='1.17.0')
        check('frame_one_on_cold_open',scene.frame_current==scene.frame_start==1)
        check('live_enabled_on_cold_open',fs.live)
        check('cold_solver_session',not scene_bridge.session().active)
        domain=fs.domain_object;check('authoritative_domain',domain is not None and domain.name=='DOMAIN | Coastal Cascade')
        REPORT['domain_settings']={'resolution_divisions':int(domain.fluidsim_domain.resolution_divisions),'whitewater_enabled':domain.fluidsim_domain.whitewater_enabled,'max_particles':domain.fluidsim_domain.max_particles}
        check('whitewater_enabled',domain.fluidsim_domain.whitewater_enabled)
        REPORT['profile_paths']={key:os.environ.get(key) for key in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES','BLENDER_USER_EXTENSIONS','WARP_CACHE_PATH','CUDA_CACHE_PATH')}
        check('all_profile_paths_private',all(value and Path(value).resolve().is_relative_to(ROOT.resolve()) for value in REPORT['profile_paths'].values()))
        controls=[ob for ob in scene.objects if ob.name.startswith(('SOURCE |','SINK |','DOMAIN |'))]
        REPORT['controls']=[{'name':o.name,'role':o.fluidsim.role,'enabled':o.fluidsim.enabled,'hidden_viewport':o.hide_get(),'hidden_render':o.hide_render,'seed':o.fluidsim.auto_inject_seed} for o in controls]
        check('controls_hidden_without_disabling',all(o.hide_get() and o.hide_render for o in controls))
        check('one_owned_seed',sum(o.fluidsim.auto_inject_seed for o in controls)==1)
        check('inflow_and_sink_enabled',any(o.fluidsim.role=='SOURCE' and o.fluidsim.enabled for o in controls) and any(o.fluidsim.role=='SINK' and o.fluidsim.enabled for o in controls))
        # Only images referenced by scene materials/world count as delivery dependencies.
        trees=[m.node_tree for ob in scene.objects for m in getattr(ob.data,'materials',()) if m and m.use_nodes]
        if scene.world and scene.world.use_nodes:trees.append(scene.world.node_tree)
        visited=set();images={}
        while trees:
            tree=trees.pop()
            if tree is None or tree.as_pointer() in visited:continue
            visited.add(tree.as_pointer())
            for node in tree.nodes:
                image=getattr(node,'image',None)
                if image is not None:images[image.name]=image
                nested=getattr(node,'node_tree',None)
                if nested is not None:trees.append(nested)
        REPORT['material_images']=[{'name':im.name,'packed':bool(im.packed_file or len(im.packed_files)),'source':im.source,'size':list(im.size)} for im in images.values()]
        check('material_images_self_contained',all(row['packed'] or row['source']=='GENERATED' for row in REPORT['material_images']))
        check('material_images_decoded',all(row['size'][0]>0 and row['size'][1]>0 for row in REPORT['material_images']))
        # Geometry/ray proof is read from the delivered bed, independent of builder functions.
        bed=bpy.data.objects.get('CASCADE | continuous bed collider');check('real_bed_collider',bed is not None and bed.fluidsim.role=='COLLIDER')
        REPORT['bay_bed_heights']={}
        for x,y in ((1.3,4.5),(8.,4.5)):
            inverse=bed.matrix_world.inverted();origin=inverse@Vector((x,y,30));direction=inverse.to_3x3()@Vector((0,0,-1))
            hit,loc,normal,index=bed.ray_cast(origin,direction)
            h=float((bed.matrix_world@loc).z) if hit else None;REPORT['bay_bed_heights'][str((x,y))]=h
            check('open_bay_at_'+str((x,y)),hit and h<1.0,h)
        camera=scene.camera;check('camera_present',camera is not None)
        direction=camera.rotation_euler.to_quaternion()@Vector((0,0,-1));horiz=Vector((direction.x,direction.y,0)).normalized()
        projected=world_to_camera_view(scene,camera,camera.location+horiz*1000)
        REPORT['horizon_y']=float(projected.y);check('horizon_inside_frame',.50<float(projected.y)<.99)
        # Advance every frame through the production handler. No solver.step or fixtures.
        for f in range(2,25):
            t=time.monotonic();scene.frame_set(f);wp.synchronize_device('cuda:0')
            state=frame_state(scene);state['wall_seconds']=time.monotonic()-t;REPORT['frames'].append(state)
            if f%6==0:record()
        check('liquid_changes_between_frames',len({r['position_sha256'] for r in REPORT['frames']})==len(REPORT['frames']))
        check('published_surface_changes',len({r['surface_sha256'] for r in REPORT['frames']})>1)
        check('real_whitewater_emitted',max(r['whitewater'] for r in REPORT['frames'])>0)
        # Evaluate the actual saved Geometry Nodes modifier after fresh frame advancement.
        surf=bpy.data.objects['FluidSim Surface'];ww=bpy.data.objects.get('FluidSim Whitewater')
        mods=[m for m in surf.modifiers if m.type=='NODES' and m.node_group and m.node_group.name.startswith('Coastal |')]
        check('foam_graph_survives_reopen',len(mods)>0)
        infos=[n for m in mods for n in m.node_group.nodes if n.bl_idname=='GeometryNodeObjectInfo']
        check('foam_graph_reads_real_whitewater',any(n.inputs['Object'].default_value==ww for n in infos))
        dg=bpy.context.evaluated_depsgraph_get();evaluated=surf.evaluated_get(dg)
        mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
        try:
            attr=mesh.attributes.get('coastal_foam');check('evaluated_foam_attribute_present',attr is not None)
            values=np.empty(len(attr.data),dtype=np.float32);attr.data.foreach_get('value',values)
            REPORT['foam_shading']={'count':len(values),'positive_vertices':int(np.count_nonzero(values>1e-5)),'min':float(values.min()),'max':float(values.max())}
            check('actual_foam_influences_surface',np.isfinite(values).all() and np.count_nonzero(values>1e-5)>0)
        finally:evaluated.to_mesh_clear()
        REPORT['gpu_name']=wp.get_device(scene_bridge.session().solver.device).name
        scene.frame_set(scene.frame_start)
        for f in (2,3,4):
            scene.frame_set(f);REPORT['reset_frames'].append(frame_state(scene))
        check('reset_repopulates',REPORT['reset_frames'][0]['particles']>0)
        check('reset_advances_again',len({r['position_sha256'] for r in REPORT['reset_frames']})==3)
        scene.frame_set(1);scene.frame_set(2);STATE['gui_initial_hash']=frame_state(scene)['position_sha256']
        scene.sync_mode='NONE';STATE['gui_start']=time.monotonic();STATE['gui_seen_frames']=set()
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':area.spaces.active.shading.type='SOLID'
        result=bpy.ops.screen.animation_play();check('gui_play_operator_started',result=={'FINISHED'} and bpy.context.screen.is_animation_playing)
        bpy.app.timers.register(gui_poll,first_interval=.15);record()
    except BaseException as exc:return finish(exc)
    return None

# Run after Blender's normal UI and frame-handler registration have completed.
bpy.app.timers.register(run_checks,first_interval=.25)
print('COASTAL COLD REOPEN VERIFIER SCHEDULED',os.getpid(),flush=True)