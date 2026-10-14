"""Reproducible Coastal Cascade visual refinement; real GPU water and whitewater.
Run inside the repository profile-safe Blender harness. Offshore water is set dressing.
"""
from __future__ import annotations
import bpy, os, sys, math, json, time, traceback, hashlib, random
import numpy as np
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
ROOT=Path(os.environ.get('COASTAL_ROOT',r'F:\LiquidWarp-CoastalCascade-20261011'))
sys.path.insert(0,str(ROOT/'source'))
from tools.demos import coastal_assets as c
OUT=ROOT/'delivery';EV=ROOT/'evidence';START=time.time()
REPORT={'schema':'coastal-cascade-refinement-v2','source_commit':'0d1d4fa0ce13a7f1b1abe59e7324c115edc3595a','frames':[],'ok':False}

def progress(stage,**kw):
    (EV/'refinement_progress.json').write_text(json.dumps({'stage':stage,'elapsed_seconds':round(time.time()-START,2),**kw},indent=2),encoding='utf-8')
    print('[coastal-v2]',stage,kw,flush=True)

def bed_height(x,y):
    x=np.asarray(x);y=np.asarray(y);ct=c.center(y);w=c.width(y)
    channel=c.bed_level(y)+4.8*c.smooth(np.abs(x-ct),w-.4,w+1.35)
    opening=c.smooth(x,ct+w-2.3,ct+w+.25)
    bay=-1.35+.90*c.smooth(y,-7.5,-5.8)+.07*np.sin(.9*x+.7*y)
    rough=.075*np.sin(1.6*x+.2*y)*np.sin(y*1.4)+.045*np.sin(x*4.1-y*1.7)
    left=c.smooth(ct-x,w-.15,w+2.0)
    ridges=left*(.65*np.sin(.73*x+.31*y)+.38*np.sin(1.7*x-.54*y)+.6*np.sin(y*.22))
    return channel*(1-opening)+bay*opening+rough+ridges

def shoreline_ocean(mat,col):
    # Only geometries OUTSIDE the exact solver AABB. Never overlays foreground liquid.
    for name,x0,x1,y0,y1,nx,ny in [('horizon',-150,260,24.20,280,180,140),('open sea',12.20,220,-12.0,24.20,160,42),('downstream',-80,220,-110,-12.20,120,80)]:
        verts=[];faces=[]
        for j in range(ny):
            y=y0+(y1-y0)*j/(ny-1)
            for i in range(nx):
                x=x0+(x1-x0)*i/(nx-1)
                z=1.22+.12*math.sin(.64*x+.31*y)+.06*math.sin(1.43*x-.92*y)
                if name=='downstream':z=-.90+.10*math.sin(.64*x+.31*y)
                verts.append((x,y,z))
        for j in range(ny-1):
            for i in range(nx-1):a=j*nx+i;faces.append((a,a+1,a+nx+1,a+nx))
        o=c.mesh_obj('SET | offshore '+name+' - NOT simulated',verts,faces,mat,col,True)
        o['provenance']='Non-simulated offshore scenery, entirely outside the Liquid Warp domain.'

def sample_water(dx):
    spacing=dx*.57;xs=np.arange(-18.8,11.65,spacing,dtype=np.float32);ys=np.arange(-10.8,22.05,spacing,dtype=np.float32)
    gx,gy=np.meshgrid(xs,ys,indexing='ij');x=gx.ravel();y=gy.ravel();bottom=bed_height(x,y)
    top=np.maximum(c.bed_level(y)+.83,1.42+.10*np.sin(.8*x+.9*y))
    # Bay has its own level, rather than inheriting the high river terraces.
    opening=c.smooth(x,c.center(y)+c.width(y)-2.3,c.center(y)+c.width(y)+.25)
    top=top*(1-opening)+(1.42+.10*np.sin(.8*x+.9*y))*opening
    valid=(bottom<top-.12)&(x>c.center(y)-c.width(y)+.55)
    x=x[valid];y=y[valid];bottom=bottom[valid];top=top[valid];parts=[]
    for h in np.arange(.13,float((top-bottom).max()),spacing):
        keep=bottom+h<top;pts=np.column_stack((x[keep],y[keep],bottom[keep]+h)).astype(np.float32)
        for loc,axes in c.ROCK_EXCLUSIONS:pts=pts[np.sum(((pts-loc)/axes)**2,axis=1)>1]
        parts.append(pts)
    return np.concatenate(parts).astype(np.float32)

def gpu_render(scene):
    p=bpy.context.preferences.addons['cycles'].preferences;p.compute_device_type='OPTIX';p.get_devices()
    for d in p.devices:d.use=d.type=='OPTIX'
    devices=[d.name for d in p.devices if d.use];assert devices,'OptiX GPU is required'
    scene.render.engine='CYCLES';scene.cycles.device='GPU';scene.cycles.use_denoising=True
    scene.cycles.samples=48;scene.cycles.max_bounces=9;scene.cycles.transmission_bounces=7
    return devices

def surface_foam(scene):
    """Persistent shading only: actual FOAM-class particle proximity, not fake emitted foam."""
    from fluidSim_warp_blender51.sim.whitewater_types import WhitewaterType
    surf=bpy.data.objects.get('FluidSim Surface');ww=bpy.data.objects.get('FluidSim Whitewater')
    assert surf is not None and ww is not None,'Production output is required'
    group=bpy.data.node_groups.new('Coastal | real FOAM particle proximity','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry');group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    n=group.nodes;l=group.links;gin=n.new('NodeGroupInput');gout=n.new('NodeGroupOutput')
    obj=n.new('GeometryNodeObjectInfo');obj.transform_space='RELATIVE';obj.inputs['Object'].default_value=ww
    r=n.new('GeometryNodeInputRadius');positive=n.new('ShaderNodeMath');positive.operation='GREATER_THAN';positive.inputs[1].default_value=.00001;l.new(r.outputs[0],positive.inputs[0])
    typ=n.new('GeometryNodeInputNamedAttribute');typ.data_type='INT';typ.inputs['Name'].default_value='fluidsim_whitewater_type'
    eq=n.new('ShaderNodeMath');eq.operation='COMPARE';eq.inputs[1].default_value=int(WhitewaterType.FOAM);eq.inputs[2].default_value=.01;l.new(typ.outputs['Attribute'],eq.inputs[0])
    both=n.new('ShaderNodeMath');both.operation='MULTIPLY';l.new(positive.outputs[0],both.inputs[0]);l.new(eq.outputs[0],both.inputs[1])
    sep=n.new('GeometryNodeSeparateGeometry');sep.domain='POINT';l.new(obj.outputs['Geometry'],sep.inputs['Geometry']);l.new(both.outputs[0],sep.inputs['Selection'])
    near=n.new('GeometryNodeSampleNearest');near.domain='POINT';pos=n.new('GeometryNodeInputPosition');l.new(sep.outputs['Selection'],near.inputs['Geometry']);l.new(pos.outputs[0],near.inputs['Sample Position'])
    sample=n.new('GeometryNodeSampleIndex');sample.data_type='FLOAT_VECTOR';sample.domain='POINT';l.new(sep.outputs['Selection'],sample.inputs['Geometry']);l.new(pos.outputs[0],sample.inputs['Value']);l.new(near.outputs['Index'],sample.inputs['Index'])
    dist=n.new('ShaderNodeVectorMath');dist.operation='DISTANCE';l.new(pos.outputs[0],dist.inputs[0]);l.new(sample.outputs[0],dist.inputs[1])
    remap=n.new('ShaderNodeMapRange');remap.inputs['From Min'].default_value=.01;remap.inputs['From Max'].default_value=.115;remap.inputs['To Min'].default_value=1;remap.inputs['To Max'].default_value=0;l.new(dist.outputs['Value'],remap.inputs[0])
    store=n.new('GeometryNodeStoreNamedAttribute');store.domain='POINT';store.data_type='FLOAT';store.inputs['Name'].default_value='coastal_foam';l.new(gin.outputs['Geometry'],store.inputs['Geometry']);l.new(remap.outputs['Result'],store.inputs['Value']);l.new(store.outputs[0],gout.inputs[0])
    for i,node in enumerate(n):node.location=((i%6)*220,-(i//6)*260)
    mod=surf.modifiers.new('Real whitewater-derived surface foam','NODES');mod.node_group=group
    mat=scene.fluidsim.surface_material;n=mat.node_tree.nodes;l=mat.node_tree.links
    att=n.new('ShaderNodeAttribute');att.attribute_name='coastal_foam'
    tex=c.noise_tex(n,34,3);tc=n.new('ShaderNodeTexCoord');l.new(tc.outputs['Object'],tex.inputs['Vector'])
    mul=n.new('ShaderNodeMath');mul.operation='MULTIPLY';l.new(att.outputs['Fac'],mul.inputs[0]);l.new(tex.outputs['Fac'],mul.inputs[1]);strength=n.new('ShaderNodeMath');strength.operation='MULTIPLY';strength.inputs[1].default_value=.45;l.new(mul.outputs[0],strength.inputs[0])
    diff=n.new('ShaderNodeBsdfDiffuse');diff.inputs['Color'].default_value=(.87,.95,1,1);diff.inputs['Roughness'].default_value=.45
    mix=n.new('ShaderNodeMixShader');l.new(strength.outputs[0],mix.inputs[0]);l.new(n.get('Principled BSDF').outputs[0],mix.inputs[1]);l.new(diff.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],n.get('Material Output').inputs['Surface'])
    scene['foam_provenance']='Surface shading derives only from active, FOAM-class Liquid Warp particles. Fine noise breaks up coverage, not fluid motion.'

def read_state(scene):
    from fluidSim_warp_blender51.blender import scene_bridge
    sess=scene_bridge.session();assert sess.active,'Inactive runtime'
    count=int(sess.solver.particles.refresh_host_count());assert count>0,'Empty runtime'
    a=sess.solver.particles.position.numpy()[:count]
    assert np.isfinite(a).all(),'Nonfinite particle coordinates'
    stats=dict(sess.frame_stats[-1]) if sess.frame_stats else {}
    surface=bpy.data.objects.get('FluidSim Surface')
    return {'frame':scene.frame_current,'particles':count,'whitewater':int(stats.get('whitewaterParticles') or 0),'whitewater_dropped':int(stats.get('whitewaterDropped') or 0),'position_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'surface_vertices':len(surface.data.vertices) if surface else 0,'finite':True,'device':str(sess.solver.device)}

def build():
    for key in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES','BLENDER_USER_EXTENSIONS'):assert os.environ.get(key,'').startswith(str(ROOT)),key
    progress('building open coastal bay')
    scene=c.fresh_scene();scene.name='Coastal Cascade | Liquid Warp';scene['coastal_cascade_demo']=True
    c.COLS.clear();c.ROCK_EXCLUSIONS.clear();random.seed(27011)
    sim=c.collection(c.SIM_COLLECTION);env=c.collection('02 | SET \u2014 cliffs and offshore ocean');plants=c.collection('03 | SET \u2014 coastal conifers')
    basalt=c.rock_material();water=c.water_material('Water | turquoise cascade - SIMULATED');sea=c.water_material('Water | distant sea - SET',True)
    foam,bs=c.mat_principled('Whitewater | actual solver foam and spray',(.87,.95,1),.42)
    c.bed_height=bed_height;bed=c.build_bed(basalt,sim);c.build_set(scene,basalt,env,plants)
    # Render the same top surface, but not the artificial closed collider box walls.
    bed.hide_render=True
    verts=[tuple(v.co) for v in bed.data.vertices]
    faces=[tuple(p.vertices) for p in bed.data.polygons if p.normal.z>.05 and p.center.z>-2.9]
    visual=c.mesh_obj('SET | same bed surface without artificial boundary walls',verts,faces,basalt,env,True)
    visual['provenance']='Exact top faces of the closed simulation bed. Boundary caps hidden from camera.'
    # Remove the erroneous right enclosing bank and its unnecessary decorations.
    for ob in list(scene.objects):
        if ob.name.startswith(('SET | basalt headland 1 ','SET | fractured ledge 1 ')) and -3.5<ob.location.y<25:bpy.data.objects.remove(ob,do_unlink=True)
        elif ob.name.startswith('SET | distant coastal ridge '):
            if int(ob.name.rsplit(' ',1)[1])>9:bpy.data.objects.remove(ob,do_unlink=True)
    for i,(x,y,rx,ry,h) in enumerate([(8,2,1.4,1.8,1.9),(9,9,1.25,1.4,2.0),(4,12,.75,1.1,1.45),(-.5,.8,.55,.65,.9),(3,-3,.65,.8,.8)]):
        c.rock('COLLIDER | bay breaker '+str(i),(x,y,-.5+h*.32),(rx,ry,h),basalt,sim,seed=700+i,collision=True,sub=3)
    # The reference's right-hand headland and broken offshore reefs conceal domain edges,
    # while remaining real collisions where they intersect the simulation.
    reef_specs=[(11.9,-.2,2.4,3.3,2.0),(12.5,4.8,2.9,3.8,2.8),(12.5,10.0,2.7,3.1,2.4),(11.9,15.6,2.3,3.0,1.7),(10.3,21.3,2.6,2.8,1.6),(6.0,23.1,2.7,1.7,1.25),(2.3,23.5,1.8,1.6,1.10)]
    for i,(x,y,rx,ry,h) in enumerate(reef_specs):
        c.rock('COLLIDER | coastal headland reef '+str(i),(x,y,.20),(rx,ry,h),basalt,sim,seed=1400+i,collision=True,sub=3)
        for j in range(4):
            c.rock('SET | fractured reef detail '+str(i)+'.'+str(j),(x+random.uniform(-rx*.8,rx*.8),y+random.uniform(-ry*.8,ry*.8),.20+h*.3),(rx*.45,ry*.43,h*.50),basalt,env,seed=1800+i*7+j,sub=2)
    hero=bpy.data.objects['COLLIDER | tree-topped sea stack']
    for v in hero.data.vertices:v.co.x*=1.55;v.co.y*=1.55
    c.ROCK_EXCLUSIONS[:]=[(loc,axes*np.array((1.55,1.55,1))) if np.linalg.norm(loc-np.array((1.3,4.5,1.8)))<.01 else (loc,axes) for loc,axes in c.ROCK_EXCLUSIONS]
    shoreline_ocean(sea,env);c.lighting(scene)
    light=bpy.data.lights.new('LIGHT | warm rock grazing accent','AREA');light.energy=4500;light.color=(1,.58,.27);light.shape='DISK';light.size=8
    lamp=bpy.data.objects.new(light.name,light);scene.collection.objects.link(lamp);lamp.location=(-18,-9,12);c.look_at(lamp,(0,5,1))
    # Rock strata and rough fractured silhouettes at the hero distance.
    noise=bpy.data.textures.new('Basalt fracturing','CLOUDS');noise.noise_scale=.35;noise.noise_depth=2
    for ob in scene.objects:
        if ob.type=='MESH' and ob.data.materials and ob.data.materials[0]==basalt and ob.name.startswith(('SET | basalt','COLLIDER | tree','SET | near')):
            sub=ob.modifiers.new('Basalt surface tessellation','SUBSURF');sub.subdivision_type='SIMPLE';sub.levels=1;sub.render_levels=2
            dis=ob.modifiers.new('Coastal fractures','DISPLACE');dis.texture=noise;dis.texture_coords='GLOBAL';dis.strength=.18
    for ramp in [n for n in basalt.node_tree.nodes if n.type=='VALTORGB']:
        for e,col in zip(sorted(ramp.color_ramp.elements,key=lambda e:e.position),[(.005,.009,.013,1),(.026,.035,.040,1),(.12,.12,.106,1)]):e.color=col
    p=water.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(.035,.36,.46,1);p.inputs['Transmission Weight'].default_value=.95;p.inputs['Roughness'].default_value=.085
    scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.105
    bpy.data.lights['LIGHT | sunset rim'].energy=1.65;bpy.data.lights['LIGHT | blue overcast softbox'].energy=11000
    # Layered storm clouds in world coordinates; no generated picture or background plate.
    n=scene.world.node_tree.nodes;l=scene.world.node_tree.links;bg=n.get('Background');sky=next(node for node in n if node.type=='TEX_SKY')
    tex=c.noise_tex(n,3.7,6);tc=n.new('ShaderNodeTexCoord');l.new(tc.outputs['Normal'],tex.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.30;ramp.color_ramp.elements[0].color=(.075,.12,.20,1);ramp.color_ramp.elements[1].position=.72;ramp.color_ramp.elements[1].color=(.8,.85,.93,1);l.new(tex.outputs['Fac'],ramp.inputs[0])
    mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;l.new(sky.outputs[0],mix.inputs[1]);l.new(ramp.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],bg.inputs[0])
    c.configure_camera(scene);cam=scene.camera;cam.location=(6,-13.5,7.7);c.look_at(cam,(-.5,7,2.7));cam.data.lens=27;cam.data.shift_y=0
    scene.view_settings.exposure=.5;scene.view_settings.look='AgX - Medium High Contrast'
    gpu=gpu_render(scene);scene.render.resolution_x=1600;scene.render.resolution_y=900
    lo=(-20,-12,-2.5);hi=(12,24,13);dom=c.make_domain(scene,lo,hi,'DOMAIN | Coastal Cascade');c.move_to(dom,sim);ds=dom.fluidsim_domain
    divisions=int(os.environ.get('COASTAL_DIVISIONS','240'));dx=36/divisions;c.set_resolution(ds,lo,hi,dx)
    ds.max_particles=3000000;ds.leaf_capacity=12288;ds.min_substeps=2;ds.max_substeps=32;ds.pressure_preconditioner='MULTIGRID';ds.pressure_iterations=200
    ds.whitewater_enabled=True;ds.max_whitewater=1500000;ds.ww_churn_gain=2;ds.ww_churn_threshold=.20;ds.ww_emission_rate=.65
    ds.ww_min_speed=.45;ds.ww_life_min=.8;ds.ww_life_max=2.2;ds.ww_surface_band=2;ds.ww_foam_band=1.2
    ds.surface_smooth_iterations=1;ds.surface_smooth_strength=.55
    fs=scene.fluidsim;fs.live=False;fs.liquid_view_mode='SURFACE';fs.show_whitewater=True;fs.surface_material=water;fs.whitewater_material=foam;fs.whitewater_display_scale=.45
    fs.whitewater_display_fraction=1;scene.frame_start=1;scene.frame_end=192;scene.frame_set(1);scene.sync_mode='NONE'
    pts=sample_water(dx);assert 0<len(pts)<2800000,len(pts)
    seed=c.make_particle_seed(scene,pts,'SOURCE | initial cascade and open bay - owned seed');c.move_to(seed,sim);seed.fluidsim.enabled=True;seed.fluidsim.auto_inject_seed=True;seed.fluidsim.initial_velocity=(.7,-2.7,0)
    cx=float(c.center(21.3));h=float(c.bed_level(21.3))
    src=c.make_source_box(scene,(cx-2.65,20.9,h+.15),(cx+2.65,22,h+.95),'SOURCE | continuous mountain inflow',emit_rate=8,emission_mode='CONTINUOUS_INFLOW');c.move_to(src,sim);src.fluidsim.initial_velocity=(1.1,-3.8,0)
    from fluidSim_warp_blender51.blender import sources,scene_bridge
    bpy.ops.mesh.primitive_cube_add(size=1,location=(-3,-11.35,4));sink=bpy.context.object;sink.name='SINK | open sea outlet';sink.dimensions=(34,.95,20);sources.tag_sink(sink);c.move_to(sink,sim)
    for ob in (dom,seed,src,sink):ob.hide_render=True;ob.display_type='WIRE';ob.hide_set(True)
    from tools.demos.coastal_finishing import enhance_scene
    enhance_scene(scene,ROOT)
    bpy.context.view_layer.update();progress('initializing GPU simulation',initial_particles=len(pts),divisions=divisions)
    assert bpy.ops.fluidsim.initialize()=={'FINISHED'}
    sess=scene_bridge.session()
    assert bpy.ops.fluidsim.step(frames=1)=={'FINISHED'}  # Publish a real starting-frame surface, not an empty bowl
    surface_foam(scene);fs.live=True;scene['fluidsim_playback_ready']=True
    doc=bpy.data.texts.get('README | COASTAL CASCADE') or bpy.data.texts.new('README | COASTAL CASCADE');doc.clear();doc.write('COASTAL CASCADE - LIVE LIQUID WARP\n\nPress Space from frame 1. Return to frame 1 to reset.\nThe visible cascade, open bay surface, foam and spray are real GPU Liquid Warp outputs.\nThe owned hidden initial seed, continuing inlet and sink remain active.\nThe pine-topped sea stack, river bed and named breakers are real colliders.\nOffshore meshes outside the domain, trees, cliffs and sky are scenery.\nSurface foam appearance derives from actual FOAM-class particle proximity.\n\nThe separate RENDER_STATE file is a frozen solver frame, not the live demo.\nBlender 5.2.0 / Warp 1.17.0 / Liquid Warp source 0d1d4fa.\n')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':a=area.spaces.active;a.region_3d.view_perspective='CAMERA';a.region_3d.view_camera_zoom=20;a.overlay.show_overlays=False;a.shading.type='MATERIAL'
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'Coastal_Cascade_LIVE.blend'),check_existing=False,compress=True)
    REPORT.update(blender=bpy.app.version_string,initial_particles=len(pts),divisions=divisions,gpu=gpu,voxel_size=dx,live_file=str(OUT/'Coastal_Cascade_LIVE.blend'))
    REPORT['frames'].append(read_state(scene));times=[]
    for frame in range(2,int(os.environ.get('COASTAL_FRAMES','72'))+1):
        t=time.perf_counter();scene.frame_set(frame);stat=read_state(scene);stat['wall_seconds']=time.perf_counter()-t;times.append(stat['wall_seconds']);REPORT['frames'].append(stat)
        if frame%6==0:progress('advancing real frame handlers',**stat)
    fs.live=False;REPORT['median_frame_seconds']=float(np.median(times[2:]));REPORT['simulation_device']=str(scene_bridge.session().solver.device)
    for ob in scene.objects:
        if ob.name=='FluidSim Surface':
            for poly in ob.data.polygons:poly.use_smooth=True
    REPORT['render_frame']=scene.frame_current;REPORT['ok']=True
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'Coastal_Cascade_RENDER_STATE.blend'),check_existing=False,compress=True)
    scene.render.resolution_percentage=60;scene.cycles.samples=32;scene.render.filepath=str(OUT/'Coastal_Cascade_preview.png');progress('rendering camera preview')
    from fluidSim_warp_blender51.blender import playback
    with playback.suppress_frame_handlers():bpy.ops.render.render(write_still=True)
    (EV/'refinement_report.json').write_text(json.dumps(REPORT,indent=2),encoding='utf-8')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.shading.type='RENDERED'
    progress('preview ready',file=scene.render.filepath,render_frame=scene.frame_current)

if __name__=='__main__':
    try:build()
    except BaseException as exc:
        REPORT['ok']=False;REPORT['error']=repr(exc);REPORT['traceback']=traceback.format_exc();(EV/'refinement_failure.json').write_text(json.dumps(REPORT,indent=2));progress('FAILED',error=repr(exc));raise