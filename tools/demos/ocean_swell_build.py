"""Open-ocean swell and rocky cove. Rebuilds via native CUDA Liquid Warp, no river source.
Never reroute the background Ocean modifier into the visible solver domain.
"""
from __future__ import annotations
from pathlib import Path
import os,sys,math,random,time,traceback,hashlib,json
import numpy as np
import bpy
from mathutils import Vector,Matrix,Euler
ROOT=Path(r'F:\LiquidWarp-CoastalCascade-20261011\ocean_swell_v2')
REPO=ROOT/'source';DEL=ROOT/'delivery';EVIDENCE=ROOT/'evidence'
sys.path.insert(0,str(REPO))
sys.path.insert(0,r'C:\Users\Tyler\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages')
from tools.repo_addon_loader import enable_repo_addon
enable_repo_addon(str(REPO),default_set=False,persistent=False)
from tools.demos.build_demo_scenes import fresh_scene,make_domain,set_resolution,make_particle_seed
from tools.demos import coastal_utilities as c
from fluidSim_warp_blender51.blender import sources,colliders,scene_bridge,playback
from fluidSim_warp_blender51.sim.whitewater_types import WhitewaterType
random.seed(7911)
np.random.seed(7911)
EVENTS=[]
def progress(step,**details):
    entry={'step':step,'wall_utc':time.time(),**details};EVENTS.append(entry)
    (EVIDENCE/'build_progress.json').write_text(json.dumps({'events':EVENTS},indent=2),encoding='utf8')
    print('OCEAN_SCENE_PROGRESS '+json.dumps(entry),flush=True)
def smooth(v,a,b):
    t=np.clip((np.asarray(v)-a)/(b-a),0,1)
    return t*t*(3-2*t)
def bathy(x,y):
    """Offshore deep basin -> coherent shoal -> broken reef field -> shore. No channel."""
    x,y=np.broadcast_arrays(np.asarray(x),np.asarray(y))
    deep=-3.60+2.45*smooth(y,-13,7)
    # Irregular submerged ridges: uneven crest tripping, not a chute.
    shoal=.85*np.exp(-((y-1.1-.95*np.sin(x*.24))/2.15)**2)*(0.52+.44*np.cos(.21*x+.6)**2)
    reef=.53*np.exp(-((y+2.4+.5*np.cos(.35*x))/1.30)**2)*(.28+.72*np.cos(.52*x)**2)
    relief=.12*np.sin(.49*x+1.2*y)*np.cos(.72*y-.4*x)
    # Dry headland on nearshore left, broken right reef leaving most of the bay open.
    left=(6.1+1.0*np.sin(y*.32))*smooth(-x,10.4,15.6)*smooth(y,1.0,7.7)
    right=(2.55+.35*np.cos(y*.51))*smooth(x,13.5,18.6)*smooth(y,2,10)
    shore=2.8*smooth(y,9.5,15.0)
    return deep+shoal+reef+relief+left+right+shore
def make_bathymetry(scene,rock_mat,sim,scenic):
    xs=np.linspace(-19.7,19.7,145,dtype=np.float32)
    ys=np.linspace(-17.7,15.7,130,dtype=np.float32)
    xx,yy=np.meshgrid(xs,ys,indexing='xy');zz=bathy(xx,yy)
    nv=xx.size; vertices=list(zip(xx.ravel().astype(float),yy.ravel().astype(float),zz.ravel().astype(float)))
    faces=[];topfaces=[];nx=len(xs);ny=len(ys)
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;faces.append((a,a+1,a+nx+1,a+nx))
    topfaces=list(faces)
    ring=list(range(nx))+[j*nx+nx-1 for j in range(1,ny)]+list(range(ny*nx-2,(ny-1)*nx-1,-1))+[j*nx for j in range(ny-2,0,-1)]
    lower=[]
    for a in ring:
        lower.append(len(vertices));vertices.append((vertices[a][0],vertices[a][1],-4.6))
    for k,a in enumerate(ring):
        j=(k+1)%len(ring);faces.append((a,ring[j],lower[j],lower[k]))
    faces.append(tuple(reversed(lower)))
    col=c.mesh_obj('COLLIDER | watertight ocean bathymetry',vertices,faces,rock_mat,sim,True)
    colliders.tag_collider(col,friction=.10,restitution=0.0,transfer_velocity=False)
    col.hide_render=True;col.display_type='WIRE';col.hide_set(True)
    # Visual shell shares the exact top surface, but omits rectangular side walls.
    visual=c.mesh_obj('SET | genuine exposed seabed / reef surface',vertices[:nv],topfaces,rock_mat,scenic,True)
    for poly in visual.data.polygons:poly.use_smooth=True
    return col,visual
def swell_elevation(x,y):
    """Two long-crested nonstationary packets, used ONLY for initial fluid seed."""
    phase0=2*math.pi*(y+8-.12*np.sin(x*.22))/9.5
    phase1=2*math.pi*(y+1+.23*np.sin(x*.19))/6.8
    return 1.03*np.cos(phase0)*np.exp(-((y+5.7)/10)**2)+.25*np.cos(phase1)*np.exp(-((y+.4)/7.5)**2)
def ocean_seed(spacing):
    xs=np.arange(-18.5,18.51,spacing,dtype=np.float32)
    ys=np.arange(-16.4,14.45,spacing,dtype=np.float32)
    xyz=[]
    exclusions=list(c.ROCK_EXCLUSIONS)
    for y in ys:
        depth=bathy(xs,y)
        eta=swell_elevation(xs,y)
        floor=depth+.10
        for z in np.arange(-3.52,1.42,spacing,dtype=np.float32):
            mask=(z>floor)&(z<eta-.075)
            if not mask.any():continue
            for loc,axes in exclusions:
                if abs(y-loc[1])>axes[1]*1.5 or abs(z-loc[2])>axes[2]*1.5:continue
                mask &= ((xs-loc[0])/max(axes[0],.01))**2+((y-loc[1])/max(axes[1],.01))**2+((z-loc[2])/max(axes[2],.01))**2>1.0
            if mask.any():
                a=np.empty((int(mask.sum()),3),dtype=np.float32)
                a[:,0]=xs[mask];a[:,1]=y;a[:,2]=z;xyz.append(a)
    return np.concatenate(xyz,axis=0)
def make_wavemaker(sim,frames=240):
    """Native analytic BOX FORCE; zero mean *acceleration*, no net liquid injection."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    ob=bpy.context.object;ob.name='WAVEMAKER | 37m oscillatory deep-ocean swell driver'
    ob.location=(0,-14.5,-1.40);ob.dimensions=(37.0,3.15,4.1)
    c.move_to(ob,sim)
    sources.tag_force(ob,shape='BOX')
    pg=ob.fluidsim;pg.force_type='DIRECTIONAL';pg.force_direction=(0,1,0)
    pg.feather=.18;pg.enabled=True
    pg.force_strength=0
    for f in range(1,frames+1):
        t=(f-1)/24
        pg.force_strength=4.4*math.sin(math.tau*t/1.82)+.85*math.sin(math.tau*t/2.63+.34)
        ob.keyframe_insert(data_path='fluidsim.force_strength',frame=f)
    scene=bpy.context.scene
    scene.frame_set(1);ob.hide_render=True;ob.hide_set(True);ob.display_type='WIRE'
    return ob
def tree_variants(foliage,bark):
    vars=[]
    for i in range(5):
        v,f,mids=c.pine_mesh(321+i)
        mesh=bpy.data.meshes.new('Wind-twisted coast pine %s'%i)
        mesh.from_pydata(v,[],f);mesh.materials.append(bark);mesh.materials.append(foliage)
        mesh.polygons.foreach_set('material_index',mids);mesh.update();vars.append(mesh)
    return vars
def make_pines(scene,plants):
    foliage,_=c.mat_principled('Pine needles | near-black deep spruce',(.017,.038,.029),.82)
    bark,_=c.mat_principled('Pine bark | weathered cold brown',(.073,.053,.034),.86)
    meshes=tree_variants(foliage,bark);rng=random.Random(710)
    placements=[]
    for i in range(95):
        x=rng.uniform(-20.8,-11.1);y=rng.uniform(4.4,18.5)
        zz=float(bathy(x,min(y,15.7)))
        if zz<.65:continue
        placements.append((x,y,zz-.12,rng.uniform(1.65,3.8)))
    for i,(x,y,z,h) in enumerate([(2.25,3.4,6.4,1.9),(3.1,4.0,6.3,1.4),(1.85,4.4,6.5,1.4),
                                     (-14,8.2,7.0,2.3),(-15,9.5,7.2,2.5)]):placements.append((x,y,z,h))
    for i,(x,y,z,h) in enumerate(placements):
        ob=bpy.data.objects.new('PINE | coastal fir '+str(i).zfill(3),meshes[i%len(meshes)])
        plants.objects.link(ob);ob.location=(x,y,z);ob.scale=(h,h,h);ob.rotation_euler[2]=rng.random()*math.tau
def make_rocks(rock_mat,sim,scenic):
    # Explicit physics proxies; no decorative rigid body silently replaces them.
    c.rock('COLLIDER | pine-topped center sea stack',(2.45,3.5,1.70),(2.25,1.82,4.80),rock_mat,sim,seed=101,collision=True,sub=3)
    c.rock('COLLIDER | west stack splitter',(-4.35,1.45,.50),(1.25,1.05,2.80),rock_mat,sim,seed=102,collision=True,sub=3)
    c.rock('COLLIDER | eastern breaking spur',(10.3,-1.5,-.2),(2.4,1.9,2.30),rock_mat,sim,seed=103,collision=True,sub=3)
    c.rock('COLLIDER | foreground reef right',(15.5,7.5,.50),(3.35,3.20,2.8),rock_mat,sim,seed=104,collision=True,sub=3)
    c.rock('COLLIDER | foreground reef left',(-14.6,10.5,1.55),(3.3,2.0,3.20),rock_mat,sim,seed=105,collision=True,sub=3)
    for i,(x,y,s) in enumerate([(-9,-2,.80),(-6,4,.90),(5.5,-3,1.0),(7,5,.75),(-2,8,.80),(13,3,.95)]):
        z=float(bathy(x,y))
        c.rock('COLLIDER | submerged trip reef %02d'%i,(x,y,z+.32),(s*1.25,s,s*.88),rock_mat,sim,seed=140+i,collision=True,sub=2)
    for i in range(22):
        y=-2+i*.89
        x=-17.1-.9*math.sin(y*.6)
        z=float(bathy(x,y))
        c.rock('COLLIDER | west coastal headland %02d'%i,(x,y,z+1.3),(1.70,1.7,2.6),rock_mat,sim,seed=200+i,collision=True,sub=3)
    for i in range(9):
        y=-1+i*1.7;x=17.1+.6*math.sin(y*.4);z=float(bathy(x,y))
        c.rock('COLLIDER | east coastal reef %02d'%i,(x,y,z+.8),(1.3,1.1,1.65),rock_mat,sim,seed=330+i,collision=True,sub=2)
    for i,(x,y,z,s) in enumerate([(-13,13,2,4.2),(-17,6,2.2,4.0),(-19,0,1.0,3.1),(18,12,.6,3.0),(19,0,.5,3.5)]):
        c.rock('SET | sculpted shoreline silhouette %02d'%i,(x,y,z),(s,s*.8,s*.65),rock_mat,scenic,seed=680+i,collision=False,sub=3)
def add_scans(root,scenic):
    assets=root.parent/'assets'
    manifest=assets/'asset_manifest.json'
    if not manifest.exists():return {'used':False,'reason':'No verified source manifest'}
    materials=[]
    from mathutils import Euler
    count=0
    for key,locations in [
        ('coastal_cliff_01',[((-16,10,5),.42,115),((-20,-1,2.8),.33,90)]),
        ('coast_rocks_01',[((-15,16,2.0),.23,25),((17,11,.7),.24,-35),((-19,-5,1.4),.2,75)])]:
        p=assets/key/(key+'_2k.blend')
        if not p.exists():raise FileNotFoundError(p)
        with bpy.data.libraries.load(str(p),link=False) as (src,dst):
            name=next((n for n in src.objects if n.endswith('_LOD0')),None)
            if name is None:raise RuntimeError('Missing scanned LOD0 '+key)
            dst.objects=[name]
        proto=dst.objects[0]
        a=np.asarray([v.co[:] for v in proto.data.vertices],dtype=float)
        cent=Vector((a.min(0)+a.max(0))*.5)
        for (loc,scale,rot) in locations:
            ob=bpy.data.objects.new('SCAN | '+key+' | '+str(count),proto.data)
            scenic.objects.link(ob)
            ob.matrix_world=Matrix.Translation(Vector(loc))@Euler((0,0,math.radians(rot))).to_matrix().to_4x4()@Matrix.Scale(scale,4)@Matrix.Translation(-cent)
            ob['license']='Poly Haven CC0';ob['physics']='Scenic only; named nearby low-poly meshes are colliders.'
            count+=1
        # Make imported texture paths resolve from their source package; pack for standalone .blend.
        for im in bpy.data.images:
            if not im.name.startswith(key):continue
            name=Path(bpy.path.abspath(im.filepath)).name
            candidate=assets/key/'textures'/name
            if candidate.exists():
                im.filepath=str(candidate)
                try:im.reload();im.pack()
                except Exception as e:print('PACK WARNING',im.name,repr(e),flush=True)
        for m in proto.data.materials:
            if m and m.use_nodes:materials.append(m)
        if proto.users==0:bpy.data.objects.remove(proto)
    return {'used':True,'scans':count,'materials':len(materials),'license':'CC0'}
def backdrop(scene,ocean_mat,rock_mat,scenic,plants):
    # Scenic ocean begins strictly outside the positive ocean model: y<-18.0.
    xs=np.linspace(-95,95,108);ys=np.linspace(-180,-18.35,135)
    v=[(float(x),float(y),float(.08*np.sin(.35*y+.07*x)+.03*np.cos(.64*y-.13*x))) for y in ys for x in xs]
    f=[];nx=len(xs)
    for j in range(len(ys)-1):
        for i in range(nx-1):a=j*nx+i;f.append((a,a+1,a+nx+1,a+nx))
    water=c.mesh_obj('SET | distant horizon ocean OUTSIDE solver domain',v,f,ocean_mat,scenic,True)
    water['simulated']=False;water['exclusive_extent']='Y less than -18.35'
    for i in range(12):
        x=-52+i*7.5; y=-75+4*math.sin(i*.9)
        c.rock('SET | distant sea stacks '+str(i),(x,y,1.0),
               (2.6,3.1,random.uniform(3.5,7)),rock_mat,scenic,seed=430+i,sub=3)
    for i in range(18):
        x=-80+i*5;y=-115+15*math.sin(i*.4)
        c.rock('SET | misty distant cliffs '+str(i),(x,y,2),(5,8,random.uniform(5,12)),rock_mat,scenic,seed=700+i)
def light_scene(scene):
    scene.world=bpy.data.worlds.new('Cloud-heavy Pacific dusk sky');w=scene.world;w.use_nodes=True
    n=w.node_tree.nodes;l=w.node_tree.links;bg=n.get('Background');bg.inputs['Strength'].default_value=.35
    sky=n.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING';sky.sun_elevation=math.radians(4)
    sky.sun_rotation=math.radians(155);sky.sun_intensity=.18;sky.air_density=1.6;sky.aerosol_density=1.65
    uv=n.new('ShaderNodeTexCoord');tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=3.9;tex.inputs['Detail'].default_value=6
    l.new(uv.outputs['Normal'],tex.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.29;ramp.color_ramp.elements[0].color=(.04,.095,.15,1)
    ramp.color_ramp.elements[1].position=.72;ramp.color_ramp.elements[1].color=(.65,.63,.66,1)
    l.new(tex.outputs['Fac'],ramp.inputs[0]);mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.78
    l.new(sky.outputs['Color'],mix.inputs[1]);l.new(ramp.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],bg.inputs['Color'])
    ld=bpy.data.lights.new('LIGHT | amber sun over open ocean','SUN');ld.energy=2.8;ld.color=(1,.65,.38);ld.angle=.075
    ob=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(ob);ob.rotation_euler=(math.radians(72),math.radians(-18),math.radians(124))
    ld=bpy.data.lights.new('LIGHT | blue storm sky fill','AREA');ld.energy=7200;ld.color=(.54,.79,1);ld.shape='DISK';ld.size=32
    ob=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(ob);ob.location=(-3,-8,25);c.look_at(ob,(0,0,0))
    ld=bpy.data.lights.new('LIGHT | warm left grazing rock','AREA');ld.energy=2500;ld.color=(1,.74,.50);ld.shape='DISK';ld.size=17
    ob=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(ob);ob.location=(16,-14,13);c.look_at(ob,(-3,2,0))
    scene.render.engine='CYCLES'
    scene.cycles.samples=64;scene.cycles.use_denoising=True;scene.cycles.max_bounces=8
    scene.render.resolution_x=1600;scene.render.resolution_y=900;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='AgX'
    scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.1
    preferences=bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type='OPTIX';preferences.get_devices()
    for d in preferences.devices:d.use=d.type=='OPTIX'
    assert any(d.use for d in preferences.devices),'No OptiX GPU'
    scene.cycles.device='GPU'
    return [d.name for d in preferences.devices if d.use]
def camera(scene):
    camdata=bpy.data.cameras.new('Ocean Swell | across cove, toward horizon')
    cam=bpy.data.objects.new('CAMERA | OPEN OCEAN REFERENCE',camdata);scene.collection.objects.link(cam)
    cam.location=(-13.4,16.4,10.2);c.look_at(cam,(1.7,-9,.85))
    camdata.lens=30;camdata.clip_end=600;scene.camera=cam
    return cam
def gather_metrics(scene):
    sess=scene_bridge.session();cnt=int(sess.solver.particles.refresh_host_count())
    arr=sess.solver.particles.position.numpy()[:cnt]
    assert np.isfinite(arr).all(),'Nonfinite particle positions'
    bins={}
    for y in (-15,-12,-9,-6,-3,0,3,6,9):
        m=(np.abs(arr[:,1]-y)<.85)&(np.abs(arr[:,0])<11)
        zz=arr[m,2]
        bins[str(y)]=float(np.quantile(zz,.985)) if len(zz)>40 else None
    stats=dict(sess.frame_stats[-1]) if sess.frame_stats else {}
    surf=bpy.data.objects.get('FluidSim Surface')
    return {'frame':scene.frame_current,'count':cnt,
            'whitewater':int(stats.get('whitewaterParticles') or 0),
            'whitewater_dropped':int(stats.get('whitewaterDropped') or 0),
            'surface_vertices':len(surf.data.vertices) if surf else 0,
            'surface_bands_p985':bins}
def build():
    report={'ok':False,'schema':'open-ocean-swell-v2','frames':[],'source_commit':'0d1d4fa0','reference':'open ocean swell, not river',
            'intended_driver':'full-width zero-mean oscillating force field; physically forced wave action'}
    try:
        bpy.context.preferences.use_preferences_save=False
        os.environ['COASTAL_ROOT']=str(ROOT)
        progress('creating non-channelled ocean bathymetry')
        scene=fresh_scene();scene.name='OCEAN SWELL | Rocky Pacific Cove';scene.gravity=(0,0,-9.81)
        sim=bpy.data.collections.new('01 | PHYSICS - CUDA ocean and reefs');scene.collection.children.link(sim)
        scenic=bpy.data.collections.new('02 | SCENERY - distant shore and ocean');scene.collection.children.link(scenic)
        plants=bpy.data.collections.new('03 | TREES - conifer forest');scene.collection.children.link(plants)
        rock_mat=c.rock_material();water=c.water_material('WATER | real turquoise Liquid Warp surface');far=c.water_material('SET | navy distant ocean only',offshore=True)
        foam,_=c.mat_principled('WHITEWATER | actual solver foam and spray',(.90,.965,1),.38)
        make_bathymetry(scene,rock_mat,sim,scenic);make_rocks(rock_mat,sim,scenic)
        make_pines(scene,plants);backdrop(scene,far,rock_mat,scenic,plants)
        report['scans']=add_scans(ROOT,scenic)
        light_scene(scene);camera(scene)
        progress('initializing simulated broad ocean')
        lo=(-20,-18,-4.8);hi=(20,16,9.0)
        dom=make_domain(scene,lo,hi,'DOMAIN | full-width ocean nearshore');c.move_to(dom,sim)
        ds=dom.fluidsim_domain
        resolution=float(os.environ.get('OCEAN_DX','0.23'))
        set_resolution(ds,lo,hi,resolution)
        ds.max_particles=2000000;ds.leaf_capacity=14000;ds.max_whitewater=650000
        ds.min_substeps=2;ds.max_substeps=16;ds.pressure_preconditioner='MULTIGRID'
        ds.pressure_iterations=115;ds.pressure_tolerance=1.e-5;ds.open_boundaries=False;ds.domain_sdf_walls=True
        ds.whitewater_enabled=True;ds.ww_churn_gain=1.45;ds.ww_churn_threshold=.17
        ds.ww_emission_rate=.60;ds.ww_life_min=.8;ds.ww_life_max=2.3
        ds.surface_smooth_iterations=2;ds.surface_smooth_strength=.55
        fs=scene.fluidsim;fs.live=False;fs.liquid_view_mode='SURFACE';fs.show_surface=True;fs.show_whitewater=True
        fs.surface_material=water;fs.whitewater_material=foam;fs.whitewater_display_scale=.34
        pts=ocean_seed(resolution*.54)  # Target ~6.3 seed particles per solver voxel
        report['initial_particles']=len(pts)
        assert len(pts)<ds.max_particles
        seed=make_particle_seed(scene,pts,'SOURCE | initial ocean (non-flat coherent swell packets)')
        c.move_to(seed,sim);seed.fluidsim.role='SOURCE';seed.fluidsim.source_ui_schema_version=4;seed.fluidsim.enabled=True;seed.fluidsim.auto_inject_seed=False;seed.fluidsim.source_region_kind='POINTS';seed.fluidsim.point_source_representation='DIRECT_APPEND';seed.fluidsim.emission_mode='INITIAL_FILL'
        seed.fluidsim.initial_velocity=(0,0,0)
        driver=make_wavemaker(sim)
        report['driver_keyed']=bool(driver.animation_data and driver.animation_data.action)
        assert report['driver_keyed']
        for ob in (dom,seed,driver):
            ob.hide_render=True;ob.display_type='WIRE';ob.hide_set(True)
        scene.frame_start=1;scene.frame_end=240;scene.frame_set(1);scene.sync_mode='NONE'
        doc=bpy.data.texts.new('READ ME | REAL OCEAN SWELL')
        doc.write('COASTAL OCEAN SWELL V2: NO RIVER, NO NARROW SOURCE.\n\n1. Live GPU Liquid Warp ocean seeded to sea level with two broad cross-bay swell packets.\n2. Native animated FORCE region accelerates the offshore water in alternating directions with zero mean acceleration. No source adds water after frame1.\n3. Waves propagate over bathymetry and shallow reefs into the rocky open cove; three principal sea stacks have actual collider meshes.\n4. Background ocean beyond the simulation boundary, distant cliffs, sky, forests and Poly Haven scans are visual set dressing only.\n5. Press Space to advance from frame1. Return to frame1 to reset. A separate frozen Render State file holds one simulated crest for photography.\n6. Blender 5.2 / Warp 1.17 CUDA. Avoid production profile; use task owner launcher.\n7. Animated force is a regional acceleration wavemaker, NOT an exact non-reflecting boundary condition; far closed walls may reflect in long runs.\n')
        scene['ocean_swell_v2']=True;scene['wavetank_engineering_driver']='Analytic BOX FORCE, sinusoidal with no net acceleration'
        scene['simulated_foreground']=True
        scene['distant_ocean_set_dressing_only']=True
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':
                    area.spaces.active.region_3d.view_perspective='CAMERA'
                    area.spaces.active.region_3d.view_camera_zoom=14
                    area.spaces.active.overlay.show_overlays=False
                    area.spaces.active.shading.type='SOLID'
        bpy.context.view_layer.update()
        progress('GPU admission',initial_particles=len(pts),resolution_divisions=ds.resolution_divisions,leaf_capacity=ds.leaf_capacity)
        result=bpy.ops.fluidsim.initialize()
        if result!={'FINISHED'}:
            plan=scene_bridge.session().refresh_capacity_plan(scene)
            blockers=[{'code':x.code,'required':x.required,'available':x.available} for x in plan.blockers] if plan else []
            raise RuntimeError('Solver capacity or initialization blocked: '+str(blockers))
        sess=scene_bridge.session();assert 'cuda' in str(sess.solver.device),'CUDA mandatory'
        assert bpy.ops.fluidsim.step(frames=1)=={'FINISHED'},'Initial solver output missing'
        c.surface_foam(scene)
        fs.live=True
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(DEL/'Ocean_Swell_LIVE.blend'),check_existing=False,compress=True)
        report['live_file_sha256']=hashlib.sha256((DEL/'Ocean_Swell_LIVE.blend').read_bytes()).hexdigest()
        progress('saved independent cold-startable ocean',filepath=str(DEL/'Ocean_Swell_LIVE.blend'))
        goal=int(os.environ.get('OCEAN_FRAMES','110'))
        for frame in range(2,goal+1):
            scene.frame_set(frame)
            assert sess.active and fs.live,'GPU solver lost'
            if frame%12==2 or frame==goal:
                data=gather_metrics(scene)
                data['force_strength']=float(driver.fluidsim.force_strength)
                report['frames'].append(data)
                progress('physical wave propagation',**data)
                (EVIDENCE/'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        assert max(v['whitewater'] for v in report['frames'])>0,'No breaking whitewater'
        assert all(v['count']>0 for v in report['frames']),'Ocean disappeared'
        surf=bpy.data.objects.get('FluidSim Surface')
        assert surf and len(surf.data.vertices)>0
        fs.live=False
        report['device']=str(sess.solver.device)
        bpy.ops.wm.save_as_mainfile(filepath=str(DEL/'Ocean_Swell_RENDER_STATE.blend'),check_existing=False,compress=True)
        scene.render.resolution_percentage=60;scene.cycles.samples=32
        scene.render.filepath=str(DEL/'Ocean_Swell_preview.png')
        progress('rendering real simulated crest at frame '+str(goal))
        with playback.suppress_frame_handlers():bpy.ops.render.render(write_still=True)
        report['ok']=True;report['preview_sha256']=hashlib.sha256(Path(scene.render.filepath).read_bytes()).hexdigest()
        report['render_frame']=scene.frame_current
    except BaseException as e:
        report['error']=repr(e);report['traceback']=traceback.format_exc()
        progress('FAILED',error=repr(e));traceback.print_exc()
    finally:
        (EVIDENCE/'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        f=os.environ.get('FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT')
        if f:Path(f).write_text(json.dumps({'ok':bool(report['ok']),'report':str(EVIDENCE/'build_report.json')}))
        print('OCEAN-SWELL-BUILD-'+('PASS' if report['ok'] else 'FAIL'),flush=True)
        bpy.context.preferences.use_preferences_save=False
        bpy.context.preferences.view.use_save_prompt=False
        bpy.ops.wm.quit_blender()
if bpy.app.background:
    build()
else:
    bpy.app.timers.register(build,first_interval=.5)
