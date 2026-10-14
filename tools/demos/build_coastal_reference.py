"""Coastal Rapids: real Liquid Warp 3.0 scene, no fabricated foreground liquid.
Run with the repository production-gate lane and an isolated Blender profile.
Outputs live frame-1 .blend, solver-derived still/review .blend, and JSON evidence.
"""
from __future__ import annotations
import bpy, math, random, json, os, time, hashlib
import numpy as np
from pathlib import Path
from mathutils import Vector, noise
from tools.demos.build_demo_scenes import fresh_scene, make_domain, make_source_box, set_resolution
from fluidSim_warp_blender51.blender.colliders import tag_collider
from fluidSim_warp_blender51.blender.scene_bridge import session

ROOT=Path(os.environ.get('COASTAL_TASK_ROOT', str(Path(__file__).resolve().parents[3])))
OUT=ROOT/'deliverables'; EVIDENCE=ROOT/'evidence'
OUT.mkdir(parents=True,exist_ok=True); EVIDENCE.mkdir(parents=True,exist_ok=True)
R=random.Random(61010)
MODE=os.environ.get('COASTAL_MODE','build')
LIVE=OUT/'Coastal_Rapids_Live.blend'

def progress(stage,**values):
    data={'stage':stage,'time':time.time(),**values}
    (EVIDENCE/'progress.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('COASTAL_PROGRESS '+json.dumps(data),flush=True)

def mat(name,color,rough=.5,metal=0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1)
    m.use_nodes=True; p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough; p.inputs['Metallic'].default_value=metal
    return m,p

def rock_material():
    m,p=mat('Basalt | wet fractured coastal stone',(.055,.071,.072),.29)
    n=m.node_tree.nodes; l=m.node_tree.links
    tex=n.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value=3.2; tex.inputs['Detail'].default_value=5
    tc=n.new('ShaderNodeTexCoord'); l.new(tc.outputs['Generated'],tex.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].position=.18; ramp.color_ramp.elements[0].color=(.008,.016,.021,1)
    ramp.color_ramp.elements[1].position=.8; ramp.color_ramp.elements[1].color=(.24,.235,.19,1)
    e=ramp.color_ramp.elements.new(.55);e.color=(.06,.078,.085,1)
    l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs['Color'],p.inputs['Base Color'])
    fine=n.new('ShaderNodeTexNoise');fine.inputs['Scale'].default_value=95;fine.inputs['Detail'].default_value=3
    l.new(tc.outputs['Generated'],fine.inputs['Vector'])
    b=n.new('ShaderNodeBump');b.inputs['Strength'].default_value=.62;b.inputs['Distance'].default_value=.065
    l.new(fine.outputs['Fac'],b.inputs['Height']);l.new(b.outputs['Normal'],p.inputs['Normal'])
    rough=n.new('ShaderNodeMapRange');rough.inputs['From Min'].default_value=.2;rough.inputs['From Max'].default_value=.8
    rough.inputs['To Min'].default_value=.13;rough.inputs['To Max'].default_value=.5
    l.new(tex.outputs['Fac'],rough.inputs[0]);l.new(rough.outputs[0],p.inputs['Roughness'])
    return m

def water_material():
    m,p=mat('Liquid Warp | turquoise transmitting water',(.025,.32,.41),.085)
    p.inputs['Transmission Weight'].default_value=1;p.inputs['IOR'].default_value=1.333
    n=m.node_tree.nodes;l=m.node_tree.links
    vol=n.new('ShaderNodeVolumeAbsorption');vol.inputs['Color'].default_value=(.18,.67,.74,1);vol.inputs['Density'].default_value=.21
    l.new(vol.outputs[0],n.get('Material Output').inputs['Volume'])
    tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=19;tex.inputs['Detail'].default_value=3;tex.inputs['Roughness'].default_value=.6
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.16;bump.inputs['Distance'].default_value=.011
    l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],p.inputs['Normal'])
    return m

def mesh_obj(name,verts,faces,material,collection,collider=False):
    me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update()
    ob=bpy.data.objects.new(name,me);collection.objects.link(ob)
    if material: me.materials.append(material)
    if collider: tag_collider(ob,friction=.08,restitution=0)
    return ob

def rock(name,center,axes,height,material,collection,collider=False,seed=None):
    rng=random.Random(seed if seed is not None else R.randrange(999999))
    sides=rng.choice((9,11,13));levels=7;verts=[];faces=[]
    angular=[1+rng.uniform(-.2,.2) for _ in range(sides)]
    for k in range(levels):
        t=k/(levels-1); radius=1.0-.60*t**1.65
        if k==0: radius=.94
        for j in range(sides):
            a=2*math.pi*j/sides
            q=angular[j]*(1+rng.uniform(-.10,.10))*radius
            x=axes[0]*q*math.cos(a)+.16*height*t
            y=axes[1]*q*math.sin(a)-.10*height*t
            z=height*t + (0 if k==0 else rng.uniform(-.09,.09)*height)
            verts.append((center[0]+x,center[1]+y,center[2]+z))
    for k in range(levels-1):
        for j in range(sides):
            a=k*sides+j;b=k*sides+(j+1)%sides;c=(k+1)*sides+(j+1)%sides;d=(k+1)*sides+j
            if (j+k)%3==0: faces.extend(((a,b,c),(a,c,d)))
            else: faces.append((a,b,c,d))
    faces.append(tuple(reversed(range(sides))))
    faces.append(tuple((levels-1)*sides+j for j in range(sides)))
    ob=mesh_obj(name,verts,faces,material,collection,collider)
    # Facets remain deliberate; detailed normal variation comes from the stone shader.
    return ob

def bed_height(y):
    # Closed stair-step bed, with no thin plates or interpenetrating shells.
    if y>5.4:return 1.80
    if y>3.8:return 1.10
    if y>1.7:return .48
    if y>-2.2:return -.10
    return -.66

def make_bed(material,collection):
    xs=np.linspace(-4.5,4.5,25)
    ys=sorted(set(list(np.linspace(-6.8,8.6,75))+[-2.23,-2.2,-2.17,1.67,1.7,1.73,3.77,3.8,3.83,5.37,5.4,5.43]))
    v=[];f=[];nx=len(xs);ny=len(ys)
    for y in ys:
        for x in xs:
            z=bed_height(y)+.025*math.sin(3*x+1.8*y)+.018*math.cos(8*x-3*y)
            v.append((float(x),float(y),z))
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;f.append((a,a+1,a+nx+1,a+nx))
    boundary=list(range(nx))+[j*nx+nx-1 for j in range(1,ny)]+list(range(ny*nx-2,(ny-1)*nx-1,-1))+[j*nx for j in range(ny-2,0,-1)]
    # boundary follows top counter-clockwise, sides wind outward.
    bottom=[]
    for a in boundary:
        bottom.append(len(v));v.append((v[a][0],v[a][1],-1.25))
    for i,a in enumerate(boundary):
        j=(i+1)%len(boundary);f.append((a,bottom[i],bottom[j],boundary[j]))
    f.append(tuple(reversed(bottom)))
    return mesh_obj('Collision | stepped submerged bed',v,f,material,collection,True)

def pine(name,loc,h,material,bark,collection,seed):
    rng=random.Random(seed);v=[];f=[];mi=[]
    def cone(z,r,length,sides=9,branch=False):
        start=len(v)
        for j in range(sides):
            a=2*math.pi*j/sides;rad=r*(rng.uniform(.62,1.15) if branch else 1)
            v.append((loc[0]+rad*math.cos(a),loc[1]+rad*math.sin(a),loc[2]+z+rng.uniform(-.025,.025)*h))
        v.append((loc[0]+.035*h*(z/h),loc[1],loc[2]+z+length));top=len(v)-1
        for j in range(sides):f.append((start+j,start+(j+1)%sides,top));mi.append(0 if branch else 1)
        f.append(tuple(start+j for j in reversed(range(sides))));mi.append(0 if branch else 1)
    cone(0,.023*h,h,7)
    for k in range(9):
        t=k/9;z=.18*h+t*.68*h
        cone(z,.24*h*(1-t)*rng.uniform(.9,1.1),.26*h,11,True)
    ob=mesh_obj(name,v,f,material,collection)
    ob.data.materials.append(bark)
    for poly,i in zip(ob.data.polygons,mi):poly.material_index=i
    return ob

def point_at(ob,target):ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler()

def lighting(scene):
    world=bpy.data.worlds.new('Storm coast | procedural sky');scene.world=world;world.use_nodes=True
    n=world.node_tree.nodes;l=world.node_tree.links;n.clear()
    out=n.new('ShaderNodeOutputWorld');bg=n.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=.45
    tc=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ');l.new(tc.outputs['Normal'],sep.inputs[0])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=0;ramp.color_ramp.elements[0].color=(.22,.28,.32,1)
    ramp.color_ramp.elements[1].position=.65;ramp.color_ramp.elements[1].color=(.023,.054,.098,1)
    l.new(sep.outputs['Z'],ramp.inputs[0])
    noise_tex=n.new('ShaderNodeTexNoise');noise_tex.inputs['Scale'].default_value=4.4;noise_tex.inputs['Detail'].default_value=5
    l.new(tc.outputs['Normal'],noise_tex.inputs['Vector'])
    mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.7
    l.new(ramp.outputs[0],mix.inputs[1]);l.new(noise_tex.outputs['Fac'],mix.inputs[2]);l.new(mix.outputs[0],bg.inputs['Color']);l.new(bg.outputs[0],out.inputs[0])
    def light(name,kind,loc,power,color,size=5):
        data=bpy.data.lights.new(name,kind);data.energy=power;data.color=color
        if kind=='AREA':data.shape='DISK';data.size=size
        if kind=='SUN':data.angle=.075
        ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob);ob.location=loc;point_at(ob,(0,1,0));return ob
    light('Sky opening | cold softbox','AREA',(1,-1,11),2100,(.58,.79,1),11)
    light('Sunset | grazing warm light','SUN',(-9,-2,5),1.7,(1,.69,.40))
    light('Cove bounce','AREA',(-4,-7,4),650,(.43,.73,1),7)
    scene.render.engine='CYCLES';scene.cycles.samples=48;scene.cycles.use_denoising=True
    scene.cycles.max_bounces=10;scene.cycles.transmission_bounces=8;scene.cycles.transparent_max_bounces=8
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    used=[]
    for d in prefs.devices:
        d.use=d.type=='OPTIX'
        if d.use:used.append(d.name)
    assert used,'No OptiX GPU available'
    scene.cycles.device='GPU'
    scene.render.resolution_x=1280;scene.render.resolution_y=720;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.6
    return used

def build():
    progress('authoring geometry')
    scene=fresh_scene();scene.name='Coastal Rapids | LIVE Liquid Warp'
    natural=bpy.data.collections.new('01 | Coastal collision geometry');scene.collection.children.link(natural)
    details=bpy.data.collections.new('02 | Basalt detail and coastal forest');scene.collection.children.link(details)
    background=bpy.data.collections.new('03 | Distant environment (not simulated)');scene.collection.children.link(background)
    basalt=rock_material();water=water_material()
    foam,p=mat('Whitewater | real solver foam and spray',(.92,.97,1),.44)
    p.inputs['Subsurface Weight'].default_value=.025
    leaves,_=mat('Coastal fir | deep evergreen',(.028,.062,.035),.83)
    bark,_=mat('Coastal fir | bark',(.069,.047,.025),.9)
    make_bed(basalt,natural)
    # Bank masses form a narrowing S-shaped upper channel.
    peaks=[]
    for side in (-1,1):
        for k,y in enumerate(np.linspace(-6.1,8.0,11)):
            cx=-1.1*max(0,min(1,(y-1.5)/4))
            width=2.75 if y<1.7 else 2.25
            x=cx+side*(width+R.uniform(-.12,.22))
            base=bed_height(float(y))-.3;h=R.uniform(1.1,1.9)+(0.6 if side<0 and y>1 else 0)
            axes=(R.uniform(.85,1.3),R.uniform(.85,1.3))
            ob=rock(f'Bank {side:+d} {k:02d}',(x,float(y),base),axes,h,basalt,natural,True)
            peaks.append((x,float(y),base+h*.84,h))
    rock('Hero | central sea stack',(1.04,2.0,-.28),(.76,.65),3.0,basalt,natural,True,908)
    rock('Hero | foreground left outcrop',(-2.8,-4.4,-.78),(1.22,1.5),1.82,basalt,natural,True,310)
    rock('Hero | foreground right outcrop',(3.0,-3.7,-.75),(1.05,1.23),1.66,basalt,natural,True,311)
    for i,(x,y,r,h) in enumerate([(-1.5,-1.3,.43,.8),(.1,-2.0,.36,.53),(-.8,3.1,.34,.7),(1.65,-.2,.38,.66),(-2.0,5.0,.34,.8)]):
        rock(f'Rapids | flow-deflecting boulder {i}',(x,y,bed_height(y)-.13),(r,r*.85),h,basalt,natural,True,120+i)
    # Smaller angular shards enrich the banks without extra collider gathering.
    for i in range(100):
        side=R.choice((-1,1));y=R.uniform(-6.0,8.2);cx=-1.1*max(0,min(1,(y-1.5)/4))
        x=cx+side*R.uniform(2.55,4.7);z=bed_height(y)+R.uniform(.1,.65)
        rock(f'Basalt fragment {i:03d}',(x,y,z),(R.uniform(.12,.45),R.uniform(.13,.5)),R.uniform(.18,.6),basalt,details)
    for i,(x,y,z,h) in enumerate(peaks):
        if y<-.5:continue
        for j in range(R.randint(1,4)):
            pine(f'Wind-shaped coastal fir {i:02d}.{j}',(x+R.uniform(-.35,.35),y+R.uniform(-.3,.3),z),R.uniform(.65,1.65),leaves,bark,details,400+i*7+j)
    for i,(x,y,h) in enumerate([(1.30,1.82,.72),(1.05,2.10,.9),(.99,2.22,.54)]):
        pine(f'Sea-stack fir {i}',(x,y,2.41),h,leaves,bark,details,870+i)
    # Distant ocean is scenery only; nothing covers the foreground solver.
    verts=[];faces=[];nx=160;ny=130
    for j in range(ny):
        y=9.0+j*1.45
        for i in range(nx):
            x=-100+i*1.45;z=-.45+.10*math.sin(x*.6+y*.26)+.035*math.sin(x*1.1-y*.95)
            verts.append((x,y,z))
    for j in range(ny-1):
        for i in range(nx-1):a=j*nx+i;faces.append((a,a+1,a+1+nx,a+nx))
    ocean,p=mat('Distant ocean | set dressing only',(.018,.06,.085),.22);p.inputs['Metallic'].default_value=.25
    ob=mesh_obj('Distant ocean | outside simulation domain',verts,faces,ocean,background)
    for p in ob.data.polygons:p.use_smooth=True
    for i in range(12):
        x=-24+i*3.8;y=R.uniform(22,35);h=R.uniform(3,7)
        ob=rock(f'Distant headland {i}',(x,y,-.8),(R.uniform(2.0,4.0),R.uniform(2,5)),h,basalt,background)
        if i<7:
            for j in range(4):pine(f'Distant fir {i}.{j}',(x+R.uniform(-1.5,1.5),y,h*.85),R.uniform(.7,1.3),leaves,bark,background,1200+i*9+j)
    for i,(x,y,h) in enumerate([(13,24,2.1),(20,37,3.3),(7,44,2.4)]):rock(f'Offshore stack {i}',(x,y,-.8),(h*.3,h*.4),h,basalt,background)
    lo=(-4.45,-6.4,-1.15);hi=(4.45,8.4,4.3)
    dom=make_domain(scene,lo,hi,'Coastal Reference Domain');ds=dom.fluidsim_domain
    set_resolution(ds,lo,hi,.095)
    ds.max_particles=850000;ds.min_substeps=1;ds.max_substeps=16
    ds.pressure_preconditioner='MULTIGRID';ds.pressure_iterations=120;ds.pressure_tolerance=1e-5
    ds.whitewater_enabled=True;ds.max_whitewater=260000
    ds.ww_churn_gain=2.4;ds.ww_churn_threshold=.035;ds.ww_emission_rate=45
    ds.ww_life_min=1.2;ds.ww_life_max=3.5
    ds.surface_smooth_iterations=1;ds.surface_smooth_strength=.55
    fs=scene.fluidsim;fs.live=False;fs.liquid_view_mode='SURFACE';fs.show_surface=True;fs.show_particles=False
    fs.surface_material=water;fs.whitewater_material=foam;fs.particle_material=water
    fs.whitewater_display_scale=.9
    fill_boxes=[((-2.35,5.42,1.81),(.05,8.20,2.29)),((-2.3,3.83,1.12),(.95,5.38,1.61)),((-2.38,1.75,.50),(1.9,3.77,.99)),((-2.3,-2.14,-.06),(2.34,1.64,.50)),((-2.35,-6.22,-.61),(2.35,-2.26,.01))]
    for i,(a,b) in enumerate(fill_boxes):
        src=make_source_box(scene,a,b,f'Water source | initial pool {i+1}',emit_rate=8,emission_mode='INITIAL_FILL')
        src.fluidsim.initial_velocity=(0,-1.45,0);src.hide_render=True;src.display_type='WIRE'
    src=make_source_box(scene,(-2.26,7.55,1.84),(-.10,8.18,2.30),'Water source | continuous upper rapids',emit_rate=8,emission_mode='CONTINUOUS_INFLOW')
    src.fluidsim.initial_velocity=(.12,-2.5,0);src.hide_render=True;src.display_type='WIRE'
    bpy.context.view_layer.update()
    camdata=bpy.data.cameras.new('Coastal reference | 26 mm');cam=bpy.data.objects.new('Camera | reference composition',camdata);scene.collection.objects.link(cam)
    cam.location=(7.7,-13.5,6.1);point_at(cam,(-.2,1.5,.8));camdata.lens=26;camdata.clip_end=400;scene.camera=cam
    devices=lighting(scene)
    scene.frame_start=1;scene.frame_end=144;scene.frame_set(1)
    fs.live=True
    text=bpy.data.texts.new('START HERE | Coastal Rapids')
    text.write('COASTAL RAPIDS - LIVE LIQUID WARP 3.0\n\nPress Play from frame 1. Initial pools and the continuing upstream inlet feed a real GPU solver.\nThe riverbed, banks, central sea stack and rapids boulders are real colliders.\nFluidSim Surface and FluidSim Whitewater are solver outputs, not hand-animated water.\nReset by returning to frame 1. Object-owned settings are on Coastal Reference Domain.\nThe distant ocean beyond the domain and distant cliffs/trees are set dressing.\nThe separate RenderReview file is a frozen solver-derived frame, NOT this live demo.\nAll surface/foam visuals can be reproduced by advancing this file.\nThe source artwork is an aesthetic target, not a solver screenshot or proof of equivalence.\n')
    scene['coastal_reference_variant']=True;scene['fluidsim_playback_ready']=True
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.overlay.show_overlays=False
                area.spaces.active.shading.type='MATERIAL'
                area.spaces.active.shading.use_scene_world=True;area.spaces.active.shading.use_scene_lights=True
    bpy.ops.object.select_all(action='DESELECT');dom.select_set(True);bpy.context.view_layer.objects.active=dom
    bpy.ops.wm.save_as_mainfile(filepath=str(LIVE),check_existing=False,compress=True)
    progress('scene authored and saved',objects=len(scene.objects),colliders=len([o for o in scene.objects if getattr(o,'fluidsim',None) and o.fluidsim.role=='COLLIDER']),resolution=ds.resolution_divisions,gpu=devices)
    return scene

def verify_and_render(scene):
    from fluidSim_warp_blender51.blender import scene_bridge
    assert scene.fluidsim.domain_object and scene.fluidsim.domain_object.name=='Coastal Reference Domain'
    assert scene.frame_current==scene.frame_start==1
    assert scene.fluidsim.live
    assert scene.fluidsim.domain_object.fluidsim_domain.whitewater_enabled
    bpy.ops.fluidsim.initialize()
    timings=[];snap=None;delta=0;maxww=0;data=[]
    for frame in range(2,61):
        start=time.perf_counter();scene.frame_set(frame);sess=scene_bridge.session()
        assert sess.active, 'Solver did not initialize through production path'
        count=int(sess.solver.particles.refresh_host_count());assert count>0,'Empty solver'
        if frame in (2,3):
            a=sess.solver.particles.position[:min(1024,count)].numpy().copy()
            if snap is not None:delta=float(np.max(np.linalg.norm(a-snap,axis=1)))
            snap=a
        if frame==2 and callable(globals().get('scene_finish_hook')): scene_finish_hook(scene)
        stats=dict(sess.frame_stats[-1]) if sess.frame_stats else {}
        ww=int(stats.get('whitewaterParticles') or 0);maxww=max(maxww,ww)
        elapsed=time.perf_counter()-start;timings.append(elapsed)
        data.append({'frame':frame,'liquid':count,'whitewater':ww,'seconds':elapsed})
        progress('simulating real frame handlers',frame=frame,particles=count,whitewater=ww,seconds=round(elapsed,3))
        (EVIDENCE/'simulation_frames.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    assert delta>1e-6, 'Real timeline did not move liquid'
    assert maxww>0,'Whitewater enabled but no real whitewater emitted'
    surf=bpy.data.objects.get('FluidSim Surface');assert surf and len(surf.data.vertices)>0,'No published surface'
    scene.fluidsim.live=False
    for ob in scene.objects:
        if getattr(ob,'fluidsim',None) and ob.fluidsim.role in ('SOURCE','DOMAIN'):ob.hide_render=True
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    for d in prefs.devices: d.use=d.type=='OPTIX'
    assert any(d.use and d.type=='OPTIX' for d in prefs.devices),'No OptiX render device'
    scene.cycles.device='GPU'
    progress('rendering native Cycles frame',frame=scene.frame_current,surface_vertices=len(surf.data.vertices),render_devices=[d.name for d in prefs.devices if d.use])
    scene.render.filepath=str(OUT/'Coastal_Rapids_Cycles.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'Coastal_Rapids_RenderReview.blend'),check_existing=False,compress=True)
    bpy.ops.render.render(write_still=True)
    img=bpy.data.images.load(str(OUT/'Coastal_Rapids_Cycles.png'),check_existing=False)
    img.scale(640,360);img.file_format='JPEG';img.filepath_raw=str(OUT/'Coastal_Rapids_Review.jpg');img.save()
    report={'blender':bpy.app.version_string,'live_file':str(LIVE),'render_file':scene.render.filepath,'frame':scene.frame_current,'position_delta_2_to_3':delta,'max_whitewater':maxww,'surface_vertices':len(surf.data.vertices),'median_step_seconds':float(np.median(timings[2:])),'frame_count':len(data),'simulation_device':str(scene_bridge.session().solver.device),'status':'PASS','visual_review':'PENDING'}
    (EVIDENCE/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    progress('complete pending visual review',**report)

if __name__ == '__main__':
    if MODE=='build':
        scene=build()
    else:
        from tools.repo_addon_loader import enable_repo_addon
        enable_repo_addon(str(ROOT/'source'),default_set=False,persistent=False)
        bpy.ops.wm.open_mainfile(filepath=str(LIVE));scene=bpy.context.scene
    if MODE not in ('geometry',):
        verify_and_render(scene)
    print('COASTAL-LANE-PASS',flush=True)