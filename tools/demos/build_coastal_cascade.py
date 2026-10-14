"""Build a real Liquid Warp coastal-cascade demo in an isolated Blender session.
Run only through the repository profile-safe DCC review harness.
The cascade, its surface and its whitewater are simulated. Offshore ocean is set dressing.
"""
from __future__ import annotations
import os, sys, math, json, time, random, traceback, hashlib
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, noise

ROOT = Path(os.environ.get('COASTAL_ROOT', r'F:\LiquidWarp-CoastalCascade-20261011'))
REPO = ROOT / 'source'
OUT = ROOT / 'delivery'
OUT.mkdir(parents=True, exist_ok=True)
for p in (str(REPO), r'C:\Users\Tyler\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages'):
    if p not in sys.path: sys.path.insert(0,p)
# The DCC harness owns profile isolation. All compilation caches stay in this task.
os.environ['WARP_CACHE_PATH'] = str(ROOT / 'cache' / 'warp')
import warp as wp
wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
from fluidSim_warp_blender51.blender import colliders, sources, playback
from fluidSim_warp_blender51.blender.scene_bridge import session, release_session
from tools.demos.build_demo_scenes import fresh_scene, make_domain, set_resolution, make_particle_seed, make_source_box, look_at

random.seed(27011)
T0=time.time()
REPORT={'schema':'coastal-cascade-v1', 'source_commit':'0d1d4fa0ce13a7f1b1abe59e7324c115edc3595a', 'events':[], 'frames':[], 'ok':False}

def log(msg):
    print('[coastal] '+msg, flush=True)
    REPORT['events'].append({'seconds':round(time.time()-T0,2),'message':msg})
    (OUT/'build_progress.json').write_text(json.dumps(REPORT,indent=2),encoding='utf8')

def mat_principled(name,color,roughness=.5,metallic=0.,transmission=0.):
    m=bpy.data.materials.new(name);m.use_nodes=True;m.diffuse_color=(*color,1)
    bs=m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*color,1)
    bs.inputs['Roughness'].default_value=roughness
    bs.inputs['Metallic'].default_value=metallic
    bs.inputs['Transmission Weight'].default_value=transmission
    return m,bs

def noise_tex(nodes,scale,detail=4):
    n=nodes.new('ShaderNodeTexNoise');n.inputs['Scale'].default_value=scale;n.inputs['Detail'].default_value=detail;n.inputs['Roughness'].default_value=.7
    return n

def rock_material():
    m,bs=mat_principled('Basalt | fractured wet coastal rock',(.075,.086,.092),.35)
    n=m.node_tree.nodes;l=m.node_tree.links
    tc=n.new('ShaderNodeTexCoord');base=noise_tex(n,2.2,5)
    l.new(tc.outputs['Object'],base.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].position=.2;ramp.color_ramp.elements[0].color=(.009,.017,.021,1)
    ramp.color_ramp.elements[1].position=.78;ramp.color_ramp.elements[1].color=(.16,.19,.185,1)
    mid=ramp.color_ramp.elements.new(.49);mid.color=(.045,.056,.062,1)
    l.new(base.outputs['Fac'],ramp.inputs['Fac']);l.new(ramp.outputs['Color'],bs.inputs['Base Color'])
    fine=noise_tex(n,35,3);l.new(tc.outputs['Object'],fine.inputs['Vector'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.65;bump.inputs['Distance'].default_value=.065
    l.new(fine.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
    rough=n.new('ShaderNodeMapRange');rough.inputs['From Min'].default_value=.2;rough.inputs['From Max'].default_value=.8;rough.inputs['To Min'].default_value=.12;rough.inputs['To Max'].default_value=.5
    l.new(base.outputs['Fac'],rough.inputs['Value']);l.new(rough.outputs[0],bs.inputs['Roughness'])
    return m

def water_material(name,offshore=False):
    m,bs=mat_principled(name,(.045,.32,.39) if not offshore else (.014,.055,.078),.11,transmission=.88 if not offshore else .25)
    bs.inputs['IOR'].default_value=1.333
    n=m.node_tree.nodes;l=m.node_tree.links
    tc=n.new('ShaderNodeTexCoord');tex=noise_tex(n,8 if not offshore else 2,3)
    l.new(tc.outputs['Object'],tex.inputs['Vector'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.14;bump.inputs['Distance'].default_value=.02 if not offshore else .08
    l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
    if not offshore:
        v=n.new('ShaderNodeVolumeAbsorption');v.inputs['Color'].default_value=(.075,.63,.72,1);v.inputs['Density'].default_value=.22
        l.new(v.outputs['Volume'],n.get('Material Output').inputs['Volume'])
    return m

COLS={}
def collection(name):
    c=bpy.data.collections.new(name);bpy.context.scene.collection.children.link(c);COLS[name]=c;return c

def move_to(o,c):
    for old in list(o.users_collection):old.objects.unlink(o)
    c.objects.link(o);return o

def mesh_obj(name,verts,faces,mat,col,smooth=False):
    me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update()
    if me.validate(verbose=False):raise RuntimeError(name+' required mesh repair')
    o=bpy.data.objects.new(name,me);col.objects.link(o)
    if mat:me.materials.append(mat)
    if smooth:
        for p in me.polygons:p.use_smooth=True
    return o

def center(y):return -2.8-.47*np.asarray(y)+.42*np.sin(np.asarray(y)*.29)
def width(y):return 4.65-1.1*np.tanh((np.asarray(y)-4)/8)
def smooth(x,a,b):
    t=np.clip((np.asarray(x)-a)/(b-a),0,1);return t*t*(3-2*t)
def bed_level(y):
    y=np.asarray(y)
    return -.95+1.75*smooth(y,-5.2,-4.25)+1.7*smooth(y,1.5,2.5)+1.8*smooth(y,8,9)+1.7*smooth(y,14.7,15.8)
def bed_height(x,y):
    """Closed continuous cascade and open tidal basin, not a raised rectangular tray."""
    x,y=np.broadcast_arrays(np.asarray(x),np.asarray(y))
    c=center(y);w=width(y)
    # The left bank is land; the right side opens to the sea around the hero stack.
    left=5.1*smooth(c-x,w-.55,w+1.5)
    left*=1+.13*np.sin(y*.61)+.06*np.sin(y*1.77)
    east=smooth(x-c,1.0,3.65)
    basin=.27+.20*np.sin(y*.24)+.08*np.sin(x*.55)
    basin=np.where(y < -4.4,bed_level(y),basin)
    base=bed_level(y)*(1-east)+basin*east
    # Upper right island holds the incoming stream; it ends before the coastal bay.
    upper_bank=4.1*smooth(x-c,w-.2,w+1.6)*smooth(y,7.5,11)
    rough=.07*np.sin(1.6*x+.2*y)*np.sin(y*1.4)+.04*np.sin(x*4.1-y*1.7)
    dry_detail=(.46*np.sin(1.14*x+.42*y)*np.cos(.73*y)+.17*np.sin(3.2*x-1.8*y))*smooth(c-x,w,w+1.4)
    return base+left+upper_bank+rough+dry_detail

# Closed heightfield solid. No overlapping shells inside this collider.
def build_bed(mat,col):
    xs=np.linspace(-20,12,145);ys=np.linspace(-12,24,163)
    gx,gy=np.meshgrid(xs,ys,indexing='ij');gz=bed_height(gx,gy)
    nx,ny=gx.shape;vs=list(zip(gx.ravel(),gy.ravel(),gz.ravel()));vs+=list(zip(gx.ravel(),gy.ravel(),np.full(gx.size,-3.0)))
    nn=gx.size;faces=[]
    for i in range(nx-1):
        for j in range(ny-1):
            a=i*ny+j;b=a+ny
            faces.extend([(a,b,b+1),(a,b+1,a+1)])
            faces.append((nn+a,nn+a+1,nn+b+1,nn+b))
    ring=[i*ny for i in range(nx-1)]+[(nx-1)*ny+j for j in range(ny-1)]+[i*ny+ny-1 for i in range(nx-1,0,-1)]+[j for j in range(ny-1,0,-1)]
    for a,b in zip(ring,ring[1:]+ring[:1]):faces.append((a,nn+a,nn+b,b))
    o=mesh_obj('CASCADE | continuous bed collider',vs,faces,mat,col,True)
    colliders.tag_collider(o,friction=.08,restitution=0)
    return o

# Real closed jagged rocks. The same mesh is used by rendering and collisions.
ROCK_EXCLUSIONS=[]
def rock(name,loc,scale,mat,col,seed=0,collision=False,sub=2):
    """Watertight angular basalt with correlated vertical faults, not smooth spheres."""
    rng=random.Random(seed);sides=13 if sub>=3 else 9
    heights=(-1,-.65,-.25,.15,.56,.85,1.0);profile=(.72,1.0,.93,.85,.73,.53,.26)
    angular=[rng.uniform(.77,1.24) for _ in range(sides)]
    vs=[];fs=[];phase=rng.uniform(0,math.tau)
    for k,(z,rad) in enumerate(zip(heights,profile)):
        for j in range(sides):
            a=j*math.tau/sides+phase
            rr=rad*angular[j]*(1+rng.uniform(-.075,.075))
            zz=z+(rng.uniform(-.075,.075) if 0<k<len(heights)-1 else rng.uniform(-.035,.035))
            vs.append((scale[0]*(rr*math.cos(a)+.14*z),scale[1]*(rr*math.sin(a)-.09*z),scale[2]*zz))
    for k in range(len(heights)-1):
        for j in range(sides):
            a=k*sides+j;b=k*sides+(j+1)%sides;c=a+sides;d=b+sides
            fs.extend(((a,b,d),(a,d,c)))
    fs.append(tuple(reversed(range(sides))));fs.append(tuple((len(heights)-1)*sides+j for j in range(sides)))
    o=mesh_obj(name,vs,fs,mat,col,False);o.location=loc
    o.rotation_euler=(rng.uniform(-.04,.04),rng.uniform(-.065,.065),rng.uniform(-.3,.3))
    if collision:
        colliders.tag_collider(o,friction=.12,restitution=0)
        ROCK_EXCLUSIONS.append((np.array(loc),np.array(scale)*1.17))
    return o

# Branched, needled silhouette, not a stack of smooth cones.
def pine_mesh(seed=0):
    rng=random.Random(seed);v=[];f=[];mi=[]
    def cone_between(a,b,r1,r2,sides,matid):
        a=Vector(a);b=Vector(b);axis=(b-a).normalized();u=axis.cross(Vector((0,1,0)))
        if u.length<.1:u=axis.cross(Vector((1,0,0)))
        u.normalize();w=axis.cross(u);k=len(v)
        for p,r in [(a,r1),(b,r2)]:
            for i in range(sides):
                t=i*math.tau/sides;v.append(tuple(p+r*(u*math.cos(t)+w*math.sin(t))))
        for i in range(sides):f.append((k+i,k+(i+1)%sides,k+sides+(i+1)%sides,k+sides+i));mi.append(matid)
        f.extend([tuple(k+i for i in reversed(range(sides))),tuple(k+sides+i for i in range(sides))]);mi.extend([matid,matid])
    cone_between((0,0,0),(.06,-.02,1),.022,.003,7,0)
    for lev in range(12):
        z=.15+.065*lev;radius=.25*(1-z)**.8
        for br in range(6):
            a=br*math.tau/6+lev*2.4+rng.uniform(-.2,.2);length=radius*rng.uniform(.65,1.25)
            end=Vector((math.cos(a)*length,math.sin(a)*length,z+.025))
            start=Vector((0,0,z+.03));cone_between(start,end,.005,.001,4,0)
            for q in (.28,.52,.74,1):
                p=start.lerp(end,q)
                for side in (-1,1):
                    tip=p+Vector((math.cos(a+side*.72)*length*.40,math.sin(a+side*.72)*length*.40,.02))
                    cone_between(p-Vector((0,0,.025)),tip+Vector((0,0,.06)),.028*(1-.4*z),.001,5,1)
    return v,f,mi

def add_pines(foliage,bark,col,rockmat):
    variants=[]
    for i in range(5):
        v,f,mi=pine_mesh(i+20);me=bpy.data.meshes.new('Coastal pine variant '+str(i));me.from_pydata(v,[],f);me.materials.append(bark);me.materials.append(foliage)
        me.polygons.foreach_set('material_index',mi);me.update();variants.append(me)
    rng=random.Random(17)
    # Ragged left coastal forest and the sheltered inner island.
    for i in range(72):
        y=rng.uniform(2,23);x=float(center(y)-width(y)-rng.uniform(1.4,6.7));z=float(bed_height(x,min(y,24)))
        if y>24:z=6.5+rng.uniform(0,3)
        h=rng.uniform(2,4.4)
        o=bpy.data.objects.new('SET | wind-shaped coastal pine %02d'%i,variants[i%5]);col.objects.link(o);o.location=(x,y,z-.2);o.scale=(h,h,h);o.rotation_euler[2]=rng.random()*math.tau
    for i,(x,y,z,h) in enumerate([(1.1,4.3,5.05,2.5),(1.7,4.6,4.8,1.8),(.4,4.7,4.8,1.6),(-7.8,13,7.6,2.9),(-8.8,13.8,7.8,2.0)]):
        o=bpy.data.objects.new('SET | sea-stack pine '+str(i),variants[i]);col.objects.link(o);o.location=(x,y,z);o.scale=(h,h,h)


def build_set(scene,basalt,env,plants):
    rng=random.Random(74);sim=next(c for c in scene.collection.children if c.name.startswith('01 |'))
    # Jagged left headland and the upper island flank the diagonal cascades.
    for side in (-1,1):
        for k,y in enumerate(np.linspace(-12,24,19)):
            if side>0 and y<9:continue
            x=float(center(y)+side*(width(y)+.72+rng.uniform(0,.55)))
            level=float(bed_level(y));sx=rng.uniform(1.2,2.4);sy=rng.uniform(1.3,2.25);sz=rng.uniform(1.55,3.05)
            rock('SET | basalt headland %s %02d'%(side,k),(x,y,level+sz*.70),(sx,sy,sz),basalt,sim,seed=100+k+side*19,collision=True,sub=3)
            for j in range(4):
                rock('SET | fractured ledge %s %02d %d'%(side,k,j),(x+rng.uniform(-1.2,1.2),y+rng.uniform(-1.4,1.4),level+.35),(sx*.48,sy*.57,.65),basalt,env,seed=k*8+j,sub=2)
    # Separate low right shoreline; the central sea stack stands in WATER.
    for k,y in enumerate(np.linspace(-9,10,13)):
        x=10.2+1.1*np.sin(y*.36);z=.7
        rock('COLLIDER | exposed right shoreline %02d'%k,(x,y,z),(rng.uniform(1.2,2.0),rng.uniform(1,1.8),rng.uniform(1.2,2)),basalt,sim,seed=920+k,collision=True,sub=3)
    rock('COLLIDER | tree-topped sea stack',(1.3,4.5,1.8),(1.65,1.35,3.4),basalt,sim,seed=314,collision=True,sub=3)
    rock('COLLIDER | lower falls splitter',(-3.0,-5.8,-.1),(1.6,1.4,1.9),basalt,sim,seed=5,collision=True,sub=3)
    rock('COLLIDER | upper cascade island',(-7.8,13,5.4),(1.4,1.35,2.6),basalt,sim,seed=12,collision=True,sub=3)
    for i,(x,y,s) in enumerate([(-3,5,.8),(-6.5,.2,.8),(-5.5,11,.65),(-8.6,18,.65),(-1.2,-2,.75),(4.3,.4,.62),(6.4,7.5,.8)]):
        rock('COLLIDER | current breaker '+str(i),(x,y,float(bed_height(x,y))+.1),(s,s*.8,s*.9),basalt,sim,seed=98+i,collision=True,sub=2)
    rock('SET | near left foreground',(-9,-13,.25),(6,4,3.5),basalt,env,seed=58,sub=3)
    rock('SET | near right foreground',(10,-9,1),(4.2,5,3.7),basalt,env,seed=61,sub=3)
    # Land mass beyond the top of the domain supports the forest, rather than floating trees.
    for i in range(14):
        x=-33+i*3.9;y=36+rng.uniform(-3,12)
        rock('SET | distant coastal ridge %02d'%i,(x,y,3),(rng.uniform(3,6),rng.uniform(4,7),rng.uniform(3,7)),basalt,env,seed=330+i,sub=3)
    for i,(x,y,s) in enumerate([(25,20,1.5),(31,40,2.0),(15,33,1.7),(44,65,2.1),(8,30,1.2)]):
        rock('SET | offshore stack '+str(i),(x,y,1.2+s*.4),(s,s*.9,s*1.8),basalt,env,seed=129+i,sub=2)
    foliage,_=mat_principled('Conifer | dark sea-green needles',(.012,.035,.020),.87)
    bark,_=mat_principled('Conifer | weathered bark',(.07,.043,.025),.95)
    add_pines(foliage,bark,plants,basalt)


def offshore_ocean(mat,col):
    # Native ocean stays outside the solver's foreground, never covers its liquid.
    me=bpy.data.meshes.new('Offshore ocean base');me.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)],[],[(0,1,2,3)])
    o=bpy.data.objects.new('SET | offshore ocean - native Ocean, NOT solver output',me);col.objects.link(o);me.materials.append(mat)
    mod=o.modifiers.new('Offshore sea | set dressing only','OCEAN');mod.geometry_mode='GENERATE';mod.resolution=9;mod.viewport_resolution=7;mod.spatial_size=170;mod.wave_scale=.42;mod.wind_velocity=15;mod.choppiness=.9;mod.time=4.2
    o.location=(98,55,1.05)
    o['provenance']='Native Blender Ocean modifier beyond x=13. Distant set dressing, not Liquid Warp simulation.'
    me2=me.copy();ob=bpy.data.objects.new('SET | distant north ocean - outside solver',me2);col.objects.link(ob)
    ocean=ob.modifiers.new('North sea scenery','OCEAN');ocean.geometry_mode='GENERATE';ocean.resolution=8;ocean.viewport_resolution=6;ocean.spatial_size=220;ocean.wave_scale=.35;ocean.wind_velocity=13;ocean.time=4.2;ob.location=(0,135,1.05)
    return o


def lighting(scene):
    world=bpy.data.worlds.new('Coastal dusk | clouded cool sky');world.use_nodes=True;scene.world=world
    n=world.node_tree.nodes;l=world.node_tree.links;bg=n.get('Background');bg.inputs['Strength'].default_value=.17
    sky=n.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING';sky.sun_elevation=math.radians(5);sky.sun_rotation=math.radians(118);sky.sun_intensity=.15;sky.air_density=1.3;sky.aerosol_density=1.5
    coords=n.new('ShaderNodeTexCoord');scale=n.new('ShaderNodeVectorMath');scale.operation='MULTIPLY';scale.inputs[1].default_value=(2.2,2.2,5.5);l.new(coords.outputs['Normal'],scale.inputs[0])
    cloud=noise_tex(n,2.5,5);l.new(scale.outputs[0],cloud.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.36;ramp.color_ramp.elements[0].color=(.026,.055,.096,1);ramp.color_ramp.elements[1].position=.67;ramp.color_ramp.elements[1].color=(.48,.56,.67,1)
    l.new(cloud.outputs['Fac'],ramp.inputs[0]);mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;l.new(sky.outputs['Color'],mix.inputs[1]);l.new(ramp.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],bg.inputs['Color'])
    ld=bpy.data.lights.new('LIGHT | sunset rim','SUN');ld.energy=2.0;ld.color=(1,.64,.37);ld.angle=.09
    o=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(o);o.rotation_euler=(math.radians(77),math.radians(-22),math.radians(-78))
    ld=bpy.data.lights.new('LIGHT | blue overcast softbox','AREA');ld.energy=9500;ld.color=(.49,.73,1);ld.shape='DISK';ld.size=42
    o=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(o);o.location=(-3,0,25);look_at(o,(-3,3,0))
    haze,_=mat_principled('SET | distant atmospheric haze',(.1,.1,.1));nt=haze.node_tree;nt.nodes.clear()
    v=nt.nodes.new('ShaderNodeVolumePrincipled');v.inputs['Density'].default_value=.004;v.inputs['Color'].default_value=(.34,.45,.57,1);v.inputs['Anisotropy'].default_value=.35;out=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(v.outputs['Volume'],out.inputs['Volume'])
    bpy.ops.mesh.primitive_cube_add(size=1,location=(0,70,10));o=bpy.context.object;o.name='SET | background sea haze';o.dimensions=(170,60,26);o.data.materials.append(haze);move_to(o,next(c for c in scene.collection.children if c.name.startswith('02 |')))


def configure_camera(scene):
    d=bpy.data.cameras.new('Coastal Cascade Camera');o=bpy.data.objects.new('CAMERA | Coastal Cascade',d);scene.collection.objects.link(o)
    o.location=(10,-22,10.6);look_at(o,(-1,6,3));d.lens=28;d.clip_end=1000;scene.camera=o
    scene.render.resolution_x=1600;scene.render.resolution_y=900;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    scene.render.engine='CYCLES';scene.cycles.samples=48;scene.cycles.use_denoising=True
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for dev in prefs.devices:dev.use=dev.type=='OPTIX'
    if not any(dev.use for dev in prefs.devices):raise RuntimeError('No Cycles GPU available')
    scene.cycles.device='GPU';scene.cycles.max_bounces=10;scene.cycles.transmission_bounces=8;scene.cycles.volume_bounces=1
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.5
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active;space.region_3d.view_perspective='CAMERA';space.region_3d.view_camera_zoom=18;space.overlay.show_overlays=False;space.shading.type='SOLID';space.shading.color_type='MATERIAL'
            if area.type=='PROPERTIES':area.spaces.active.context='SCENE'


def sample_seed(dx):
    spacing=dx*.57
    xs=np.arange(-18,11.4,spacing,dtype=np.float32);ys=np.arange(-10.6,22,spacing,dtype=np.float32)
    x,y=np.meshgrid(xs,ys,indexing='ij');x=x.ravel();y=y.ravel()
    channel=(np.abs(x-center(y))<width(y)-.72)
    basin=(x>center(y)) & (x<10.6) & (y<8.2)
    mask=channel|basin;x=x[mask];y=y[mask];h=bed_height(x,y)
    depth=np.where((x>center(y)+3.65)&(y>-4.4),np.maximum(.75,1.4-h),.88)
    pts=[]
    for layer in np.arange(.13,1.4,spacing):
        keep=layer<depth;z=h+layer;a=np.stack([x[keep],y[keep],z[keep]],axis=1)
        for loc,radii in ROCK_EXCLUSIONS:a=a[np.sum(((a-loc)/radii)**2,axis=1)>1]
        pts.append(a)
    return np.concatenate(pts).astype(np.float32)


def read_stats(scene):
    s=session();p=s.solver.particles;n=p.refresh_host_count()
    # Authoritative state is read, not a visual object count.
    report={'frame':scene.frame_current,'liquid_particles':int(n)}
    for name in ['position','positions','x']:
        arr=getattr(p,name,None)
        if arr is not None and hasattr(arr,'numpy'):
            a=arr.numpy()[:n];report['finite']=bool(np.isfinite(a).all());report['position_sha256']=hashlib.sha256(a.tobytes()).hexdigest();report['bounds']=[a.min(axis=0).tolist(),a.max(axis=0).tolist()];break
    report['scene_stats']={k:getattr(scene.fluidsim,k) for k in ('stat_particles','stat_whitewater','stat_particle_count','stat_whitewater_count','stat_frame_ms','stat_error') if hasattr(scene.fluidsim,k)}
    report['owned_outputs']=[{'name':o.name,'type':o.type,'vertices':len(o.data.vertices) if o.type=='MESH' else len(o.data.points) if o.type=='POINTCLOUD' else 0} for o in scene.objects if o.name.startswith('FluidSim') or o.get('fluidsim_output')]
    return report


def build():
    log('Creating isolated scene and GPU capability proof')
    wp.init();device=wp.get_device('cuda:0')
    assert device.is_cuda,'GPU-only demo requires CUDA'
    REPORT.update(blender=bpy.app.version_string,warp=wp.__version__,device=str(device),gpu=device.name,warp_cache=wp.config.kernel_cache_dir)
    scene=fresh_scene();scene.name='Coastal Cascade | Liquid Warp';scene['coastal_cascade_demo']=True
    sim=collection('01 | LIQUID WARP — simulation');env=collection('02 | SET — cliffs and offshore ocean');plants=collection('03 | SET — coastal conifers')
    basalt=rock_material();water=water_material('Water | turquoise cascade — SIMULATED');sea=water_material('Water | offshore backdrop — SET',True)
    foam,bs=mat_principled('Whitewater | solver foam and spray',(.84,.94,1),.37);bs.inputs['Subsurface Weight'].default_value=.045
    build_bed(basalt,sim);build_set(scene,basalt,env,plants);offshore_ocean(sea,env);lighting(scene);configure_camera(scene)
    lo=(-20,-12,-2.5);hi=(12,24,13)
    dom=make_domain(scene,lo,hi,'DOMAIN | Coastal Cascade');move_to(dom,sim);ds=dom.fluidsim_domain
    divisions=int(os.environ.get('COASTAL_DIVISIONS','256'));dx=36/divisions
    set_resolution(ds,lo,hi,dx);ds.max_particles=2_000_000;ds.min_substeps=2;ds.max_substeps=16
    ds.pressure_preconditioner='MULTIGRID';ds.pressure_iterations=200
    ds.whitewater_enabled=True;ds.max_whitewater=1_400_000;ds.ww_churn_gain=3.;ds.ww_churn_threshold=.12;ds.ww_emission_rate=7;ds.ww_min_speed=.3;ds.ww_life_min=1.2;ds.ww_life_max=3.5;ds.ww_surface_band=2.5;ds.ww_foam_band=1.5
    ds.surface_smooth_iterations=1;ds.surface_smooth_strength=.6
    fs=scene.fluidsim;fs.live=False;fs.liquid_view_mode='SURFACE';fs.show_whitewater=True;fs.surface_material=water;fs.whitewater_material=foam;fs.particle_material=water;fs.whitewater_display_scale=.9
    fs.particle_display_fraction=1.;fs.whitewater_display_fraction=1.
    scene.frame_start=1;scene.frame_end=240;scene.frame_set(1)
    pts=sample_seed(dx)
    if len(pts)>1_500_000:raise RuntimeError('Initial particle budget exceeded: '+str(len(pts)))
    seed=make_particle_seed(scene,pts,'SOURCE | Cascade initial water — owned seed');move_to(seed,sim);seed.fluidsim.enabled=True;seed.fluidsim.auto_inject_seed=True;seed.fluidsim.initial_velocity=(1.1,-2.8,0);seed.hide_set(True)
    y=21.3;cx=float(center(y));h=float(bed_level(y))
    src=make_source_box(scene,(cx-2.65,20.9,h+.15),(cx+2.65,22.0,h+.9),'SOURCE | continuous mountain inflow',emit_rate=8.,emission_mode='CONTINUOUS_INFLOW')
    move_to(src,sim);src.fluidsim.initial_velocity=(1.1,-3.1,0);src.fluidsim.enabled=True
    bpy.ops.mesh.primitive_cube_add(size=1,location=(-3,-11.35,4));sink=bpy.context.object;sink.name='SINK | open sea outlet';sink.dimensions=(34,.95,20);sources.tag_sink(sink);move_to(sink,sim);sink.fluidsim.enabled=True
    bpy.context.view_layer.update()
    log('Geometry authored; initializing %s seed particles at %.4f m voxels'%(f'{len(pts):,}',dx))
    REPORT['initial_seed_particles']=int(len(pts));REPORT['resolution_divisions']=divisions;REPORT['voxel_size']=dx
    assert bpy.ops.fluidsim.initialize()=={'FINISHED'}
    s=session();count=s.solver.particles.refresh_host_count()
    if count==0:
        assert bpy.ops.fluidsim.step(frames=1)=={'FINISHED'}
    fs.live=True
    for control in (dom,seed,src,sink):
        control.hide_render=True;control.hide_set(True)
    bpy.context.view_layer.update()
    # Save actual starting state and all owned sources. Reopen can initialize afresh.
    scene['fluidsim_playback_ready']=True
    doc=bpy.data.texts.new('README | COASTAL CASCADE')
    doc.write('COASTAL CASCADE — real Liquid Warp scene\n\nPress Space to advance live simulation from frame 1.\nThe turquoise cascade and whitewater are Liquid Warp output.\nThe offshore Ocean modifier, rock scenery, conifers and background haze are set dressing.\nThe input artwork is concept art, not a promised exact simulation result.\n\nScene Properties > FluidSim controls simulation. The DOMAIN object owns resolution.\nThe three named sources/seed/sink and tagged rock colliders drive real GPU dynamics.\n\nThis file was built with Blender 5.2.0 LTS, Liquid Warp source 0d1d4fa, Warp 1.17.0.\nUse the certified add-on. A rendering snapshot is not a performance benchmark.\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'Coastal_Cascade_LIVE.blend'),check_existing=False)
    REPORT['frames'].append(read_stats(scene))
    end=int(os.environ.get('COASTAL_FRAMES','72'))
    for frame in range(2,end+1):
        before=time.perf_counter();scene.frame_set(frame);wp.synchronize_device(device)
        if frame in (2,3,6,12,24,36,48,60,72,96) or frame==end:
            stat=read_stats(scene);stat['wall_seconds']=time.perf_counter()-before;REPORT['frames'].append(stat);log('Frame %d: %s liquid particles'%(frame,f"{stat['liquid_particles']:,}"))
    # Real solver state, not a pre-modelled water stand-in.
    fs.live=False
    REPORT['render_frame']=scene.frame_current
    for o in scene.objects:
        if o.type=='MESH' and (o.name.startswith('FluidSim') or o.get('fluidsim_output')):
            for f in o.data.polygons:f.use_smooth=True
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'Coastal_Cascade_RENDER_STATE.blend'),check_existing=False)
    scene.render.filepath=str(OUT/'Coastal_Cascade_preview.png');scene.render.resolution_percentage=60;scene.cycles.samples=24
    log('Rendering genuine simulated cascade preview on OptiX')
    with playback.suppress_frame_handlers():bpy.ops.render.render(write_still=True)
    # The latest output remains as a separate explicitly frozen render-state file.
    scene.render.resolution_percentage=100;scene.cycles.samples=64
    scene.render.filepath=str(OUT/'Coastal_Cascade.png')
    log('Rendering final 1600 x 900 image')
    with playback.suppress_frame_handlers():bpy.ops.render.render(write_still=True)
    REPORT['structural_checks']={
        'named_coastal_scene':scene.get('coastal_cascade_demo',False) is True,
        'cascade_bed':bpy.data.objects.get('CASCADE | continuous bed collider') is not None,
        'hero_camera':bpy.data.objects.get('CAMERA | Coastal Cascade') is not None,
        'cuda':device.is_cuda,
        'nonempty':all(f['liquid_particles']>0 for f in REPORT['frames']),
        'simulated_frame_advance':len(REPORT['frames'])>=3,
    }
    REPORT['ok']=all(REPORT['structural_checks'].values());REPORT['elapsed_seconds']=time.time()-T0
    (OUT/'build_report.json').write_text(json.dumps(REPORT,indent=2),encoding='utf8')
    if not REPORT['ok']:raise RuntimeError('Build acceptance failed')
    log('Scene and genuine render complete; cold-reopen verification is a separate gate')
    result_path=os.environ.get('FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT')
    if result_path:Path(result_path).write_text(json.dumps({'ok':True,'report':str(OUT/'build_report.json')}),encoding='utf8')
    # Exit only this owned review process, without an interactive save prompt.
    bpy.context.preferences.view.show_splash=False
    bpy.context.preferences.view.use_save_prompt=False
    bpy.ops.wm.quit_blender()

if __name__ == "__main__":
    try:
        build()
    except BaseException as exc:
        REPORT['error']=repr(exc);REPORT['traceback']=traceback.format_exc()
        (OUT/'build_failure.json').write_text(json.dumps(REPORT,indent=2),encoding='utf8')
        print(REPORT['traceback'],flush=True)
        raise
