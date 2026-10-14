"""Procedural coastal rock, conifer, terrain and lighting assets. No import-time scene changes."""
import bpy,math,random
import numpy as np
from mathutils import Vector,noise
from fluidSim_warp_blender51.blender import colliders
from tools.demos.build_demo_scenes import fresh_scene,make_domain,set_resolution,make_particle_seed,make_source_box,look_at
COLS={}
ROCK_EXCLUSIONS=[]


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
    m,bs=mat_principled(name,(.016,.24,.30) if not offshore else (.014,.055,.078),.09,transmission=.82 if not offshore else .25)
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
    x=np.asarray(x);y=np.asarray(y)
    cross=np.abs(x-center(y));w=width(y)
    bank=4.8*smooth(cross,w-.4,w+1.35)
    rough=.10*np.sin(1.6*x+.2*y)*np.sin(y*1.4)+.07*np.sin(x*4.1-y*1.7)
    return bed_level(y)+bank+rough

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

def rock(name,loc,scale,mat,col,seed=0,collision=False,sub=2):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=sub,radius=1,location=loc)
    o=bpy.context.object;o.name=name;move_to(o,col)
    rng=random.Random(seed)
    for v in o.data.vertices:
        p=v.co.copy()
        n=noise.noise_vector(p*2.9+Vector((seed*.23,1.2,.3)))
        # angular outcrops; horizontal strata and non-uniform ridges
        factor=1+.19*n.x+.09*math.sin(p.z*14+seed)
        v.co=(p.x*scale[0]*factor+.1*scale[0]*p.z,
              p.y*scale[1]*factor,p.z*scale[2]*(1+.12*n.z))
    o.data.materials.append(mat)
    o.rotation_euler=(.07*rng.uniform(-1,1),.12*rng.uniform(-1,1),rng.uniform(-.35,.35))
    if sub>=3:
        for f in o.data.polygons:f.use_smooth=True
    if collision:
        colliders.tag_collider(o,friction=.16,restitution=0)
        ROCK_EXCLUSIONS.append((np.array(loc),np.array(scale)*1.25))
    return o

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
        y=rng.uniform(2,38);x=float(center(y)-width(y)-rng.uniform(1.4,6.7));z=float(bed_height(x,min(y,24)))
        if y>24:z=6.5+rng.uniform(0,3)
        h=rng.uniform(2,4.4)
        o=bpy.data.objects.new('SET | wind-shaped coastal pine %02d'%i,variants[i%5]);col.objects.link(o);o.location=(x,y,z-.2);o.scale=(h,h,h);o.rotation_euler[2]=rng.random()*math.tau
    for i,(x,y,z,h) in enumerate([(1.1,4.3,5.05,2.5),(1.7,4.6,4.8,1.8),(.4,4.7,4.8,1.6),(-7.8,13,7.6,2.9),(-8.8,13.8,7.8,2.0)]):
        o=bpy.data.objects.new('SET | sea-stack pine '+str(i),variants[i]);col.objects.link(o);o.location=(x,y,z);o.scale=(h,h,h)

def build_set(scene,basalt,env,plants):
    rng=random.Random(74)
    # Foreground and flanking outcrops follow the actual channel banks.
    for side in [-1,1]:
        for k,y in enumerate(np.linspace(-12,24,18)):
            x=float(center(y)+side*(width(y)+1.15+rng.uniform(0,.8)))
            level=float(bed_level(y));sx=rng.uniform(1.2,2.9);sy=rng.uniform(1.1,2.5);sz=rng.uniform(1.7,3.8)
            rock('SET | basalt headland %s %02d'%(side,k),(x,y,level+sz*.9),(sx,sy,sz),basalt,env,seed=100+k+side*19,sub=3)
            for j in range(2):
                rock('SET | fractured ledge %s %02d %d'%(side,k,j),(x+rng.uniform(-1,1),y+rng.uniform(-1,1),level+.3),(sx*.6,sy*.7,.7),basalt,env,seed=k*8+j,sub=2)
    # A few prominent rocks, including the tree-topped central stack, are genuine colliders.
    rock('COLLIDER | tree-topped sea stack',(1.3,4.5,1.8),(1.45,1.25,3.7),basalt,COLS['01 | LIQUID WARP — simulation'],seed=314,collision=True,sub=3)
    rock('COLLIDER | lower falls splitter',(-3.0,-5.8,-.1),(1.6,1.4,1.9),basalt,COLS['01 | LIQUID WARP — simulation'],seed=5,collision=True,sub=3)
    rock('COLLIDER | upper cascade island',(-7.8,13,5.4),(1.4,1.35,2.6),basalt,COLS['01 | LIQUID WARP — simulation'],seed=12,collision=True,sub=3)
    for i,(x,y,s) in enumerate([(-3,5,.8),(-6.5,.2,.8),(-5.5,11,.65),(-8.6,18,.65),(-1.2,-2,.75)]):
        rock('COLLIDER | current breaker '+str(i),(x,y,float(bed_level(y))+.1),(s,s*.8,s*.9),basalt,COLS['01 | LIQUID WARP — simulation'],seed=98+i,collision=True,sub=2)
    # Masses outside the solver frame conceal the constructed channel ends.
    rock('SET | near left foreground',(-9,-13,.25),(6,4,3.5),basalt,env,seed=58,sub=3)
    rock('SET | near right foreground',(10,-9,1),(4.2,5,3.7),basalt,env,seed=61,sub=3)
    # Receding rocky coastline, progressively lower contrast through the world/atmosphere.
    for i in range(19):
        x=-28+i*4.7;y=35+rng.uniform(0,18);z=rng.uniform(1,3)
        rock('SET | distant coastal ridge %02d'%i,(x,y,z),(rng.uniform(3,8),rng.uniform(3,7),rng.uniform(3,7)),basalt,env,seed=330+i,sub=3)
    for i,(x,y,s) in enumerate([(25,20,2.0),(31,40,3),(15,33,1.9),(44,65,3.2),(8,30,1.2)]):
        rock('SET | offshore stack '+str(i),(x,y,s*.4),(s,s*.9,s*1.8),basalt,env,seed=129+i,sub=2)
    foliage,_=mat_principled('Conifer | dark sea-green needles',(.012,.035,.020),.87)
    bark,_=mat_principled('Conifer | weathered bark',(.07,.043,.025),.95)
    add_pines(foliage,bark,plants,basalt)

def offshore_ocean(mat,col):
    # Native Ocean modifier: ONLY distant, non-simulated background sea.
    me=bpy.data.meshes.new('Offshore ocean base');me.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)],[],[(0,1,2,3)])
    o=bpy.data.objects.new('SET | offshore ocean — native Ocean, NOT solver output',me);col.objects.link(o);me.materials.append(mat)
    mod=o.modifiers.new('Offshore sea | set dressing only','OCEAN');mod.geometry_mode='GENERATE';mod.resolution=9;mod.viewport_resolution=7;mod.spatial_size=170;mod.wave_scale=.65;mod.wind_velocity=17;mod.choppiness=1.1;mod.time=4.2
    o.location=(60,55,-1.65)
    o['provenance']='Native Blender Ocean modifier. Distant set dressing, not Liquid Warp simulation.'
    return o

def lighting(scene):
    world=bpy.data.worlds.new('Coastal dusk | clouded cool sky');world.use_nodes=True;scene.world=world
    n=world.node_tree.nodes;l=world.node_tree.links
    bg=n.get('Background');bg.inputs['Strength'].default_value=.40
    sky=n.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING';sky.sun_elevation=math.radians(7);sky.sun_rotation=math.radians(118);sky.sun_intensity=.5;sky.air_density=1.5;sky.aerosol_density=3.0
    l.new(sky.outputs['Color'],bg.inputs['Color'])
    ld=bpy.data.lights.new('LIGHT | sunset rim','SUN');ld.energy=2.3;ld.color=(1,.60,.32);ld.angle=.11
    o=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(o);o.rotation_euler=(math.radians(77),math.radians(-22),math.radians(-78))
    ld=bpy.data.lights.new('LIGHT | blue overcast softbox','AREA');ld.energy=10000;ld.color=(.50,.72,1);ld.shape='DISK';ld.size=45
    o=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(o);o.location=(-3,0,27);look_at(o,(-3,3,0))
    # Subtle haze only behind the hero cascade. Does not fabricate solver mist.
    haze,_=mat_principled('SET | distant atmospheric haze',(.1,.1,.1))
    nt=haze.node_tree;nt.nodes.clear();v=nt.nodes.new('ShaderNodeVolumePrincipled');v.inputs['Density'].default_value=.007;v.inputs['Color'].default_value=(.34,.45,.57,1);v.inputs['Anisotropy'].default_value=.4;out=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(v.outputs['Volume'],out.inputs['Volume'])
    bpy.ops.mesh.primitive_cube_add(size=1,location=(0,62,10));o=bpy.context.object;o.name='SET | background sea haze';o.dimensions=(170,60,26);o.data.materials.append(haze);move_to(o,COLS['02 | SET — cliffs and offshore ocean'])

def configure_camera(scene):
    d=bpy.data.cameras.new('Coastal Cascade Camera');o=bpy.data.objects.new('CAMERA | Coastal Cascade',d);scene.collection.objects.link(o)
    o.location=(13,-28,18);look_at(o,(-3,6,2.3));d.lens=37;d.clip_end=1000;scene.camera=o
    scene.render.resolution_x=1600;scene.render.resolution_y=900;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    scene.render.engine='CYCLES';scene.cycles.samples=48;scene.cycles.use_denoising=True
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    for dev in prefs.devices:dev.use=dev.type!='CPU'
    if not any(dev.use for dev in prefs.devices):raise RuntimeError('No Cycles GPU available')
    scene.cycles.device='GPU';scene.cycles.max_bounces=8;scene.cycles.transmission_bounces=6;scene.cycles.volume_bounces=1
    scene.view_settings.view_transform='AgX';scene.view_settings.exposure=.3
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active;space.region_3d.view_perspective='CAMERA';space.overlay.show_overlays=False;space.shading.type='MATERIAL'
            if area.type=='PROPERTIES':area.spaces.active.context='SCENE'

SIM_COLLECTION = "01 | LIQUID WARP \u2014 simulation"
