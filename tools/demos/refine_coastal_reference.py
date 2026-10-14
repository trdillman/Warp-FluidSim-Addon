import bpy, math, random, os, json, time
import numpy as np
from pathlib import Path
from mathutils import Vector,noise
from tools.demos import build_coastal_reference as c
from tools.repo_addon_loader import enable_repo_addon
root=Path(os.environ['COASTAL_TASK_ROOT'])
enable_repo_addon(str(root/'source'),default_set=False,persistent=False)
bpy.ops.wm.open_mainfile(filepath=str(c.LIVE))
scene=bpy.context.scene;scene.fluidsim.live=False
rng=random.Random(27015)
c.progress('refining composition and coastline')
basalt=bpy.data.materials['Basalt | wet fractured coastal stone']
leaves=bpy.data.materials['Coastal fir | deep evergreen'];bark=bpy.data.materials['Coastal fir | bark']
land=bpy.data.collections.new('04 | Continuous surrounding coastline');scene.collection.children.link(land)
def landscape(side):
    nx=90;ny=112;v=[];f=[]
    for j in range(ny):
        y=-11+j*34/(ny-1)
        cx=-1.1*max(0,min(1,(y-1.5)/4))
        for i in range(nx):
            t=i/(nx-1);inner=cx+side*(2.63+.14*math.sin(1.5*y))
            x=inner+side*(22*t)
            edgeheight=c.bed_height(y)+.75
            rise=(1-math.exp(-t*16))
            broad=(2.0+.9*math.sin(y*.18+side)) if side<0 else (1.05+.6*math.sin(y*.22))
            n=noise.fractal(Vector((x*.65,y*.65,.37)),1.0,2.1,4)
            z=edgeheight+rise*broad+.32*n+.13*math.sin(x*2.4-y*.8)
            if side>0 and y>9:z-=min(5,(y-9)*.7)
            if side<0 and y>12:z+=1.8*math.sin((y-12)*.18)
            v.append((x,y,z))
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i
            if side>0:f.extend(((a,a+1,a+nx+1),(a,a+nx+1,a+nx)))
            else:f.extend(((a,a+nx+1,a+1),(a,a+nx,a+nx+1)))
    ob=c.mesh_obj('Continuous left headland' if side<0 else 'Continuous right shoreline',v,f,basalt,land)
    return v,nx,ny
left,nx,ny=landscape(-1);right,_,_=landscape(1)
# Finely displaced, triangulated basalt replaces the coarse polygonal silhouette.
tex=bpy.data.textures.new('Coastal stone | multi-scale fracture',type='CLOUDS');tex.noise_scale=.24;tex.noise_depth=2
for ob in list(scene.objects):
    if ob.type!='MESH' or not ob.data.materials or ob.data.materials[0]!=basalt:continue
    if ob.name.startswith(('Collision','Continuous')):continue
    sub=ob.modifiers.new('Geological detail subdivision','SUBSURF');sub.subdivision_type='SIMPLE';sub.levels=2;sub.render_levels=3
    dis=ob.modifiers.new('Fractured basalt micro-relief','DISPLACE');dis.texture=tex;dis.texture_coords='GLOBAL';dis.strength=.105;dis.mid_level=.51
    tri=ob.modifiers.new('Angular broken faces','TRIANGULATE')
    for poly in ob.data.polygons:poly.use_smooth=True
# Replace conical tree proxies with irregular radial branches and individual needle tufts.
def real_tree(name,loc,h,seed):
    h=float(h);loc=tuple(float(x) for x in loc);r=random.Random(seed);verts=[];faces=[];mats=[]
    def cylinder(a,b,rad0,rad1,sides=5):
        a=Vector(a);b=Vector(b);d=(b-a).normalized();u=d.cross(Vector((0,1,0))).normalized();w=d.cross(u)
        ix=len(verts)
        for pos,rad in ((a,rad0),(b,rad1)):
            for k in range(sides):verts.append(tuple(pos+rad*(u*math.cos(k*2*math.pi/sides)+w*math.sin(k*2*math.pi/sides))))
        for k in range(sides):faces.append((ix+k,ix+(k+1)%sides,ix+sides+(k+1)%sides,ix+sides+k));mats.append(1)
    p=Vector(loc);cylinder(p,p+Vector((.08*h,0,h)),.018*h,.004*h,7)
    for layer in range(15):
        z=(.15+.80*layer/15)*h;t=layer/15
        for b in range(5):
            ang=layer*2.399+b*2*math.pi/5+r.uniform(-.25,.25)
            length=.25*h*(1-t)**.8*r.uniform(.65,1.25)
            stem=p+Vector((.08*z,0,z));tip=stem+Vector((math.cos(ang)*length,math.sin(ang)*length,length*.12))
            cylinder(stem,tip,.006*h*(1-t),.001*h)
            for k in range(1,6):
                center=stem.lerp(tip,k/5)
                for fan in range(4):
                    a=ang+(fan-1.5)*.67;size=.037*h*(1-.6*t)*r.uniform(.7,1.2)
                    direct=Vector((math.cos(a),math.sin(a),r.uniform(-.1,.65)))
                    across=Vector((-math.sin(a),math.cos(a),.08))
                    index=len(verts);verts.extend([tuple(center-across*size*.40),tuple(center+across*size*.40),tuple(center+direct*size*2)])
                    faces.append((index,index+1,index+2));mats.append(0)
    ob=c.mesh_obj(name,verts,faces,leaves,land);ob.data.materials.append(bark)
    for poly,mi in zip(ob.data.polygons,mats):poly.material_index=mi
    return ob
oldtrees=[ob for ob in scene.objects if ob.type=='MESH' and ob.data.materials and ob.data.materials[0]==leaves]
for i,ob in enumerate(oldtrees):
    pts=np.array([v.co[:] for v in ob.data.vertices]);lo=pts.min(axis=0);hi=pts.max(axis=0);h=hi[2]-lo[2];loc=((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]);name=ob.name
    bpy.data.objects.remove(ob,do_unlink=True);real_tree(name,loc,h,2500+i)
for i in range(115):
    bank=left if i<90 else right
    index=rng.randrange(len(bank));x,y,z=bank[index]
    if y<-.5 or y>18 or abs(x)>13:continue
    real_tree(f'Headland coastal forest {i}',(x,y,z),rng.uniform(.85,2.4),6000+i)
c.progress('coastline and detailed forest authored')
# An explicit downstream sink removes the rectangular tank-wall waterfall.
sink=c.make_source_box(scene,(-4.4,-6.38,-1.13),(4.4,-5.85,4.0),'Outflow | downstream beyond camera',emit_rate=0,emission_mode='INITIAL_FILL')
sink.fluidsim.role='SINK';sink.fluidsim.sink_region_kind='ANALYTIC';sink.hide_render=True;sink.display_type='WIRE'
# Keep frame coverage immersive instead of presenting the domain as a miniature tray.
scene.camera.location=(1.25,-5.35,3.25);c.point_at(scene.camera,(-.6,3.2,.75));scene.camera.data.lens=26
scene.camera.data.clip_start=.03
fs=scene.fluidsim;ds=fs.domain_object.fluidsim_domain
ds.resolution_divisions=228;ds.max_particles=1800000;ds.max_whitewater=700000
ds.ww_emission_rate=13;ds.ww_life_min=.8;ds.ww_life_max=2.5
fs.whitewater_display_scale=.75
ds.surface_smooth_iterations=2;ds.surface_smooth_strength=.65
scene.cycles.samples=96;scene.render.resolution_x=1600;scene.render.resolution_y=900
# Physical dusk sky, with structured cloud modulation rather than a flat grey world.
n=scene.world.node_tree.nodes;l=scene.world.node_tree.links;n.clear()
out=n.new('ShaderNodeOutputWorld');bg=n.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=.23
sky=n.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING';sky.sun_disc=False;sky.sun_elevation=math.radians(7);sky.sun_rotation=math.radians(135);sky.air_density=1.2
texc=n.new('ShaderNodeTexCoord');cloud=n.new('ShaderNodeTexNoise');cloud.inputs['Scale'].default_value=4;cloud.inputs['Detail'].default_value=6
l.new(texc.outputs['Normal'],cloud.inputs['Vector'])
cr=n.new('ShaderNodeValToRGB');cr.color_ramp.elements[0].position=.3;cr.color_ramp.elements[0].color=(.13,.2,.31,1);cr.color_ramp.elements[1].position=.7;cr.color_ramp.elements[1].color=(.68,.75,.85,1)
l.new(cloud.outputs['Fac'],cr.inputs[0]);mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.8
l.new(sky.outputs[0],mix.inputs[1]);l.new(cr.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],bg.inputs[0]);l.new(bg.outputs[0],out.inputs[0])
scene.view_settings.exposure=.25
for area in [a for s in bpy.data.screens for a in s.areas if a.type=='VIEW_3D']:
    area.spaces.active.shading.type='MATERIAL';area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.overlay.show_overlays=False
scene.frame_set(1);fs.live=True
bpy.ops.wm.save_as_mainfile(filepath=str(c.LIVE),check_existing=False,compress=True)
c.progress('refined live scene saved',objects=len(scene.objects),resolution=ds.resolution_divisions)
c.verify_and_render(scene)
print('COASTAL-LANE-PASS',flush=True)