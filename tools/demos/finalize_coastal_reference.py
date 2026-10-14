"""Final coastal shot: widen the bay, open the sea view, use particle-driven foam."""
import bpy, os, json, math
import numpy as np
from pathlib import Path
from mathutils import Vector
from tools.demos import build_coastal_reference as c
from tools.repo_addon_loader import enable_repo_addon
root=Path(os.environ['COASTAL_TASK_ROOT'])
enable_repo_addon(str(root/'source'),default_set=False,persistent=False)
bpy.ops.wm.open_mainfile(filepath=str(c.LIVE));scene=bpy.context.scene;fs=scene.fluidsim;fs.live=False
c.progress('opening coastal bay and completing materials')
# Remove the upper right enclosing cliff: the reference is an open coastal bay, not a slot canyon.
for name in ['Bank +1 04','Bank +1 05','Bank +1 06','Bank +1 07','Continuous right shoreline']:
    ob=bpy.data.objects.get(name)
    if ob:bpy.data.objects.remove(ob,do_unlink=True)
for name in ['Bank +1 03','Bank +1 08']:
    ob=bpy.data.objects.get(name)
    if ob:
        pts=np.array([v.co[:] for v in ob.data.vertices]);center=(pts.min(axis=0)+pts.max(axis=0))/2;base=float(pts[:,2].min())
        for v in ob.data.vertices:
            v.co.x=float(center[0]+(v.co.x-center[0])*.68);v.co.y=float(center[1]+(v.co.y-center[1])*.72);v.co.z=base+(v.co.z-base)*.58
# Remove decorations belonging to the removed bank; preserve their original files in the preceding candidate.
for ob in list(scene.objects):
    if ob.type!='MESH' or ob.name.startswith(('FluidSim','Water source','Coastal Reference')):continue
    if ob.name.startswith(('Basalt fragment','Wind-shaped coastal fir','Headland coastal forest')):
        pts=np.array([v.co[:] for v in ob.data.vertices]);center=pts.mean(axis=0)
        if 1.1<center[0]<9 and -.3<center[1]<5.7:bpy.data.objects.remove(ob,do_unlink=True)
# Widen the actual simulation domain and its watertight submerged bed.
dom=fs.domain_object;dom.location.x=2.0;dom.dimensions.x=12.9
bed=bpy.data.objects['Collision | stepped submerged bed']
for v in bed.data.vertices:
    if v.co.x>0:v.co.x*=1.98
    if v.co.z>-1.2 and v.co.x>.7 and v.co.y>-.6:
        fac=min(1,(v.co.x-.7)/1.45);v.co.z=v.co.z*(1-fac)+(-.67+.025*math.sin(v.co.x*2+v.co.y))*fac
bed.data.update()
new=c.make_source_box(scene,(2.2,-1.8,-.60),(8.1,5.10,.08),'Water source | open coastal basin',emit_rate=8,emission_mode='INITIAL_FILL')
new.fluidsim.initial_velocity=(-.12,-.55,0)
fs.domain_object=dom;ds=dom.fluidsim_domain;ds.max_particles=2600000;ds.max_whitewater=1400000;ds.ww_emission_rate=6
# Broader downstream removal is out of the shot, not a visible domain wall.
sink=bpy.data.objects['Outflow | downstream beyond camera'];sink.dimensions.x=12.8;sink.location.x=2.0
for ob in scene.objects:
    if ob.name.startswith('Water source | continuous'):ob.fluidsim.initial_velocity=(.12,-3.25,0)
# Distant water remains explicitly scenic and outside the foreground solver footprint.
ocean=bpy.data.objects['Distant ocean | outside simulation domain']
for v in ocean.data.vertices:v.co.y-=3.5;v.co.z+=.45
far=bpy.data.collections['03 | Distant environment (not simulated)'];om=ocean.data.materials[0]
v=[];f=[]
for j in range(15):
    y=-3+j*9/14
    for i in range(40):
        x=8.5+i*2.3;v.append((x,y,.03+.035*math.sin(1.4*x+.9*y)))
for j in range(14):
    for i in range(39):a=j*40+i;f.append((a,a+1,a+41,a+40))
c.mesh_obj('Outside-domain coastal ocean extension | scenic only',v,f,om,far)
# Dark, wet basalt, not pale icy cliff faces.
basalt=bpy.data.materials['Basalt | wet fractured coastal stone']
ramps=[n for n in basalt.node_tree.nodes if n.type=='VALTORGB']
for e,color in zip(sorted(ramps[0].color_ramp.elements,key=lambda e:e.position),[(.004,.006,.008,1),(.025,.031,.034,1),(.085,.077,.064,1)]):e.color=color
for ob in scene.objects:
    if ob.name.startswith('Continuous'):
        for p in ob.data.polygons:p.use_smooth=True
# Correct clear-water transmission and volume tint; surface coloration is no longer opaque blue.
water=bpy.data.materials['Liquid Warp | turquoise transmitting water'];n=water.node_tree.nodes;l=water.node_tree.links
p=n.get('Principled BSDF');p.inputs['Base Color'].default_value=(.91,.98,1,1);p.inputs['Roughness'].default_value=.11
for node in n:
    if node.type=='VOLUME_ABSORPTION':node.inputs['Color'].default_value=(.065,.56,.64,1);node.inputs['Density'].default_value=.45
attribute=n.new('ShaderNodeAttribute');attribute.attribute_name='coastal_foam';attribute.label='Density from REAL whitewater proximity'
tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=95;tex.inputs['Detail'].default_value=3
ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.30;ramp.color_ramp.elements[1].position=.60
l.new(tex.outputs['Fac'],ramp.inputs[0]);mul=n.new('ShaderNodeMath');mul.operation='MULTIPLY';l.new(attribute.outputs['Fac'],mul.inputs[0]);l.new(ramp.outputs['Color'],mul.inputs[1])
diff=n.new('ShaderNodeBsdfDiffuse');diff.inputs['Color'].default_value=(.87,.96,1,1);diff.inputs['Roughness'].default_value=.3
mix=n.new('ShaderNodeMixShader');l.new(mul.outputs[0],mix.inputs[0]);l.new(p.outputs[0],mix.inputs[1]);l.new(diff.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],n.get('Material Output').inputs['Surface'])
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.07
bpy.data.lights['Sky opening | cold softbox'].energy=850
bpy.data.lights['Cove bounce'].energy=180
bpy.data.lights['Sunset | grazing warm light'].energy=1.65
scene.view_settings.exposure=.1
scene.camera.location=(4.0,-8.4,4.6);c.point_at(scene.camera,(-.4,2.4,.8));scene.camera.data.lens=25
scene.cycles.samples=64;scene.render.resolution_x=1600;scene.render.resolution_y=900

def attach_foam(scene):
    surf=bpy.data.objects.get('FluidSim Surface');ww=bpy.data.objects.get('FluidSim Whitewater')
    assert surf and ww,'Real surface/whitewater outputs required before adding foam shading'
    group=bpy.data.node_groups.new('Coastal | foam from real whitewater (not procedural emitters)','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    n=group.nodes;l=group.links;gin=n.new('NodeGroupInput');gout=n.new('NodeGroupOutput')
    oi=n.new('GeometryNodeObjectInfo');oi.transform_space='RELATIVE';oi.inputs['Object'].default_value=ww
    radius=n.new('GeometryNodeInputRadius');compare=n.new('FunctionNodeCompare');compare.data_type='FLOAT';compare.operation='GREATER_THAN';compare.inputs[1].default_value=.0001
    l.new(radius.outputs[0],compare.inputs[0]);separate=n.new('GeometryNodeSeparateGeometry');separate.domain='POINT';l.new(oi.outputs['Geometry'],separate.inputs['Geometry']);l.new(compare.outputs['Result'],separate.inputs['Selection'])
    nearest=n.new('GeometryNodeSampleNearest');nearest.domain='POINT';pos=n.new('GeometryNodeInputPosition');l.new(separate.outputs['Selection'],nearest.inputs['Geometry']);l.new(pos.outputs[0],nearest.inputs['Sample Position'])
    sample=n.new('GeometryNodeSampleIndex');sample.data_type='FLOAT_VECTOR';sample.domain='POINT';l.new(separate.outputs['Selection'],sample.inputs['Geometry']);l.new(pos.outputs[0],sample.inputs['Value']);l.new(nearest.outputs['Index'],sample.inputs['Index'])
    distance=n.new('ShaderNodeVectorMath');distance.operation='DISTANCE';l.new(pos.outputs[0],distance.inputs[0]);l.new(sample.outputs['Value'],distance.inputs[1])
    remap=n.new('ShaderNodeMapRange');remap.inputs['From Min'].default_value=.01;remap.inputs['From Max'].default_value=.13;remap.inputs['To Min'].default_value=1;remap.inputs['To Max'].default_value=0
    l.new(distance.outputs['Value'],remap.inputs['Value']);store=n.new('GeometryNodeStoreNamedAttribute');store.domain='POINT';store.data_type='FLOAT';store.inputs['Name'].default_value='coastal_foam'
    l.new(gin.outputs['Geometry'],store.inputs['Geometry']);l.new(remap.outputs['Result'],store.inputs['Value']);l.new(store.outputs['Geometry'],gout.inputs[0])
    for i,node in enumerate(n):node.location=(i*180,0)
    mod=surf.modifiers.get('Coastal | particle-driven surface foam') or surf.modifiers.new('Coastal | particle-driven surface foam','NODES');mod.node_group=group
    scene['coastal_foam_provenance']='Surface foam shading is derived from nearest nonzero-radius solver whitewater points; noise only breaks up its appearance.'
    c.progress('particle-driven foam attached',whitewater_type=ww.type,whitewater_attributes=[a.name for a in ww.data.attributes],whitewater_modifiers=[m.name for m in ww.modifiers])

c.scene_finish_hook=attach_foam
scene.frame_set(1);fs.live=True
bpy.ops.wm.save_as_mainfile(filepath=str(c.LIVE),check_existing=False,compress=True)
c.verify_and_render(scene)
print('COASTAL-LANE-PASS',flush=True)