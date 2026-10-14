import bpy, os, json, math
import numpy as np
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from tools.demos import build_coastal_reference as c
from tools.repo_addon_loader import enable_repo_addon
root=Path(os.environ['COASTAL_TASK_ROOT'])
enable_repo_addon(str(root/'source'),default_set=False,persistent=False)
bpy.ops.wm.open_mainfile(filepath=str(c.LIVE));scene=bpy.context.scene;fs=scene.fluidsim;fs.live=False
c.progress('matching reference landmarks and correcting foam classification')
# Put the pine-topped stack in the open bay, with the wider silhouette in the reference.
hero=bpy.data.objects['Hero | central sea stack']
for v in hero.data.vertices:
    v.co.x=3.84+(v.co.x-1.04)*1.45;v.co.y=2.0+(v.co.y-2.0)*1.40
for ob in scene.objects:
    if ob.name.startswith('Sea-stack fir'):
        for v in ob.data.vertices:v.co.x=3.84+(v.co.x-1.04)*1.45;v.co.y=2.0+(v.co.y-2.0)*1.40
right=bpy.data.objects['Hero | foreground right outcrop']
for v in right.data.vertices:v.co.x+=1.7;v.co.y-=1.8
for i in range(8,12):
    ob=bpy.data.objects.get(f'Distant headland {i}')
    if ob:bpy.data.objects.remove(ob,do_unlink=True)
cam=scene.camera;cam.location=(-2.50364,-7.86655,3.41869)
c.point_at(cam,(3.70986,5.17246,2.73933));cam.data.lens=24.18797;cam.data.shift_y=-.198463
bpy.context.view_layer.update()
anchors={name:list(world_to_camera_view(scene,cam,Vector(p)))[:2] for name,p in {'stack_top':(4.2,1.8,3),'stack_base':(3.84,2,.2),'upper_cascade':(-1.1,6.1,2.4),'foreground_drop':(0,-2.2,0)}.items()}
(root/'evidence'/'camera_landmarks.json').write_text(json.dumps(anchors,indent=2))
# More turquoise transmission, less broad white foam from submerged bubbles.
water=bpy.data.materials['Liquid Warp | turquoise transmitting water'];n=water.node_tree.nodes;l=water.node_tree.links
n.get('Principled BSDF').inputs['Base Color'].default_value=(.11,.49,.61,1)
for node in n:
    if node.type=='VOLUME_ABSORPTION':node.inputs['Density'].default_value=.85
mix=next(node for node in n if node.type=='MIX_SHADER');old=mix.inputs[0].links[0].from_socket
strength=n.new('ShaderNodeMath');strength.operation='MULTIPLY';strength.inputs[1].default_value=.52;l.new(old,strength.inputs[0]);l.new(strength.outputs[0],mix.inputs[0])
fs.whitewater_display_scale=.60
wn=scene.world.node_tree.nodes
for node in wn:
    if node.type=='VALTORGB':
        node.color_ramp.elements[0].color=(.025,.055,.09,1);node.color_ramp.elements[1].color=(.52,.59,.69,1)
    if node.type=='MIX_RGB':node.inputs[0].default_value=1
# Far atmospheric haze is environmental set dressing; the liquid and spray remain solver-derived.
bpy.ops.mesh.primitive_cube_add(size=1,location=(0,38,8));haze=bpy.context.object;haze.name='Far coastal atmosphere | not simulated';haze.dimensions=(110,50,24)
m=bpy.data.materials.new('Distant sea haze');m.use_nodes=True;n=m.node_tree.nodes;n.clear();v=n.new('ShaderNodeVolumeScatter');v.inputs['Color'].default_value=(.38,.50,.64,1);v.inputs['Density'].default_value=.006;v.inputs['Anisotropy'].default_value=.20;o=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(v.outputs[0],o.inputs['Volume']);haze.data.materials.append(m)
# Reuse the authored foam graph while excluding submerged BUBBLE particles.
from fluidSim_warp_blender51.sim.whitewater_types import WhitewaterType
foam_type=int(WhitewaterType.FOAM)
def hook(scene):
    surf=bpy.data.objects['FluidSim Surface'];ww=bpy.data.objects['FluidSim Whitewater']
    group=bpy.data.node_groups.get('Coastal | foam from real whitewater (not procedural emitters)')
    if group is None:
        # The render snapshot retains the proven graph; append only that datablock, not a frozen simulation.
        with bpy.data.libraries.load(str(root/'deliverables'/'Coastal_Rapids_RenderReview.blend'),link=False) as (available,target):
            target.node_groups=[name for name in available.node_groups if name.startswith('Coastal | foam from real whitewater')]
        group=next(g for g in target.node_groups if g is not None)
    n=group.nodes;l=group.links
    oi=next(x for x in n if x.bl_idname=='GeometryNodeObjectInfo');oi.inputs['Object'].default_value=ww
    sep=next(x for x in n if x.bl_idname=='GeometryNodeSeparateGeometry')
    positive=sep.inputs['Selection'].links[0].from_socket
    attribute=n.new('GeometryNodeInputNamedAttribute');attribute.data_type='INT';attribute.inputs['Name'].default_value='fluidsim_whitewater_type'
    equals=n.new('ShaderNodeMath');equals.operation='COMPARE';equals.inputs[1].default_value=foam_type;equals.inputs[2].default_value=.1;l.new(attribute.outputs['Attribute'],equals.inputs[0])
    both=n.new('ShaderNodeMath');both.operation='MULTIPLY';l.new(positive,both.inputs[0]);l.new(equals.outputs[0],both.inputs[1]);l.new(both.outputs[0],sep.inputs['Selection'])
    remap=next(x for x in n if x.bl_idname=='ShaderNodeMapRange');remap.inputs['From Min'].default_value=.004;remap.inputs['From Max'].default_value=.07
    mod=surf.modifiers.get('Coastal | particle-driven surface foam') or surf.modifiers.new('Coastal | particle-driven surface foam','NODES');mod.node_group=group
    scene['coastal_foam_provenance']='Only actual nonzero-radius FOAM-class solver particles drive surface-foam shading; submerged bubbles are excluded.'
    c.progress('verified FOAM-only surface shading',foam_enum=foam_type,graph=group.name)
c.scene_finish_hook=hook
for area in [a for s in bpy.data.screens for a in s.areas if a.type=='VIEW_3D']:
    area.spaces.active.shading.type='SOLID';area.spaces.active.shading.color_type='MATERIAL';area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.overlay.show_overlays=False
scene.frame_set(1);fs.live=True
bpy.ops.wm.save_as_mainfile(filepath=str(c.LIVE),check_existing=False,compress=True)
c.verify_and_render(scene)
print('COASTAL-LANE-PASS',flush=True)