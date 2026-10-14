"""CC0 scanned coastal dressing with rigid placement and explicit collider proxies."""
import bpy,math,random
import numpy as np
from pathlib import Path
from mathutils import Vector,Matrix

def enhance_scene(scene,root):
    root=Path(root)
    for collection in list(bpy.data.collections):
        if collection.name.startswith('04 | CC0'):
            for obj in list(collection.objects):bpy.data.objects.remove(obj,do_unlink=True)
            bpy.data.collections.remove(collection)
    for obj in list(scene.objects):
        if obj.name.startswith('SET | basalt fracture slab '):bpy.data.objects.remove(obj,do_unlink=True)
    col=bpy.data.collections.new('04 | CC0 coastal scans (physics proxies retained)');scene.collection.children.link(col)
    prototypes={}
    for key in ('coastal_cliff_01','coast_rocks_01'):
        path=root/'assets'/key/(key+'_2k.blend')
        with bpy.data.libraries.load(str(path),link=False) as (a,b):b.objects=[key+'_LOD0']
        obj=b.objects[0];assert obj and obj.type=='MESH';prototypes[key]=obj
        for image in bpy.data.images:
            if image.name.startswith(key):
                filename=Path(bpy.path.abspath(image.filepath)).name;local=root/'assets'/key/'textures'/filename
                if local.exists():image.filepath=str(local);image.reload();image.pack()
        for material in obj.data.materials:
            n=material.node_tree.nodes;l=material.node_tree.links;bs=next(v for v in n if v.type=='BSDF_PRINCIPLED')
            if bs.inputs['Base Color'].is_linked:
                src=bs.inputs['Base Color'].links[0].from_socket;mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;mix.inputs[2].default_value=(.35,.40,.44,1);l.new(src,mix.inputs[1]);l.new(mix.outputs[0],bs.inputs['Base Color'])
            for link in list(bs.inputs['Roughness'].links):l.remove(link)
            bs.inputs['Roughness'].default_value=.47
    def placed(name,proto,center,scale,rotation):
        a=np.array([v.co[:] for v in proto.data.vertices]);local_center=(a.min(0)+a.max(0))/2
        obj=bpy.data.objects.new(name,proto.data);col.objects.link(obj)
        obj.matrix_world=Matrix.Translation(Vector(center))@rotation.to_matrix().to_4x4()@Matrix.Scale(scale,4)@Matrix.Translation(-Vector(local_center))
        obj['provenance']='Poly Haven '+proto.name.split('_LOD')[0]+' CC0. Rigidly positioned scan; no distorted or wrapped photogrammetry.'
        return obj
    from mathutils import Euler
    cliff=prototypes['coastal_cliff_01']
    # Preserve the original scan geometry and UV proportions.
    placed('SCAN | natural eroded left coastline',cliff,(-12,6,4.4),.39,Euler((0,-.17,math.radians(114))))
    stones=prototypes['coast_rocks_01']
    for name,center,scale,angle in [('left foreground',(-8,-13,0),.22,12),('right foreground',(11,-10,.1),.23,-15),('right coast',(15,9,.9),.22,82)]:
        placed('SCAN | '+name,stones,center,scale,Euler((0,0,math.radians(angle))))
    for obj in scene.objects:
        if obj.name.startswith(('SET | basalt headland -1 ','SET | fractured ledge -1 ')) or obj.name in ('SET | near left foreground','SET | near right foreground'):
            obj.hide_render=True;obj.hide_set(True)
    # Angular basalt fractures remain inside the closed hero collider's envelope.
    hero=bpy.data.objects['COLLIDER | tree-topped sea stack'];basalt=hero.data.materials[0];bounds=np.array([hero.matrix_world@Vector(p) for p in hero.bound_box]);lo=bounds.min(0);hi=bounds.max(0);center=(lo+hi)/2;axes=(hi-lo)/2
    for i in range(7):
        rng=random.Random(820+i);a=i*math.tau/7;b=a+math.tau/7*.91
        z0=lo[2]+.12;z1=hi[2]-rng.uniform(.1,1.0);verts=[]
        for z,rf in ((z0,.94),(z1,.55+rng.uniform(0,.18))):
            for angle,radius in [(a,.50),(a,rf),(b,rf),(b,.50)]:verts.append((center[0]+axes[0]*radius*math.cos(angle),center[1]+axes[1]*radius*math.sin(angle),z))
        faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        me=bpy.data.meshes.new('Basalt fracture slab');me.from_pydata(verts,[],faces);me.materials.append(basalt);me.update();ob=bpy.data.objects.new('SET | basalt fracture slab '+str(i),me);col.objects.link(ob)
        bevel=ob.modifiers.new('Chipped stone edges','BEVEL');bevel.width=.075;bevel.segments=1
    scene.camera.location=(-4,-20.5,8.4);scene.camera.rotation_euler=(Vector((1,8.5,1.5))-scene.camera.location).to_track_quat('-Z','Y').to_euler();scene.camera.data.lens=28;scene.camera.data.shift_y=-.045
    scene.view_settings.exposure=.45
    lamp=bpy.data.lights.get('LIGHT | warm rock grazing accent')
    if lamp:lamp.energy=1700
    text=bpy.data.texts.get('ASSET SOURCES AND COLLISION SCOPE') or bpy.data.texts.new('ASSET SOURCES AND COLLISION SCOPE');text.clear();text.write('ASSET SOURCES\nPoly Haven coastal_cliff_01 and coast_rocks_01; 2K maps; CC0.\nhttps://polyhaven.com/a/coastal_cliff_01\nhttps://polyhaven.com/a/coast_rocks_01\nhttps://polyhaven.com/license\n\nScans are coastal set dressing only. The named closed bed, sea stack and reef meshes are the collision proxies; fine scan details are not individually simulated. All referenced images are packed.\nWater, foam and spray are actual Liquid Warp output. Distant offshore water beyond the domain is scenery.\n')
    for proto in prototypes.values():
        if proto.users==0:bpy.data.objects.remove(proto)
    for mesh in list(bpy.data.meshes):
        if mesh.users==0 and mesh.name.startswith(('coastal_cliff','coast_rocks')):bpy.data.meshes.remove(mesh)
    bpy.ops.file.pack_all()