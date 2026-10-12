"""Presentation-only fixes. Must never alter solver sources, seafloor physics or outputs."""
import bpy, math
from mathutils import Vector

def apply(scene):
    props=scene.get('ocean_swell_v2')
    assert bool(props),'Not the certified ocean wave scene'
    before={o.name:(o.fluidsim.role,o.fluidsim.enabled) for o in scene.objects
            if getattr(o,'fluidsim',None) and o.fluidsim.role in ('SOURCE','FORCE','COLLIDER','SINK')}
    far=bpy.data.objects.get('SET | distant horizon ocean OUTSIDE solver domain')
    if far and not far.get('presentation_stitched'):
        # Underlap a small section of scenic offshore ocean with the solver's boundary,
        # always below and outside the visible active cove. It is NOT a replacement.
        for vert in far.data.vertices:
            x,y,z=vert.co
            d=min(1.0,max(0.0,(-y-16.65)/(45-16.65)))
            s=d*d*(3-2*d)
            if y>-19:
                vert.co.y=-16.65
            vert.co.z=-1.28*(1-s)-.24*s +.10*math.sin(.27*float(x)+.22*float(y))*s
        far.data.update()
        far['presentation_stitched']='Scenic horizon plane underlaps offshore simulation edge by at most 1.35m; always well outside inshore hero. Not a simulated surface.'
    # Atmospheric navy-blue ocean; avoid flat mirror-gold water.
    m=bpy.data.materials.get('SET | navy distant ocean only')
    if m and m.use_nodes:
        p=m.node_tree.nodes.get('Principled BSDF')
        if p:
            p.inputs['Base Color'].default_value=(.016,.065,.105,1)
            p.inputs['Metallic'].default_value=.0
            p.inputs['Transmission Weight'].default_value=.0
            p.inputs['Roughness'].default_value=.33
    # Make the sourced Poly Haven stone read as wet DARK BASALT under the dusk sky.
    used={m for ob in scene.objects if getattr(ob.data,'materials',None)
           for m in ob.data.materials if m and m.use_nodes}
    darkened=[]
    for material in used:
        if not any(s in material.name.lower() for s in ('coast_rocks_01','coastal_cliff_01')):
            continue
        nodes=material.node_tree.nodes;links=material.node_tree.links
        bs=next((v for v in nodes if v.type=='BSDF_PRINCIPLED'),None)
        if not bs:continue
        inp=bs.inputs['Base Color']
        if inp.links:
            linked=inp.links[0].from_socket
            mix=next((v for v in nodes if v.name.startswith('COAST | dark basalt pigment')),None)
            if mix is None:
                mix=nodes.new('ShaderNodeMixRGB');mix.name='COAST | dark basalt pigment'
                mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1
                mix.inputs[2].default_value=(.10,.14,.17,1)
                links.new(linked,mix.inputs[1]);links.new(mix.outputs[0],inp)
        else:
            inp.default_value=(.035,.045,.052,1)
        rough=bs.inputs.get('Roughness')
        if rough and not rough.links: rough.default_value=.40
        darkened.append(material.name)
    # Remove repetitive vertically aligned far-offshore columns; replace with
    # irregular rare rock silhouettes behind the active cove.
    for ob in list(scene.objects):
        if ob.name.startswith('SET | distant sea stacks '):
            try:idx=int(ob.name.split()[-1])
            except Exception:idx=0
            if idx % 3 != 0:
                bpy.data.objects.remove(ob,do_unlink=True)
            else:
                ob.scale.z*=.75 if idx in (0,6) else 1.2
                ob.rotation_euler.z+=.35*idx
    world=scene.world
    if world and world.use_nodes:
        bg=world.node_tree.nodes.get('Background')
        if bg:bg.inputs['Strength'].default_value=.27
        for node in world.node_tree.nodes:
            if node.type=='TEX_SKY':
                node.sun_intensity=.09
            if node.type=='VALTORGB':
                ramp=node.color_ramp
                if len(ramp.elements)>=2:
                    ramp.elements[0].color=(.022,.055,.105,1)
                    ramp.elements[-1].color=(.50,.57,.64,1)
    # More legible contrast: translucent turquoise fluid and cold foam accents.
    liquid=scene.fluidsim.surface_material
    if liquid and liquid.use_nodes:
        ps=liquid.node_tree.nodes.get('Principled BSDF')
        if ps:
            ps.inputs['Base Color'].default_value=(.10,.46,.55,1)
            ps.inputs['Transmission Weight'].default_value=.9
            ps.inputs['Roughness'].default_value=.12
    cam=scene.camera
    if cam:
        cam.data.lens=27.7
        cam.data.shift_y=.01
    # Reduce look-dev geometry seams without obscuring wave action.
    if not bpy.data.objects.get('SET | distant marine spray haze (scenery)'):
        data=bpy.data.meshes.new('Sea-spray mist bounds')
        data.from_pydata([(-100,-45,-3),(100,-45,-3),(100,-45,16),(-100,-45,16),(-100,-8,-3),(100,-8,-3),(100,-8,16),(-100,-8,16)],[],[(0,1,2,3),(4,7,6,5),(0,4,5,1),(3,2,6,7),(0,3,7,4),(1,5,6,2)])
        o=bpy.data.objects.new('SET | distant marine spray haze (scenery)',data);scene.collection.objects.link(o)
        vol=bpy.data.materials.new('Volumetric distant surf mist');vol.use_nodes=True
        n=vol.node_tree.nodes;n.clear()
        scatter=n.new('ShaderNodeVolumeScatter')
        scatter.inputs['Color'].default_value=(.40,.50,.65,1)
        scatter.inputs['Density'].default_value=.003
        scatter.inputs['Anisotropy'].default_value=.26
        out=n.new('ShaderNodeOutputMaterial')
        vol.node_tree.links.new(scatter.outputs[0],out.inputs['Volume'])
        data.materials.append(vol)
        o['simulated']=False
    _add_rough_coast(scene)
    _dark_sky(scene)
    _distant_west_coast_in_frame(scene)
    _final_reference_alignment(scene)
    _repair_backdrop_alignment(scene)
    after={o.name:(o.fluidsim.role,o.fluidsim.enabled) for o in scene.objects
           if getattr(o,'fluidsim',None) and o.fluidsim.role in ('SOURCE','FORCE','COLLIDER','SINK')}
    assert before==after,'Physics role unexpectedly changed'
    scene['ocean_presentation_version']='v2 ocean horizon seam and photographic dark-basalt refinements'
    return {'wet_scans_darkened':darkened,'scenic_stitch':bool(far),'physics_roles_unchanged':True,
            'simulated_water_modified':False}


def _add_rough_coast(scene):
    """Geological horizon outside solver, replacing empty flat offshore left edge."""
    from tools.demos import coastal_utilities as cu
    import random
    rng=random.Random(92142)
    group=next(col for col in scene.collection.children if col.name.startswith('02 | SCENERY'))
    mat=bpy.data.materials.get('Basalt | fractured wet coastal rock')
    assert mat
    pine=next((o.data for o in scene.objects if o.type=='MESH' and o.name.startswith('PINE | coastal fir')),None)
    for i in range(27):
        name='SET | forested distant headland %02d'%i
        if name in scene.objects:continue
        y=16-3.2*i
        x=-22-(16-y)*.175+1.3*math.sin(i*.8)
        z=1.6+.6*math.sin(i*.63)
        scales=(rng.uniform(2.7,3.9),rng.uniform(2.5,4.0),rng.uniform(3.3,6.8))
        ob=cu.rock(name,(x,y,z),scales,mat,group,seed=712+i,collision=False,sub=3)
        ob['simulated']=False
        for j in range(rng.randrange(1,4)):
            if not pine:break
            p=bpy.data.objects.new('SET | silhouetted coastal conifer %02d-%d'%(i,j),pine)
            group.objects.link(p)
            p.location=(x+rng.uniform(-1.4,1.4),y+rng.uniform(-1.5,1.5),z+scales[2]*.86)
            h=rng.uniform(1.4,2.7)*(1-.40*i/27)
            p.scale=(h,h,h);p.rotation_euler.z=rng.random()*math.tau
    right=bpy.data.objects.get('SCAN | coast_rocks_01 | 3')
    if right and not right.get('shrunken_for_open_sea'):
        right.scale*=.66;right['shrunken_for_open_sea']=True

def _dark_sky(scene):
    """Native procedural clouded sky with slate-blue ocean reflection."""
    w=scene.world
    if w is None or w.get('deep_blue_pacific_sky'):return
    w.use_nodes=True;n=w.node_tree.nodes;l=w.node_tree.links;n.clear()
    out=n.new('ShaderNodeOutputWorld');bg=n.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value=.92
    tc=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ')
    l.new(tc.outputs['Normal'],sep.inputs[0])
    remap=n.new('ShaderNodeMapRange')
    remap.inputs['From Min'].default_value=-.10
    remap.inputs['From Max'].default_value=.58
    l.new(sep.outputs['Z'],remap.inputs['Value'])
    ramp=n.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=.04
    ramp.color_ramp.elements[0].color=(.23,.15,.17,1)
    ramp.color_ramp.elements[1].position=1
    ramp.color_ramp.elements[1].color=(.009,.028,.062,1)
    e=ramp.color_ramp.elements.new(.34);e.color=(.070,.115,.19,1)
    e=ramp.color_ramp.elements.new(.62);e.color=(.030,.064,.12,1)
    l.new(remap.outputs['Result'],ramp.inputs['Fac'])
    cloud=n.new('ShaderNodeTexNoise')
    cloud.inputs['Scale'].default_value=8
    cloud.inputs['Detail'].default_value=5
    l.new(tc.outputs['Normal'],cloud.inputs['Vector'])
    cr=n.new('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].position=.28
    cr.color_ramp.elements[0].color=(.17,.24,.33,1)
    cr.color_ramp.elements[1].position=.71
    cr.color_ramp.elements[1].color=(1,1,1,1)
    l.new(cloud.outputs['Fac'],cr.inputs['Fac'])
    mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY'
    mix.inputs[0].default_value=.78
    l.new(ramp.outputs[0],mix.inputs[1])
    l.new(cr.outputs[0],mix.inputs[2])
    l.new(mix.outputs[0],bg.inputs['Color'])
    l.new(bg.outputs[0],out.inputs['Surface'])
    w['deep_blue_pacific_sky']=True


def _distant_west_coast_in_frame(scene):
    from tools.demos import coastal_utilities as cu
    import random
    group=next(col for col in scene.collection.children if col.name.startswith('02 | SCENERY'))
    material=bpy.data.materials['Basalt | fractured wet coastal rock']
    mesh=next((o.data for o in scene.objects if o.type=='MESH' and o.name.startswith('PINE | coastal fir')),None)
    rng=random.Random(684012)
    for i in range(20):
        name='SET | distant western forested coastal cliffs %02d'%i
        if name in scene.objects:continue
        y=-21-3.5*i
        x=-12.4-.24*i+1.35*math.sin(i*.62)
        z=1.5+.75*math.sin(i*.46)
        s=(rng.uniform(2.3,3.8),rng.uniform(2.2,3.9),rng.uniform(3.3,6.6))
        rock=cu.rock(name,(x,y,z),s,material,group,seed=8700+i,collision=False,sub=3)
        rock['simulated']=False;rock['outside_solver']='Y less than -18'
        if mesh:
            for j in range(rng.randint(2,4)):
                tree=bpy.data.objects.new('SET | distant Pacific fir %02d.%d'%(i,j),mesh)
                group.objects.link(tree)
                tree.location=(x+rng.uniform(-1.6,1.6),y+rng.uniform(-1.5,1.5),z+s[2]*rng.uniform(.77,.96))
                size=rng.uniform(1.1,2.5)*(1-.48*i/20)
                tree.scale=(size,size,size)
                tree.rotation_euler.z=rng.random()*math.tau
    cam=scene.camera
    if cam:
        cam.location=(1.0,21.0,11.0)
        cam.rotation_euler=(Vector((0,-6,1))-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.lens=28
        cam.data.shift_y=.0
    scene['composition']='wide_ocean_with_visible_western_coast_v3'


def _final_reference_alignment(scene):
    if scene.get('coastal_reference_alignment_v4'):return
    # The actual camera faces offshore to negative Y, so positive world X
    # appears on the LEFT side of frame. Reposition only outside-domain SET objects.
    rock_prefix=('SET | distant western forested coastal cliffs',
                 'SET | rugged west coast horizon')
    tree_prefix=('SET | distant Pacific fir',
                 'SET | silhouetted coastal conifer')
    for o in list(scene.objects):
        if o.name.startswith(rock_prefix) and o.type=='MESH':
            for v in o.data.vertices:v.co.x=-v.co.x
            o.data.update()
            o['outside_solver']='Outside ocean physics; visual coastline left frame'
        elif o.name.startswith(tree_prefix):
            o.location.x=-o.location.x
    # Huge scan cutouts flatten the hero water; retain scans in the file and
    # assign a much smaller right reef silhouette.
    for o in scene.objects:
        if o.name.startswith('SCAN | coastal_cliff_01'):
            o.hide_render=True
            o.hide_set(True)
    cam=scene.camera
    if cam:
        cam.location=(-2,19,10)
        cam.rotation_euler=(Vector((0,-9,4.3))-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.lens=31
        cam.data.shift_y=0
    # A horizon glow, not a flat midday sky.
    world=scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type=='VALTORGB':
                ramp=node.color_ramp
                if len(ramp.elements)>2:
                    ramp.elements[0].color=(.43,.26,.24,1)
                    break
    scene['coastal_reference_alignment_v4']=True


def _repair_backdrop_alignment(scene):
    if scene.get('coastal_backdrop_rigid_flip_repaired'):return
    names=('SET | distant western forested coastal cliffs',
           'SET | rugged west coast horizon')
    repaired=[]
    for o in scene.objects:
        if o.name.startswith(names) and o.type=='MESH':
            # Previous orientation pass mirrored LOCAL mesh X, not the object's
            # WORLD translation. Restore the local shape, then mirror its origin,
            # aligning the already-translated firs and actual geological cliffs.
            for v in o.data.vertices:v.co.x=-v.co.x
            o.location.x=-o.location.x
            o.data.update()
            repaired.append(o.name)
    assert repaired,'No scenic terrain geometry to align'
    scene['coastal_backdrop_rigid_flip_repaired']=True
    scene['backdrop_repair_count']=len(repaired)
