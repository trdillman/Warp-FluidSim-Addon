"""Bounded look development shared by the live scene and solver-derived still.
No simulation data, source state, or collision geometry is replaced.
"""
import bpy,math,os,json
from pathlib import Path
from mathutils import Vector,noise
ROOT=Path(r'F:\LiquidWarp-CoastalCascade-20261011')

def apply(scene):
    assert scene.get('coastal_cascade_demo',False) or 'Coastal Cascade' in scene.name
    bed=bpy.data.objects['CASCADE | continuous bed collider'];bed.hide_render=True
    basalt=next(m for m in bpy.data.materials if m.name.startswith('Basalt |'))
    col=bpy.data.collections.get('04 | LOOK - scenic land, no physics')
    if not col:
        col=bpy.data.collections.new('04 | LOOK - scenic land, no physics');scene.collection.children.link(col)
    for obj in list(col.objects):bpy.data.objects.remove(obj,do_unlink=True)
    def smooth(t):t=max(0,min(1,t));return t*t*(3-2*t)
    def level(y):return -.95+1.75*smooth((y+5.2)/.95)+1.7*smooth((y-1.5))+1.8*smooth((y-8))+1.7*smooth((y-14.7)/1.1)
    def center(y):return -2.8-.47*y+.42*math.sin(y*.29)
    def width(y):return 4.65-1.1*math.tanh((y-4)/8)
    def make(name,nx,ny,position):
        vs=[];fs=[]
        for j in range(ny):
            for i in range(nx):vs.append(position(i/(nx-1),j/(ny-1)))
        for j in range(ny-1):
            for i in range(nx-1):a=j*nx+i;fs.extend(((a,a+1,a+nx+1),(a,a+nx+1,a+nx)))
        me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);me.materials.append(basalt);me.update()
        ob=bpy.data.objects.new(name,me);col.objects.link(ob)
        for p in me.polygons:p.use_smooth=True
        ob['provenance']='Outside-water scenic land, not Liquid Warp output or a new collider'
    def left(t,u):
        y=-16+65*u;x=center(y)-width(y)-.40-27*t
        n=noise.fractal(Vector((x*.52,y*.52,.4)),1,2,4)
        z=level(min(y,24))+(1-math.exp(-t*12))*(3+1.5*math.sin(y*.16))+.4*n
        if y>28:z+=.09*(y-28)
        return x,y,z
    make('Scenic | irregular left coastal headland',75,150,left)
    def island(t,u):
        y=8.4+25*u;x=center(y)+width(y)+.4+7*t
        taper=math.sin(math.pi*min(.98,u*.92+.05))**.7
        base=level(y)*(.3+.7*taper)
        z=base+3.2*taper*math.sin(math.pi*t)**.7+.38*noise.fractal(Vector((x*.55,y*.55,3)),1,2,4)
        return x,y,z
    make('Scenic | broken upper coastal island',65,90,island)
    scene.camera.data.shift_y=-.055
    scene.camera.rotation_euler=(Vector((-1.6,10,2.4))-scene.camera.location).to_track_quat('-Z','Y').to_euler()
    fs=scene.fluidsim;fs.whitewater_display_scale=.35
    group=bpy.data.node_groups.get('Coastal | real FOAM particle proximity')
    if group:
        for node in group.nodes:
            if node.bl_idname=='ShaderNodeMapRange':node.inputs['From Max'].default_value=.095
    mat=fs.surface_material;n=mat.node_tree.nodes;l=mat.node_tree.links
    shader=next(node for node in n if node.type=='MIX_SHADER')
    mult=n.get('Coastal foam strength')
    if mult is None:
        link=shader.inputs[0].links[0];src=link.from_socket;mult=n.new('ShaderNodeMath');mult.operation='MULTIPLY';mult.name='Coastal foam strength';l.new(src,mult.inputs[0]);l.new(mult.outputs[0],shader.inputs[0])
    mult.inputs[1].default_value=.48
    p=n.get('Principled BSDF');p.inputs['Base Color'].default_value=(.08,.47,.57,1);p.inputs['Transmission Weight'].default_value=.87
    bpy.data.lights['LIGHT | blue overcast softbox'].energy=11500
    scene.view_settings.exposure=.45
    scene['coastal_lookdev_version']=3
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.shading.type='SOLID';area.spaces.active.overlay.show_overlays=False

if __name__=='__main__':
    s=bpy.context.scene;assert not s.fluidsim.live and s.frame_current==72
    apply(s)
    def render():
        from fluidSim_warp_blender51.blender import playback
        s.render.resolution_percentage=60;s.cycles.samples=32;s.render.filepath=str(ROOT/'delivery/Coastal_Cascade_lookdev.png')
        with playback.suppress_frame_handlers():bpy.ops.render.render(write_still=True)
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':area.spaces.active.shading.type='RENDERED'
        (ROOT/'evidence/lookdev_v3.json').write_text(json.dumps({'status':'rendered','frame':s.frame_current,'path':s.render.filepath,'solver_unchanged':True},indent=2))
        return None
    bpy.app.timers.register(render,first_interval=.3)
    print('LOOK DEVELOPMENT APPLIED; GENUINE FRAME72 RENDER SCHEDULED')