"""Read-only final scene packaging and visual-evidence inventory.
Execute in the final owner-review process after images have been packed/saved.
"""
import bpy,json,hashlib,os
from pathlib import Path
ROOT=Path(r'F:\LiquidWarp-CoastalCascade-20261011')
s=bpy.context.scene
assert Path(bpy.data.filepath).is_relative_to(ROOT)
images=[]
for im in bpy.data.images:
    if im.users==0 or im.source in ('VIEWER','GENERATED'):continue
    images.append({'name':im.name,'source':im.source,'size':list(im.size),'packed':bool(im.packed_file or len(im.packed_files)),'path':im.filepath,'exists':Path(bpy.path.abspath(im.filepath)).is_file() if im.filepath else False})
external=[im for im in images if not im['packed']]
prefs=bpy.context.preferences.addons['cycles'].preferences
report={'schema':'coastal-delivery-packaging-v1','file':bpy.data.filepath,'file_sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),'frame':s.frame_current,'live':s.fluidsim.live,'render_settings':{'engine':s.render.engine,'device':s.cycles.device,'samples':s.cycles.samples,'size':[s.render.resolution_x,s.render.resolution_y],'percentage':s.render.resolution_percentage,'enabled_devices':[{'name':d.name,'type':d.type} for d in prefs.devices if d.use]},'images':images,'external_images':external,'linked_libraries':[x.filepath for x in bpy.data.libraries],'scene_counts':{'objects':len(s.objects),'meshes':sum(o.type=='MESH' for o in s.objects),'colliders':sum(o.fluidsim.role=='COLLIDER' and o.fluidsim.enabled for o in s.objects),'owned_controls':sum(o.name.startswith(('DOMAIN |','SOURCE |','SINK |')) for o in s.objects)},'missing_images':[im['name'] for im in external if not im['exists']]}
path=ROOT/'evidence'/'delivery_packaging.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'file':report['file'],'frame':s.frame_current,'images':len(images),'external_images':len(external),'missing_images':report['missing_images'],'linked_libraries':report['linked_libraries'],'path':str(path)}))