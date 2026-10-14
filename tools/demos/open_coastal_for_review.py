"""Show the delivered live demo in a private, source-qualified Blender process."""
import bpy,os,json,sys
from pathlib import Path
ROOT=Path(os.environ['COASTAL_ROOT'])
site=Path(r'C:\Users\Tyler\AppData\Roaming\Blender Foundation\Blender\5.2\extensions\.local\lib\python3.13\site-packages')
if str(site) not in sys.path:sys.path.insert(0,str(site))
bpy.context.preferences.use_preferences_save=False
s=bpy.context.scene
assert Path(bpy.data.filepath).resolve()==(ROOT/'delivery/Coastal_Cascade_LIVE.blend').resolve()
assert s.frame_current==s.frame_start==1 and s.fluidsim.live
for scr in bpy.data.screens:
    for area in scr.areas:
        if area.type=='VIEW_3D':
            space=area.spaces.active;space.region_3d.view_perspective='CAMERA';space.region_3d.view_camera_zoom=12;space.overlay.show_overlays=False
            space.shading.type='MATERIAL';space.shading.use_scene_lights=True;space.shading.use_scene_world=True
        elif area.type=='PROPERTIES':area.spaces.active.context='SCENE'
receipt={'ok':True,'purpose':'owner review; independent qualification is in evidence/cold_reopen_verification.json','file':bpy.data.filepath,'pid':os.getpid(),'frame':s.frame_current,'live':s.fluidsim.live}
Path(os.environ['FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT']).write_text(json.dumps(receipt),encoding='utf-8')
print('COASTAL LIVE OWNER REVIEW READY',flush=True)