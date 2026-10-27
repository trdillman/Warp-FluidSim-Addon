"""Readiness + real RNA persistence probe, NOT a full GUI acceptance result.
Screenshots and manual interactions must separately establish visible UI behavior.
No fake runtime modules, solver substitution or production preferences writes.
"""
import bpy, gpu, json, os, traceback
from pathlib import Path
OUT=Path(os.environ['CLOUD_UI_OUTPUT'])
def probe():
    result={'ok':False,'scope':'foreground readiness and presentation-only RNA save; visual inspection and reopen separate','errors':[]}
    try:
        assert not bpy.app.background, 'Must run in a foreground Blender window'
        assert bpy.app.version==(5,2,2), bpy.app.version
        assert bpy.app.build_hash.decode()=='d13f752e3b9c'
        assert len(bpy.context.window_manager.windows)>0, 'No foreground window'
        import fluidSim_warp_blender51 as addon
        from fluidSim_warp_blender51 import panels
        ready,message=panels.runtime_ui_state()
        assert not ready and panels.playback_only(), 'Expected genuine unavailable CUDA runtime'
        assert message==panels.PLAYBACK_ONLY_TEXT
        # This is a presentation-only scene setting, not a simulation request.
        bpy.context.scene.fluidsim.ui_mode='ADVANCED'
        assert bpy.context.scene.fluidsim.live is False
        filepath=str(OUT/'ui-fixture.blend')
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type=='VIEW_3D':
                    area.spaces.active.show_region_ui=True
                    area.tag_redraw()
        result.update(version=bpy.app.version_string,build_hash=bpy.app.build_hash.decode(),renderer=gpu.platform.renderer_get(),backend=gpu.platform.backend_type_get(),addon_file=addon.__file__,runtime_ui_state=[ready,message],playback_only=True,window_count=len(bpy.context.window_manager.windows),presentation_setting_saved=True,live_simulation=False,visual_panel_validation='requires separately inspected screenshot',ok=True)
    except Exception:
        result['errors'].append(traceback.format_exc())
    (OUT/'foreground-readiness.json').write_text(json.dumps(result,indent=2))
    Path(os.environ['FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT']).write_text(json.dumps(result))
    return None
bpy.app.timers.register(probe,first_interval=5)
