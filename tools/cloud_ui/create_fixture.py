import bpy, os
from pathlib import Path
bpy.ops.wm.save_as_mainfile(filepath=str(Path(os.environ['CLOUD_UI_OUTPUT'])/'ui-fixture.blend'))
# Seed only this newly created isolated profile, before canonical before/after snapshot.
bpy.context.preferences.view.show_splash=False
bpy.context.preferences.use_preferences_save=False
bpy.ops.wm.save_userpref()
