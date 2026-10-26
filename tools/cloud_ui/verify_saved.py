"""Fresh-process saved RNA check; never starts solver or probes GPU context."""
import bpy, json, os, sys
from pathlib import Path
sys.path.insert(0,os.environ['CLOUD_UI_SOURCE'])
from tools.repo_addon_loader import enable_repo_addon
enable_repo_addon(os.environ['CLOUD_UI_SOURCE'], default_set=False, persistent=False)
assert bpy.context.scene.fluidsim.ui_mode=='ADVANCED'
assert bpy.context.scene.fluidsim.live is False
out=Path(os.environ['CLOUD_UI_OUTPUT'])
(out/'presentation-reopen.json').write_text(json.dumps({'ok':True,'ui_mode':bpy.context.scene.fluidsim.ui_mode,'live':bpy.context.scene.fluidsim.live,'background':bpy.app.background,'scope':'real RNA persisted across fresh process; no simulation'},indent=2))
print('CLOUD-UI-PRESENTATION-REOPEN-PASS')
