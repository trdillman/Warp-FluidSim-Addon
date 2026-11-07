"""Reproduce bounded qualification commands with private Blender resources."""
from pathlib import Path
import os
import subprocess
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BLENDER = Path('C:/Users/Tyler/Downloads/blender-5.2.0-windows-x64/blender-5.2.0-windows-x64/blender.exe')
PYTHON = BLENDER.parent / '5.2/python/bin/python.exe'
WARP118 = Path('C:/Users/Tyler/AppData/Roaming/Blender Foundation/Blender/5.2/extensions/.user/user_default/fluidsim_warp/.local')
env = os.environ.copy()
env['BLENDER_EXE'] = str(BLENDER)
env['BLENDER_PY'] = str(PYTHON)
env['BLENDER_USER_EXTENSIONS'] = str(ROOT / 'artifacts/pr47-check-extensions')
env['BLENDER_USER_CONFIG'] = str(ROOT / 'artifacts/pr47-check-config')
env['PYTHONPATH'] = str(WARP118)
env['PYTHONNOUSERSITE'] = '1'
results = []
commands = [
 ('production-gate', [str(PYTHON), 'tools/run_production_gate.py']),
 ('whitewater118', [str(BLENDER), '--background', '--factory-startup', '--offline-mode', '--python-use-system-env', '--python-exit-code', '1', '--python', 'tests/acceptance/blender_whitewater_truth_acceptance.py']),
 ('package', [str(PYTHON), 'tools/release/build_extension.py', '--blender', str(BLENDER), '--output', str(ROOT / 'artifacts/pr47-packages'), '--json-report', str(OUT / 'package.json')]),
]
for name, command in commands:
    task_env = env.copy()
    if name == 'package':
        task_env['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run(command, cwd=ROOT, env=task_env, capture_output=True, text=True)
    (OUT / (name + '.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
    results.append({'name': name, 'command': command, 'returncode': result.returncode})
    print(name, result.returncode, (result.stdout + result.stderr)[-1000:], flush=True)
    (OUT / 'commands.json').write_text(json.dumps(results, indent=2))
