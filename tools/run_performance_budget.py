"""Run D7 through benchmark_versions' rotating fresh-process machinery.

Required reference checkouts and Warp 1.16 site directory are explicit: never
silently benchmark a different release or import a user-installed runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.benchmark_versions import run_paired_trials
from tools.performance_budget import (
    MIN_TRIALS,
    REFERENCES,
    SCENE_REFS,
    comparable_settings,
    evaluate,
)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def reference_roots():
    roots = {}
    for name, demo in (('10255', 'dam_break'), ('110', 'chromatic_vortex')):
        value = os.environ.get(f'FLUIDSIM_PERF_{name}_ROOT')
        if not value:
            raise ValueError(f'FLUIDSIM_PERF_{name}_ROOT must name the pinned reference checkout')
        root = Path(value).resolve(strict=True)
        if git(root, 'rev-parse', 'HEAD') != REFERENCES[demo] or git(root, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError(f'{root} must be clean at {REFERENCES[demo]}')
        roots[name] = root
    return roots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', required=True)
    parser.add_argument('--trials', type=int, default=MIN_TRIALS)
    parser.add_argument('--out-dir', default='artifacts/performance_budget')
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / 'budget.json'
    report.unlink(missing_ok=True)
    if args.trials < MIN_TRIALS or args.trials % 2:
        parser.error('--trials must be even and at least eight')
    try:
        refs = reference_roots()
        reference_warp = os.environ.get('FLUIDSIM_PERF_REFERENCE_WARP_PATH')
        if not reference_warp:
            raise ValueError('FLUIDSIM_PERF_REFERENCE_WARP_PATH must contain Warp 1.16.0')
        # Scenes frozen before the rocky-cove redesign. Both processes read the
        # exact same immutable bridge-saved bytes. No demo builder is invoked.
        scenes = {}
        for demo in REFERENCES:
            path = out_dir / f'{demo}.blend'
            data = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                            f'{SCENE_REFS[demo]}:demos/{demo}.blend'])
            path.write_bytes(data)
            scenes[demo] = path
        runs = []
        completed = {}
        for mode in ('timing', 'stage'):
            def trial(label, root, demo, rnd, mode=mode):
                actual_root = ROOT if label == 'candidate' else refs['110' if demo == 'chromatic_vortex' else '10255']
                out = out_dir / f'{mode}-{demo}-{rnd}-{label}.json'
                out.unlink(missing_ok=True)
                env = os.environ.copy()
                env['PYTHONNOUSERSITE'] = '1'
                if label == 'reference':
                    env['PYTHONPATH'] = reference_warp
                with tempfile.TemporaryDirectory(prefix='fluidsim-budget-', dir=out_dir) as profile:
                    for key, child in (('BLENDER_USER_CONFIG', 'config'), ('BLENDER_USER_SCRIPTS', 'scripts'),
                                       ('BLENDER_USER_DATAFILES', 'datafiles'), ('BLENDER_USER_EXTENSIONS', 'extensions')):
                        env[key] = str(Path(profile) / child)
                    env.pop('BLENDER_USER_RESOURCES', None)
                    proc = subprocess.run([
                        args.blender, '--python-use-system-env', '--background', '--factory-startup',
                        '--python', str(ROOT / 'tools/benchmark_budget_demo.py'), '--',
                        str(actual_root), str(scenes[demo]), str(out), mode,
                        '1.17.0' if label == 'candidate' else '1.16.0'],
                        cwd=actual_root, env=env, capture_output=True, text=True, timeout=900, check=False)
                (out_dir / f'{mode}-{demo}-{rnd}-{label}.log').write_text(proc.stdout + proc.stderr)
                if proc.returncode or 'D7-DEMO-TRIAL-PASS' not in proc.stdout:
                    raise RuntimeError(f'{demo}/{mode}/{rnd}/{label} failed; see {out_dir}')
                row = json.loads(out.read_text())
                row.update(demo=demo, mode=mode, reference_commit=REFERENCES[demo])
                pair = completed.setdefault((mode, demo, rnd), {})
                pair[label] = row
                if len(pair) == 2:
                    a, b = pair['reference'], pair['candidate']
                    if comparable_settings(a['settings']) != comparable_settings(b['settings']):
                        differences = {key: [a['settings'].get(key), b['settings'].get(key)]
                                       for key in a['settings'].keys() | b['settings'].keys()
                                       if comparable_settings(a['settings']).get(key) != comparable_settings(b['settings']).get(key)}
                        raise ValueError(f'{demo}: incomparable settings {differences}; no budget verdict')
                print(f'{mode} {demo} round={rnd} {label}', flush=True)
                return row
            runs.extend(run_paired_trials([('reference', None), ('candidate', ROOT)],
                                          list(REFERENCES), args.trials, trial))
            # Keep completed raw rounds even if a later process fails.
            (out_dir / 'runs.json').write_text(json.dumps(runs, indent=2, allow_nan=False))
        result = evaluate(runs)
        result.update(candidate_head=git(ROOT, 'rev-parse', 'HEAD'),
                      candidate_dirty=git(ROOT, 'status', '--porcelain', '--untracked-files=no'),
                      scene_refs=SCENE_REFS, trials=args.trials,
                      raw_sha256=hashlib.sha256((out_dir / 'runs.json').read_bytes()).hexdigest())
        report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print('PERFORMANCE-BUDGET-' + result['verdict'])
        return 0 if result['verdict'] == 'PASS' else 1
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        report.write_text(json.dumps({'schema': 'fluidsim-d7-budget-v1', 'verdict': 'ERROR', 'error': str(exc)}, indent=2) + '\n')
        print(f'PERFORMANCE-BUDGET-ERROR: {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
