"""Fail-closed qualification identities. Policy approvals are not observations."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

GENERATED_DEMOS = frozenset({'demos/dam_break.blend', 'demos/crown_splash.blend'})
BINDING_FIELDS = ('source_fingerprint', 'verifier_fingerprint', 'environment_fingerprint')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).rstrip('\n')


def _relevant(name):
    path = Path(name)
    return (path.parts[0] in ('fluidSim_warp_blender51', 'tools', 'tests', 'demos')
            and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.md')
            and not any(part.startswith('.') for part in path.parts)
            and not name.startswith('fluidSim_warp_blender51/docs/'))


def source_files(root):
    root = Path(root)
    names = _git(root, 'ls-files', '-z').split('\0')
    return {name: sha256(root / name) for name in sorted(names) if name and _relevant(name)}


def freeze_source(root):
    dirty = _git(root, 'status', '--porcelain', '--untracked-files=all')
    changed = [line[3:] for line in dirty.splitlines() if len(line) > 3]
    # Strip quotes emitted by git for unusual names; ambiguity must fail closed.
    if any(_relevant(name.strip('"')) for name in changed):
        raise RuntimeError('dirty product or verifier inputs; commit/revert before qualification')
    files = source_files(root)
    return {'git_head': _git(root, 'rev-parse', 'HEAD'), 'files': files,
            'source_fingerprint': fingerprint({k: v for k, v in files.items()
                                               if not k.startswith(('tools/', 'tests/'))
                                               and k not in GENERATED_DEMOS}),
            'verifier_fingerprint': fingerprint({k: v for k, v in files.items()
                                                 if k.startswith(('tools/', 'tests/'))})}


def check_source(root, frozen, before_semantics=None, after_semantics=None):
    if _git(root, 'rev-parse', 'HEAD') != frozen['git_head']:
        raise RuntimeError('HEAD changed during qualification')
    current = source_files(root)
    changed = {name for name in current.keys() | frozen['files'].keys()
               if current.get(name) != frozen['files'].get(name)}
    untracked = _git(root, 'ls-files', '--others', '--exclude-standard', '-z').split('\0')
    if changed - GENERATED_DEMOS or any(name and _relevant(name) for name in untracked):
        raise RuntimeError('product or verifier changed during qualification: ' + ', '.join(sorted(changed)))
    exceptions = {}
    for name in changed:
        before = (before_semantics or {}).get(name)
        after = (after_semantics or {}).get(name)
        if not before or before != after:
            raise RuntimeError(f'{name}: generated demo semantic identity changed or is missing')
        exceptions[name] = {'before': frozen['files'][name], 'after': current[name],
                            'semantic_fingerprint': before}
    return exceptions


def measurement_sources(root, demo):
    """Immutable inputs; saved blend bytes are never inferred from a commit label."""
    root = Path(root)
    paths = list((root / 'fluidSim_warp_blender51').rglob('*'))
    paths += list((root / 'tools/demos').rglob('*')) if demo in ('dam_break', 'crown_splash') else []
    paths += [root / 'demos' / (demo + '.blend')]
    return fingerprint({p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)
                        if p.is_file() and _relevant(p.relative_to(root).as_posix())})


def read_provenance(path, executable_sha256):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        required = ('blender_archive_sha256', 'blender_executable_sha256', 'verifier_fingerprint')
        if any(not isinstance(value.get(key), str) or len(value[key]) != 64
               or any(c not in '0123456789abcdef' for c in value[key]) for key in required):
            raise ValueError('missing SHA-256 identities')
        url = urlparse(value['blender_download_url'])
        if url.scheme != 'https' or url.hostname != 'download.blender.org' or not url.path.endswith(('.zip', '.tar.xz')):
            raise ValueError('expected official Blender distribution URL')
        if not isinstance(value.get('reviewed_by'), str) or not value['reviewed_by'].strip():
            raise ValueError('missing reviewer')
        if value['blender_executable_sha256'] != executable_sha256:
            raise ValueError('approved executable digest differs')
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'invalid reviewed distribution provenance: {exc}') from exc


def validate_binding(report, expected):
    if report.get('gitStatus'):
        raise RuntimeError('dirty measurement provenance cannot qualify a release')
    for name in (*BINDING_FIELDS, 'run_id'):
        if not isinstance(report.get(name), str) or not report[name]:
            raise RuntimeError(f'missing qualification provenance: {name}; remeasure legacy evidence')
    observed = report.get('observed_environment')
    if not isinstance(observed, dict) or fingerprint(observed) != report['environment_fingerprint']:
        raise RuntimeError('observed environment provenance does not match its fingerprint')
    for name in BINDING_FIELDS:
        if expected.get(name) != report[name]:
            raise RuntimeError(f'{name} changed since measurement')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def retained_trial(run):
    try:
        path = Path(run['trial_artifact'])
        if sha256(path) != run['trial_sha256']:
            raise RuntimeError('retained trial report checksum changed')
        raw = json.loads(path.read_text(encoding='utf-8'))
        if raw != {key: value for key, value in run.items()
                   if key not in ('trial_artifact', 'trial_sha256')}:
            raise RuntimeError('retained trial report disagrees with summary')
    except (KeyError, OSError, ValueError) as exc:
        raise RuntimeError('missing or invalid retained trial report') from exc


def validate_environment(observed, policy):
    for field in ('blender_version', 'warp_version', 'blender_sha256', 'python_sha256',
                  'warp_wheel_sha256', 'python_prefix'):
        if observed.get(field) != policy.get(field):
            raise RuntimeError(f'observed {field} differs from approved qualification policy')
    for field in ('gpu_uuid', 'gpu_model', 'driver', 'warp_distribution', 'warp_loaded_path',
                  'blender_build', 'platform', 'architecture', 'python_version', 'python_executable'):
        if not observed.get(field):
            raise RuntimeError(f'missing observed environment identity: {field}')
    if observed['warp_distribution'].split('+')[0] != policy['warp_version']:
        raise RuntimeError('observed warp_distribution differs from approved policy')


def observe_runtime(root, blender, python, env, directory, verifier_fingerprint):
    """Observe the approved executable, its Python and the actual loaded Warp/GPU."""
    root, directory = Path(root), Path(directory)
    executable, interpreter = Path(blender).resolve(), Path(python).resolve()
    approved = env.get('BLENDER_SHA256', '')
    if not approved or sha256(executable) != approved:
        raise RuntimeError('set BLENDER_SHA256 to the reviewed certified executable SHA-256')
    provenance = read_provenance(env.get('BLENDER_PROVENANCE', ''), approved)
    if provenance['verifier_fingerprint'] != verifier_fingerprint:
        raise RuntimeError('reviewed verifier_fingerprint differs; review and commit verifier before qualification')
    if not interpreter.is_relative_to(executable.parent):
        raise RuntimeError('BLENDER_PY must be bundled inside the approved Blender distribution')
    identity = {}
    exec(compile((root / 'fluidSim_warp_blender51/version.py').read_text(), 'version.py', 'exec'), identity)
    wheel = Path(env.get('FLUIDSIM_WARP_WHEEL', '')).resolve()
    if not wheel.is_file():
        raise RuntimeError('FLUIDSIM_WARP_WHEEL must name the retained locked wheel archive')
    lock = json.loads((root / 'tools/release/wheels.lock.json').read_text())
    entry = lock['files'].get(wheel.name)
    if not entry or sha256(wheel) != entry['sha256']:
        raise RuntimeError('retained Warp wheel does not match the certified lock')
    raw = subprocess.check_output([str(interpreter), '-c',
        'import json,sys; print(json.dumps(dict(prefix=sys.prefix,version=sys.version)))'],
        cwd=root, env=env, text=True)
    bundled = json.loads(raw)
    policy = {'blender_version': list(identity['CERTIFIED_BLENDER_VERSION']),
              'warp_version': identity['CERTIFIED_WARP_VERSION'],
              'warp_distribution': identity['CERTIFIED_WARP_DISTRIBUTION'],
              'blender_sha256': approved, 'python_sha256': sha256(interpreter),
              'warp_wheel_sha256': entry['sha256'], 'python_prefix': str(Path(bundled['prefix']).resolve())}
    request = {'wheel': str(wheel), 'python': str(interpreter)}
    request_path, report_path = directory / 'environment-request.json', directory / 'environment.json'
    atomic_json(request_path, request)
    report_path.unlink(missing_ok=True)
    subprocess.run([str(executable), '--python-use-system-env', '--background', '--factory-startup',
        '--python-exit-code', '1', '--python', str(root / 'tools/gate/observe_environment.py'),
        '--', str(request_path), str(report_path)], cwd=root, env=env, check=True)
    observed = json.loads(report_path.read_text(encoding='utf-8'))
    validate_environment(observed, policy)
    if observed['python_version'] != bundled['version']:
        raise RuntimeError('Blender and its bundled Python report different versions')
    return {'observed_environment': observed, 'environment_fingerprint': fingerprint(observed),
            'compatibility_policy': policy, 'distribution_provenance': provenance}
