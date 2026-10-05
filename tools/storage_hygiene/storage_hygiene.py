"""Bounded, read-only storage warnings; no deletion or process inspection."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time


GUIDANCE = ('Owner must check current references, active leases and retention evidence; '
            'clean only known owned disposable files through a verified recoverable '
            'route. Permanent deletion needs exact action-time approval. Preserve '
            'source, assets, packages, baselines and failure proofs.')


def is_reparse(st):
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, 'st_file_attributes', 0) & 0x400)


def normalized(path):
    return os.path.normcase(os.path.abspath(path))  # deliberately not resolve()


def within(path, parent):
    try:
        return os.path.commonpath([normalized(path), normalized(parent)]) == normalized(parent)
    except ValueError:
        return False


def validate(config):
    if not isinstance(config, dict) or config.get('version') != 1 or not isinstance(config.get('roots'), list):
        raise ValueError('Expected version=1 and roots list')
    denied = config.get('never_probe', [])
    if not isinstance(denied, list) or any(not isinstance(p, str) or not os.path.isabs(p) for p in denied):
        raise ValueError('never_probe must be a list of absolute paths')
    for field in ('free_warn_bytes', 'max_entries', 'max_seconds'):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid ' + field)
    paths = []
    records = config.get('records', [])
    if not isinstance(records, list):
        raise ValueError('records must be a list')
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str) or not os.path.isabs(record['path']):
            raise ValueError('Records require absolute paths')
    for root in config['roots']:
        if not isinstance(root, dict) or not isinstance(root.get('path'), str) or not os.path.isabs(root['path']):
            raise ValueError('Roots require absolute paths')
        for field in ('budget_bytes', 'stale_days'):
            value = root.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid root ' + field)
        protected = root.get('protected', [])
        if not isinstance(protected, list) or any(not isinstance(p, str) or not os.path.isabs(p) for p in protected):
            raise ValueError('Protected paths require absolute paths')
        if any(within(root['path'], p) or within(p, root['path']) for p in paths):
            raise ValueError('Root overlap would double count bytes')
        paths.append(root['path'])


def check(config, *, now=None, free_bytes=None):
    validate(config)
    now = time.time() if now is None else now
    result = {'version': 1, 'checked_at': now, 'roots': [], 'warnings': [],
              'measurement': 'logical and unique-file lengths, not allocated/reclaimable bytes'}

    def warn(code, root=None, detail='', **fields):
        result['warnings'].append(dict(code=code, severity='warning',
            path=root['path'] if root else config.get('drive', Path.cwd().anchor),
            owner=root.get('owner', 'unregistered') if root else 'all project owners',
            detail=detail, guidance=GUIDANCE, **fields))

    try:
        free_bytes = shutil.disk_usage(config.get('drive', Path.cwd().anchor)).free if free_bytes is None else free_bytes
        result['free_bytes'] = free_bytes
        if free_bytes < config['free_warn_bytes']:
            warn('low_free_space', detail='Drive free space below configured warning threshold', bytes=free_bytes)
    except OSError as exc:
        result['free_bytes'] = None
        warn('free_space_unavailable', detail=str(exc))

    for root in config['roots']:
        registered = all(root.get(f) for f in ('project', 'owner', 'retention_reason')) and root.get('state') in ('active', 'completed', 'retained')
        if not registered:
            warn('registration', root, 'Owner/project/state/retention record incomplete; no cleanup eligibility')
        measure = dict(path=root['path'], owner=root.get('owner'), logical_bytes=None,
                       unique_file_bytes=None, files=0, complete=True, newest_mtime=None)
        result['roots'].append(measure)
        deadline = time.monotonic() + config['max_seconds']
        count = 0
        identities = set()
        protected = root.get('protected', []) + config.get('never_probe', [])
        failed_codes = set()
        unknown_children = []
        known_children = {normalized(record['path']) for record in config.get('records', [])
                          if all(record.get(f) for f in ('project', 'owner', 'retention_reason'))
                          and record.get('state') in ('active', 'completed', 'retained')}

        def incomplete(code, path, detail=''):
            measure['complete'] = False
            if code not in failed_codes:
                warn(code, root, str(path) + (': ' + detail if detail else ''))
                failed_codes.add(code)

        # Check each ancestor without resolving links or probing protected paths.
        ancestors = list(reversed(Path(root['path']).parents)) + [Path(root['path'])]
        allowed = True
        for ancestor in ancestors:
            if any(within(ancestor, p) for p in protected):
                incomplete('protected_unmeasured', ancestor)
                allowed = False
                break
            try:
                st = os.lstat(ancestor)
                if is_reparse(st):
                    incomplete('reparse_skipped', ancestor)
                    allowed = False
                    break
            except FileNotFoundError:
                incomplete('missing', ancestor)
                allowed = False
                break
            except OSError as exc:
                incomplete('inaccessible', ancestor, str(exc))
                allowed = False
                break
        if not allowed:
            continue
        if not stat.S_ISDIR(st.st_mode):
            incomplete('not_directory', root['path'])
            continue
        stack = [root['path']]
        while stack:
            directory = stack.pop()
            if count >= config['max_entries'] or time.monotonic() >= deadline:
                incomplete('scan_limit', directory, 'Bound reached; measurements are lower bounds')
                break
            try:
                if is_reparse(os.lstat(directory)):
                    incomplete('reparse_skipped', directory)
                    continue
                # scandir is streaming; never build a full list of a large directory.
                with os.scandir(directory) as entries:
                    # Successful enumeration establishes at least a zero-byte lower bound.
                    if measure['logical_bytes'] is None:
                        measure['logical_bytes'] = measure['unique_file_bytes'] = 0
                    for entry in entries:
                        if count >= config['max_entries'] or time.monotonic() >= deadline:
                            incomplete('scan_limit', directory, 'Bound reached; measurements are lower bounds')
                            stack.clear()
                            break
                        count += 1
                        if any(within(entry.path, p) for p in protected):
                            incomplete('protected_unmeasured', entry.path)
                            continue
                        try:
                            est = entry.stat(follow_symlinks=False)
                            if is_reparse(est):
                                incomplete('reparse_skipped', entry.path)
                            elif stat.S_ISDIR(est.st_mode):
                                if root.get('require_child_records') and normalized(directory) == normalized(root['path']) and normalized(entry.path) not in known_children:
                                    unknown_children.append(entry.path)
                                stack.append(entry.path)
                            elif stat.S_ISREG(est.st_mode):
                                # os.stat supplies real NTFS file IDs; DirEntry can cache zero IDs.
                                est = os.stat(entry.path, follow_symlinks=False)
                                if is_reparse(est):
                                    incomplete('reparse_skipped', entry.path)
                                    continue
                                measure['logical_bytes'] += est.st_size
                                identity = (est.st_dev, est.st_ino) if est.st_ino else normalized(entry.path)
                                if identity not in identities:
                                    identities.add(identity)
                                    measure['unique_file_bytes'] += est.st_size
                                measure['files'] += 1
                                measure['newest_mtime'] = max(measure['newest_mtime'] or est.st_mtime, est.st_mtime)
                        except OSError as exc:
                            incomplete('inaccessible', entry.path, str(exc))
            except OSError as exc:
                incomplete('inaccessible', directory, str(exc))
        size = measure['logical_bytes']
        newest = measure['newest_mtime']
        age = max(0, (now - newest) / 86400) if newest is not None else None
        measure['age_days'] = age
        if unknown_children:
            warn('unregistered_children', root, 'Top-level folders lack ownership/retention records; names do not establish ownership', count=len(unknown_children), sample=unknown_children[:10])
        if size is not None and size > root['budget_bytes']:
            warn('budget', root, 'Measured bytes exceed folder budget' + (' (lower bound)' if not measure['complete'] else ''), bytes=size, age_days=age)
        if registered and root.get('disposable') is True and root.get('state') == 'completed' and measure['complete'] and age is not None and age > root['stale_days']:
            warn('stale', root, 'Completed owned disposable output exceeds retention age; eligibility still requires owner review', bytes=size, age_days=age)
    return result


def emit(config, phase, project):
    try:
        result = check(config)
        print(f'STORAGE-HYGIENE phase={phase} project={project} roots={len(result["roots"])} warnings={len(result["warnings"])}', file=sys.stderr, flush=True)
        for warning in result['warnings']:
            print('STORAGE-HYGIENE-WARN ' + json.dumps(warning, ensure_ascii=False), file=sys.stderr, flush=True)
        return result
    except Exception as exc:
        print(f'STORAGE-HYGIENE-DIAGNOSTIC phase={phase} {type(exc).__name__}: {exc}; project test result remains independent', file=sys.stderr, flush=True)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('policy.json')))
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('check')
    run = commands.add_parser('run')
    run.add_argument('--project', required=True)
    run.add_argument('--self-test', action='store_true', help='run storage regression tests before the production command')
    run.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        if args.action != 'run':
            print(f'STORAGE-HYGIENE-DIAGNOSTIC {exc}', file=sys.stderr)
            return 2
        config = {}  # emit reports the missing policy; never silently disable diagnostics
        print(f'STORAGE-HYGIENE-DIAGNOSTIC policy unavailable: {exc}', file=sys.stderr, flush=True)
    if args.action == 'check':
        try:
            print(json.dumps(check(config), indent=2))
            return 0
        except (OSError, ValueError) as exc:
            print(f'STORAGE-HYGIENE-DIAGNOSTIC {exc}', file=sys.stderr)
            return 2
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('run requires a production command after --')
    if args.self_test:
        rc = subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).with_name('tests'))], cwd=Path(__file__).parent)
        if rc:
            print('STORAGE-HYGIENE-TEST-FAIL: production command was not started', file=sys.stderr, flush=True)
            return rc
    emit(config, 'pre-production', args.project)
    try:
        return subprocess.call(command)
    finally:
        emit(config, 'end-task', args.project)


if __name__ == '__main__':
    raise SystemExit(main())
