"""Native Windows process tests: executable proxies, no Blender or CUDA."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.gate import windows_trial_process as process

SPEC = importlib.util.spec_from_file_location(
    'tree_test_benchmark', Path(__file__).parents[2] / 'tools/benchmark_versions.py')
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='native Windows process identities')
TRIAL = 'DEMO-PERFORMANCE-TRIAL-PASS'
EXTRACT = 'DEMO-SETTINGS-EXTRACT-PASS'


def _proxies(tmp_path, script):
    second = tmp_path / 'second.cmd'
    second.write_text(f'@"{sys.executable}" "{script}"\n', encoding='utf8')
    first = tmp_path / 'first.cmd'
    first.write_text(
        f'@"{os.environ["COMSPEC"]}" /d /c ""{second}""\n', encoding='utf8')
    return [os.environ['COMSPEC'], '/d', '/c', str(first)]


@pytest.mark.parametrize('marker', [TRIAL, EXTRACT])
@pytest.mark.parametrize('reported', ['child', 'foreign'])
def test_multihop_success_and_identity_sidecar(monkeypatch, tmp_path, marker, reported):
    out, script = tmp_path / 'trial.json', tmp_path / 'trial.py'
    pid = 'os.getpid()' if reported == 'child' else str(os.getpid())
    script.write_text(
        'import json, os, pathlib, time\n'
        'time.sleep(.6)\n'
        f'pathlib.Path({str(out)!r}).write_text(json.dumps({{"processId": {pid}}}))\n'
        f'print({marker!r}, flush=True)\n', encoding='utf8')
    monkeypatch.setattr(bench, '_launch_demo_trial', lambda *args, **kw: process.launch_trial(
        _proxies(tmp_path, script), **kw))
    if reported == 'foreign' and marker == TRIAL:
        with pytest.raises(RuntimeError, match='reported child process identity'):
            bench._run_trial([sys.executable, str(script)], out, expect_marker=marker)
    else:
        bench._run_trial([sys.executable, str(script)], out, expect_marker=marker)
    identity = json.loads(out.with_suffix('.identity.json').read_text())
    assert len(identity['identities']) >= 3
    native = next(x for x in identity['identities'] if x['pid'] == identity['observed_pid'])
    assert native['creation_time'] > 0
    assert os.path.samefile(native['image'], sys.executable)
    assert identity['reported_pid'] == (native['pid'] if reported == 'child' else os.getpid())


@pytest.mark.parametrize('marker', [TRIAL, EXTRACT])
@pytest.mark.parametrize('interruption', ['timeout', 'cancel'])
def test_multihop_cleanup_preserves_foreign_process(monkeypatch, tmp_path, marker, interruption):
    ready, completed = tmp_path / 'ready.json', tmp_path / 'completed'
    script = tmp_path / 'trial.py'
    script.write_text(
        'import json, os, pathlib, time\n'
        'print("owned deep child", flush=True)\n'
        f'ready = pathlib.Path({str(ready)!r})\n'
        'pending = ready.with_suffix(".tmp")\n'
        'pending.write_text(json.dumps(os.getpid()))\n'
        'pending.replace(ready)\n'
        'time.sleep(15)\n'
        f'pathlib.Path({str(completed)!r}).write_text("must not complete")\n', encoding='utf8')
    popen = subprocess.Popen
    foreign = popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    launchers = []
    handles = []
    api = process._kernel()

    def launch(*args, **kwargs):
        child = process.launch_trial(_proxies(tmp_path, script), **kwargs)
        launchers.append(child)
        return child

    def interrupt(child, command, timeout):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pid = process.observe_executable(child, command[0])
            if pid and ready.exists():
                assert pid == json.loads(ready.read_text())
                for record in child._trial_launch_tree.records.values():
                    handle = api.OpenProcess(0x100000, False, record['pid'])
                    assert handle
                    handles.append(handle)
                break
            time.sleep(.01)
        else:
            pytest.fail('Deep native child did not become ready')
        if interruption == 'timeout':
            raise subprocess.TimeoutExpired(command, timeout)
        raise KeyboardInterrupt()

    monkeypatch.setattr(bench, '_launch_demo_trial', launch)
    monkeypatch.setattr(bench, '_communicate_demo_trial', interrupt)
    out = tmp_path / 'trial.json'
    try:
        with pytest.raises(subprocess.TimeoutExpired if interruption == 'timeout' else KeyboardInterrupt):
            bench._run_trial([sys.executable, str(script)], out, expect_marker=marker)
        assert foreign.poll() is None
        assert len(handles) >= 3
        assert all(api.WaitForSingleObject(handle, 0) == 0 for handle in handles)
        assert not completed.exists()
        identity = json.loads(out.with_suffix('.identity.json').read_text())
        native = next(x for x in identity['identities'] if x['pid'] == identity['observed_pid'])
        handle = api.OpenProcess(0x100000, False, native['pid'])
        if handle:
            try:
                assert api.WaitForSingleObject(handle, 0) == 0
            finally:
                api.CloseHandle(handle)
        assert 'owned deep child' in out.with_suffix('.log').read_text()
    finally:
        for handle in handles:
            api.CloseHandle(handle)
        for child in launchers + [foreign]:
            if child.poll() is None:
                if hasattr(child, '_trial_launch_tree'):
                    process.terminate_executable_children(child, sys.executable)
                child.kill()
            child.wait(timeout=5)


def test_retains_verified_child_after_intermediate_launcher_exits(tmp_path):
    ready = tmp_path / 'ready.json'
    script = tmp_path / 'child.py'
    script.write_text(
        'import json, os, pathlib, time\n'
        f'pathlib.Path({str(ready)!r}).write_text(json.dumps(os.getpid()))\n'
        'time.sleep(15)\n', encoding='utf8')
    launcher = process.launch_trial(_proxies(tmp_path, script))
    try:
        deadline = time.monotonic() + 5
        pid = None
        while time.monotonic() < deadline:
            pid = process.observe_executable(launcher, sys.executable)
            if pid and ready.exists():
                break
            time.sleep(.01)
        assert pid
        tree = launcher._trial_launch_tree
        native = tree.records[pid]
        intermediate = tree.records[native['parent_pid']]
        native_handle = tree.api.OpenProcess(0x100000, False, pid)
        assert native_handle
        assert intermediate['pid'] != launcher.pid
        assert tree.api.TerminateProcess(intermediate['handle'], 1)
        assert tree.api.WaitForSingleObject(intermediate['handle'], 5000) == 0
        launcher.wait(timeout=5)
        assert process.observe_executable(launcher, sys.executable) == pid
        try:
            process.terminate_executable_children(launcher, sys.executable)
            assert tree.api.WaitForSingleObject(native_handle, 5000) == 0
        finally:
            tree.api.CloseHandle(native_handle)
    finally:
        process.terminate_executable_children(launcher, sys.executable)
        if launcher.poll() is None:
            launcher.kill()
        launcher.wait(timeout=5)


def test_changed_pinned_identity_fails_closed(monkeypatch):
    tree = process._LaunchTree.__new__(process._LaunchTree)
    tree.closed = False
    tree.api = SimpleNamespace(WaitForSingleObject=lambda *args: 258)
    tree.records = {10: {'pid': 10, 'creation_time': 100, 'handle': 1000}}
    monkeypatch.setattr(process, '_times', lambda *args: (101, 0))
    with pytest.raises(RuntimeError, match='Pinned trial process identity changed'):
        tree.live_records()


def test_foreign_same_image_cannot_gain_job_membership():
    api = process._kernel()
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    job = process._TrialJob()
    tree = process._LaunchTree.__new__(process._LaunchTree)
    tree.api, tree.job = api, job
    try:
        assert tree._open(foreign.pid, os.getpid()) is None
        assert foreign.poll() is None
    finally:
        job.close()
        foreign.kill()
        foreign.wait(timeout=5)


def test_launcher_pid_cannot_replace_authoritative_popen_handle():
    owned = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        reused_pid = SimpleNamespace(pid=foreign.pid, _handle=owned._handle)
        with pytest.raises(RuntimeError, match='launcher handle'):
            process._LaunchTree(reused_pid, None)
        assert owned.poll() is None and foreign.poll() is None
    finally:
        for child in (owned, foreign):
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize('marker', [TRIAL, EXTRACT])
def test_cancels_queued_wrapper_before_it_can_launch_native_child(monkeypatch, tmp_path, marker):
    ready, completed = tmp_path / 'queued', tmp_path / 'must-not-launch'
    second = tmp_path / 'second.cmd'
    second.write_text(
        f'@echo queued>"{ready}"\n@ping 127.0.0.1 -n 15 >nul\n@echo bad>"{completed}"\n',
        encoding='utf8')
    first = tmp_path / 'first.cmd'
    first.write_text(
        f'@"{os.environ["COMSPEC"]}" /d /c ""{second}""\n', encoding='utf8')
    launchers = []

    def launch(*args, **kwargs):
        child = process.launch_trial([os.environ['COMSPEC'], '/d', '/c', str(first)], **kwargs)
        launchers.append(child)
        return child

    def interrupt(child, command, timeout):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert process.observe_executable(child, command[0]) is None
            if ready.exists() and len(child._trial_launch_tree.records) >= 2:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(.01)
        pytest.fail('Owned wrapper did not enter queued state')

    monkeypatch.setattr(bench, '_launch_demo_trial', launch)
    monkeypatch.setattr(bench, '_communicate_demo_trial', interrupt)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            bench._run_trial([sys.executable, 'not-yet-launched.py'], tmp_path / 'trial.json',
                             expect_marker=marker)
        assert launchers[0].poll() is not None
        assert not completed.exists()
        assert all(not os.path.samefile(record['image'], sys.executable)
                   for record in process.process_identity(launchers[0])['identities']
                   if record['pid'] != launchers[0].pid)
    finally:
        for child in launchers:
            process.terminate_executable_children(child, sys.executable)
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()


def test_job_finds_and_cleans_leaf_after_unobserved_intermediate_exit(tmp_path):
    ready = tmp_path / 'leaf.json'
    leaf = tmp_path / 'leaf.py'
    leaf.write_text(
        'import json, os, pathlib, time\n'
        f'ready = pathlib.Path({str(ready)!r})\n'
        'pending = ready.with_suffix(".tmp")\n'
        'pending.write_text(json.dumps(os.getpid()))\n'
        'pending.replace(ready)\n'
        'time.sleep(30)\n', encoding='utf8')
    middle = tmp_path / 'middle.py'
    middle.write_text(
        'import subprocess, sys\n'
        f'subprocess.Popen([sys.executable, {str(leaf)!r}], stdout=subprocess.DEVNULL, '
        'stderr=subprocess.DEVNULL)\n', encoding='utf8')
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    launcher = process.launch_trial([sys.executable, str(middle)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    api, handle = process._kernel(), None
    try:
        # The bootstrap waits for middle; after this, middle is definitively
        # gone. No observation/ancestry snapshot has happened before this point.
        launcher.wait(timeout=5)
        assert launcher.returncode == 0
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        leaf_pid = json.loads(ready.read_text())
        handle = api.OpenProcess(0x100000 | 0x1000 | 1, False, leaf_pid)
        assert handle and api.WaitForSingleObject(handle, 0) == 258
        assert process.observe_executable(launcher, sys.executable) == leaf_pid
        assert all(x['job_membership_verified'] for x in process.process_identity(launcher)['identities'])
        process.terminate_executable_children(launcher, sys.executable)
        assert api.WaitForSingleObject(handle, 5000) == 0
        assert foreign.poll() is None
    finally:
        process.terminate_executable_children(launcher, sys.executable)
        if handle:
            if api.WaitForSingleObject(handle, 0) != 0:
                api.TerminateProcess(handle, 1)
                api.WaitForSingleObject(handle, 5000)
            api.CloseHandle(handle)
        foreign.kill()
        foreign.wait(timeout=5)
        launcher.communicate(timeout=5)


def test_job_assignment_failure_never_executes_command(monkeypatch, tmp_path):
    marker = tmp_path / 'must-not-run'

    def reject(*args):
        raise OSError('assignment intentionally rejected')

    monkeypatch.setattr(process._TrialJob, 'assign', reject)
    with pytest.raises(OSError, match='assignment intentionally rejected'):
        process.launch_trial([sys.executable, '-c',
                              f'from pathlib import Path;Path({str(marker)!r}).touch()'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert not marker.exists()


def test_uncontained_process_is_rejected_without_stopping_it():
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        with pytest.raises(RuntimeError, match='not contained before'):
            process.observe_executable(foreign, sys.executable)
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait(timeout=5)


def test_completed_capture_is_retained_if_containment_finalization_fails(monkeypatch, tmp_path):
    out, script = tmp_path / 'trial.json', tmp_path / 'trial.py'
    script.write_text(
        'import json, os, pathlib, time\n'
        'time.sleep(.4)\n'
        f'pathlib.Path({str(out)!r}).write_text(json.dumps({{"processId": os.getpid()}}))\n'
        f'print({TRIAL!r}, flush=True)\n', encoding='utf8')
    cleanup = process.terminate_executable_children

    def fail_after_safe_cleanup(*args):
        cleanup(*args)
        raise OSError('intentional finalization failure')

    monkeypatch.setattr(process, 'terminate_executable_children', fail_after_safe_cleanup)
    with pytest.raises(OSError, match='intentional finalization failure'):
        bench._run_trial([sys.executable, str(script)], out, expect_marker=TRIAL)
    assert TRIAL in out.with_suffix('.log').read_text()
    identity = json.loads(out.with_suffix('.identity.json').read_text())
    assert identity['observed_pid'] == json.loads(out.read_text())['processId']
    assert identity['contained_before_go']
