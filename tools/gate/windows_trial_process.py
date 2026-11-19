"""Observe and retain OS identities throughout a Windows trial's launch tree."""

import ctypes
import json
import os
import subprocess
import sys
import time
import weakref
from ctypes import wintypes


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD), ("usage", wintypes.DWORD),
        ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
        ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
        ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
        ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260),
    ]


def _kernel():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
    api.OpenProcess.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    api.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    )
    api.CreateToolhelp32Snapshot.argtypes = wintypes.DWORD, wintypes.DWORD
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Process32FirstW.argtypes = wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)
    api.Process32NextW.argtypes = api.Process32FirstW.argtypes
    api.TerminateProcess.argtypes = wintypes.HANDLE, wintypes.UINT
    api.TerminateProcess.restype = wintypes.BOOL
    api.GetProcessTimes.argtypes = (
        wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4),
    )
    api.GetProcessTimes.restype = wintypes.BOOL
    api.GetProcessId.argtypes = (wintypes.HANDLE,)
    api.GetProcessId.restype = wintypes.DWORD
    api.WaitForSingleObject.argtypes = wintypes.HANDLE, wintypes.DWORD
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.IsProcessInJob.argtypes = wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)
    api.IsProcessInJob.restype = wintypes.BOOL
    return api


def _children(api, parent):
    return [pid for pid, ppid in _snapshot(api).items() if ppid == parent]


def _snapshot(api):
    snapshot = api.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ProcessEntry(size=ctypes.sizeof(_ProcessEntry))
        found = api.Process32FirstW(snapshot, ctypes.byref(entry))
        parents = {}
        while found:
            parents[entry.pid] = entry.parent
            found = api.Process32NextW(snapshot, ctypes.byref(entry))
        return parents
    finally:
        api.CloseHandle(snapshot)


def _times(api, handle):
    values = [wintypes.FILETIME() for _ in range(4)]
    if not api.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    return tuple(value.dwLowDateTime | (value.dwHighDateTime << 32) for value in values[:2])


def _image(api, handle):
    name = ctypes.create_unicode_buffer(32768)
    size = wintypes.DWORD(len(name))
    if not api.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return name.value


def _parent(handle):
    # Query the pinned process, not a possibly stale Toolhelp PID entry.
    class BasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_void_p) for name in
                    ('reserved', 'peb', 'reserved2', 'reserved3', 'pid', 'parent')]

    query = ctypes.WinDLL('ntdll').NtQueryInformationProcess
    query.argtypes = (wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p,
                      wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
    query.restype = wintypes.LONG
    info, size = BasicInfo(), wintypes.ULONG()
    status = query(handle, 0, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(size))
    if status < 0:
        raise OSError(f'Cannot verify trial process parent (NTSTATUS {status:#x})')
    return info.parent


def _same_image(actual, expected):
    try:
        return os.path.samefile(actual, expected)
    except OSError:
        return False


class _TrialJob:
    """A pre-execution, non-breakaway, kill-on-close Windows process boundary."""

    def __init__(self):
        class Basic(ctypes.Structure):
            _fields_ = [('ProcessTime', ctypes.c_longlong), ('JobTime', ctypes.c_longlong),
                        ('Flags', wintypes.DWORD), ('MinWorking', ctypes.c_size_t),
                        ('MaxWorking', ctypes.c_size_t), ('ProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('Priority', wintypes.DWORD),
                        ('Schedule', wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(f'c{i}', ctypes.c_ulonglong) for i in range(6)]

        class Extended(ctypes.Structure):
            _fields_ = [('Basic', Basic), ('Io', IO)] + [
                (f'm{i}', ctypes.c_size_t) for i in range(4)]

        class Accounting(ctypes.Structure):
            _fields_ = [(f't{i}', ctypes.c_longlong) for i in range(4)] + [
                ('Faults', wintypes.DWORD), ('Total', wintypes.DWORD),
                ('Active', wintypes.DWORD), ('Terminated', wintypes.DWORD)]

        self.accounting = Accounting
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                         wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                           wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.Basic.Flags = 0x2000  # KILL_ON_JOB_CLOSE; no BREAKAWAY_OK.
        if not self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.k.CloseHandle(self.handle)
            self.handle = None
            raise error

    def assign(self, child):
        if not self.k.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def pids(self):
        capacity = 16
        for _ in range(8):
            class Members(ctypes.Structure):
                _fields_ = [('assigned', wintypes.DWORD), ('count', wintypes.DWORD),
                            ('pids', ctypes.c_size_t * capacity)]

            members = Members()
            if self.k.QueryInformationJobObject(
                    self.handle, 3, ctypes.byref(members), ctypes.sizeof(members), None):
                if members.count > capacity:
                    raise RuntimeError('Invalid trial Job membership count')
                return list(members.pids[:members.count])
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity = max(capacity * 2, members.assigned)
        raise RuntimeError('Trial Job membership did not stabilize')

    def terminate(self):
        if not self.handle:
            return
        try:
            if not self.k.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + 5
            while True:
                counts = self.accounting()
                if not self.k.QueryInformationJobObject(
                        self.handle, 1, ctypes.byref(counts), ctypes.sizeof(counts), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not counts.Active:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Owned trial Job did not exit within five seconds')
                time.sleep(.01)
        finally:
            self.k.CloseHandle(self.handle)
            self.handle = None

    close = terminate


def launch_trial(command, **kwargs):
    """Keep Popen capture/wait semantics; assign the blocked bootstrap before GO."""
    if 'stdin' in kwargs:
        raise ValueError('Trial bootstrap owns stdin')
    if kwargs.get('creationflags', 0) & 0x01000000:
        raise ValueError('Trial process must not break away from containment')
    job, child = _TrialJob(), None
    try:
        child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), '--trial-child', json.dumps(list(command))],
            stdin=subprocess.PIPE, **kwargs)
        job.assign(child)
        tree = _LaunchTree(child, job)
        child._trial_launch_tree = tree
        weakref.finalize(child, tree.close)
        child.stdin.write('GO\n' if child.text_mode else b'GO\n')
        child.stdin.flush()
        return child
    except BaseException:
        try:
            job.close()
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()
        raise


class _LaunchTree:
    """Pin native identities in the Job established before command execution.

    Membership survives intermediates that exit before the first observation.
    Parent PIDs remain diagnostic ancestry, while OS Job membership authorizes
    ownership. Held handles and creation times prevent recycled-PID acceptance.
    No identity is supplied by the child report.
    """

    def __init__(self, launcher, job):
        self.api = _kernel()
        self.records = {}
        self.closed = False
        self.observed_pid = None
        self.root_pid = launcher.pid
        self.job = job
        # Popen's original handle is authoritative even if its PID was recycled.
        root_handle = launcher._handle
        if self.api.GetProcessId(root_handle) != launcher.pid:
            raise RuntimeError("trial launcher handle does not identify its PID")
        created, _ = _times(self.api, root_handle)
        root = self._open(launcher.pid, None)
        if root is None or root['creation_time'] != created:
            if root is not None:
                self.api.CloseHandle(root['handle'])
            raise RuntimeError("trial launcher process identity changed")
        self.records[launcher.pid] = root

    def _open(self, pid, parent):
        handle = self.api.OpenProcess(0x100000 | 0x1000 | 1, False, pid)
        if not handle:
            return None
        try:
            created, exited = _times(self.api, handle)
            actual_parent = _parent(handle)
            member = wintypes.BOOL()
            if not self.api.IsProcessInJob(handle, self.job.handle, ctypes.byref(member)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not member.value or (parent is not None and actual_parent != parent):
                self.api.CloseHandle(handle)
                return None
            record = {'pid': pid, 'parent_pid': actual_parent, 'creation_time': created,
                      'exit_time': exited, 'image': _image(self.api, handle), 'handle': handle,
                      'job_membership_verified': True}
        except OSError:
            self.api.CloseHandle(handle)
            return None
        return record

    def refresh(self):
        if self.closed:
            return
        # Membership was established before GO and inherited by every child.
        # It survives missing/exited intermediates, unlike Toolhelp ancestry.
        for pid in self.job.pids():
            if pid not in self.records:
                child = self._open(pid, None)
                if child is not None:
                    self.records[pid] = child

    def live_records(self):
        if self.closed:
            return []
        live = []
        for record in self.records.values():
            state = self.api.WaitForSingleObject(record['handle'], 0)
            if state == 0:
                continue
            if state != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            if _times(self.api, record['handle'])[0] != record['creation_time']:
                raise RuntimeError('Pinned trial process identity changed')
            live.append(record)
        return live

    def matches(self, expected):
        return [record for record in self.live_records()
                if record['pid'] != self.root_pid and _same_image(record['image'], expected)]

    def receipt(self):
        return {'launcher_pid': self.root_pid, 'observed_pid': self.observed_pid,
                'contained_before_go': True,
                'cleanup_status': getattr(self, 'cleanup_status', None),
                'cleanup_error': getattr(self, 'cleanup_error', None),
                'cleanup_expected_image': getattr(self, 'cleanup_expected_image', None), 'identities': [
            {key: value for key, value in record.items() if key != 'handle'}
            for record in self.records.values()]}

    def close(self):
        if not self.closed:
            self.closed = True
            for record in self.records.values():
                self.api.CloseHandle(record['handle'])
            self.job.close()


def _tree(launcher):
    tree = getattr(launcher, '_trial_launch_tree', None)
    if tree is None:
        raise RuntimeError('Trial was not contained before command execution')
    return tree


def process_identity(launcher):
    """Return serializable OS evidence already captured, never child assertions."""
    tree = getattr(launcher, '_trial_launch_tree', None)
    return tree.receipt() if tree is not None else {'launcher_pid': launcher.pid, 'identities': []}


def observe_executable(launcher, expected):
    """Return the unique live exact-image process in the verified launch tree."""
    tree = _tree(launcher)
    tree.refresh()
    matches = tree.matches(expected)

    if len(matches) > 1:
        raise RuntimeError("multiple exact trial executables descended from the launcher")
    if matches:
        tree.observed_pid = matches[0]['pid']
        return tree.observed_pid
    return None


def terminate_executable_children(launcher, expected):
    """Stop the pinned trial tree, including wrappers that could launch later.

    Keep the existing API, but include the launcher and intermediate executables:
    killing just the Blender leaf would leave a queued gate wrapper able to
    launch after cancellation. Atomic Job termination covers even descendants
    not yet observed, followed by a bounded wait for zero active members.
    An unrelated same-image process is never selected.
    """
    tree = _tree(launcher)
    tree.cleanup_expected_image = os.fspath(expected)
    deadline = time.monotonic() + 5
    try:
        if not tree.closed:
            tree.refresh()
            tree.job.terminate()
            for record in tree.records.values():
                remaining = max(0, int((deadline - time.monotonic()) * 1000))
                if tree.api.WaitForSingleObject(record['handle'], remaining) != 0:
                    raise RuntimeError('Pinned trial process did not exit within five seconds')
            tree.cleanup_status = 'PASS'
    except BaseException as error:
        tree.cleanup_status, tree.cleanup_error = 'FAIL', str(error)
        raise
    finally:
        tree.close()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--trial-child':
        raise SystemExit('Internal trial bootstrap arguments required')
    if sys.stdin.buffer.readline() not in (b'GO\n', b'GO\r\n'):
        raise SystemExit('Trial command was not admitted')
    raise SystemExit(subprocess.call(json.loads(sys.argv[2]), stdin=subprocess.DEVNULL))
