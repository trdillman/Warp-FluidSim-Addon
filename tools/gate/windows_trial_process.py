"""Observe a Windows trial's native executable and direct launch ancestry."""

import ctypes
import os
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
    return api


def _matches_image(api, pid, expected, parent=None):
    handle = api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        name = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(name))
        if not api.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            return False
        if not os.path.samefile(name.value, expected):
            return False
        # While the handle is held Windows cannot recycle this PID. Recheck the
        # parent in a fresh snapshot in case the enumeration was already stale.
        return parent is None or pid in _children(api, parent)
    except OSError:
        return False
    finally:
        api.CloseHandle(handle)


def _children(api, parent):
    snapshot = api.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ProcessEntry(size=ctypes.sizeof(_ProcessEntry))
        found = api.Process32FirstW(snapshot, ctypes.byref(entry))
        children = []
        while found:
            if entry.parent == parent:
                children.append(entry.pid)
            found = api.Process32NextW(snapshot, ctypes.byref(entry))
        return children
    finally:
        api.CloseHandle(snapshot)


def observe_executable(launcher, expected):
    """Return the exact-image PID of a live launcher or its direct child."""
    if launcher.poll() is not None:
        return None
    api = _kernel()
    if _matches_image(api, launcher.pid, expected):
        return launcher.pid

    matches = [pid for pid in _children(api, launcher.pid)
               if _matches_image(api, pid, expected, parent=launcher.pid)]
    if launcher.poll() is not None:
        return None
    if len(matches) > 1:
        raise RuntimeError("multiple exact trial executables descended from the launcher")
    return matches[0] if matches else None
