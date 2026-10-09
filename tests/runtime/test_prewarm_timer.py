"""The deferred prewarm timer registers once, unregisters cleanly, never raises."""

from __future__ import annotations

import sys
import types

from fluidSim_warp_blender51 import registration
from fluidSim_warp_blender51.runtime import prewarm_timer


def _fake_bpy(monkeypatch, *, broken=False):
    registered = []
    timers = types.SimpleNamespace()
    if broken:
        def boom(*_a, **_k):
            raise RuntimeError("timers unavailable")
        timers.register = timers.unregister = timers.is_registered = boom
    else:
        timers.register = lambda fn, first_interval=0.0: registered.append((fn, first_interval))
        timers.unregister = lambda fn: registered.remove(
            next(entry for entry in registered if entry[0] is fn))
        timers.is_registered = lambda fn: any(entry[0] is fn for entry in registered)
    bpy = types.ModuleType("bpy")
    bpy.app = types.SimpleNamespace(timers=timers)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    return registered


def test_the_timer_registers_once_and_unregisters(monkeypatch):
    registered = _fake_bpy(monkeypatch)

    prewarm_timer.register_timer()
    prewarm_timer.register_timer()
    assert registered == [(prewarm_timer._background_prewarm, 0.5)]

    prewarm_timer.unregister_timer()
    prewarm_timer.unregister_timer()
    assert registered == []


def test_a_broken_timer_api_never_fails_enable_or_disable(monkeypatch):
    _fake_bpy(monkeypatch, broken=True)

    prewarm_timer.register_timer()
    prewarm_timer.unregister_timer()


def test_without_bpy_the_steps_are_no_ops(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", None)

    prewarm_timer.register_timer()
    prewarm_timer.unregister_timer()


def test_the_prewarm_step_is_last_in_the_registration_table():
    last = registration.STEPS[-1]
    assert (last.module, last.register, last.unregister) == (
        "runtime.prewarm_timer", "register_timer", "unregister_timer")
