"""4.0 lane D: Multigrid is the default preconditioner, older files keep Jacobi.

A domain saved before the switch that never stored a preconditioner ran
Jacobi. Opening it in 4.0 must not change its results, so the load migration
writes Jacobi explicitly and leaves a flag for the one-time Bake-stage notice.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pending_settings_registry import _modules  # noqa: E402


def _file_compat(monkeypatch):
    _modules(monkeypatch)
    return importlib.import_module("fluidSim_warp_blender51.blender.file_compat")


def _domain(stored_group, *, library=None, preconditioner="MULTIGRID"):
    """A domain object whose raw group can grow, with an RNA-like settings group."""
    stored = {} if stored_group is None else {"fluidsim_domain": stored_group}
    obj = SimpleNamespace(library=library,
                          bl_system_properties_get=lambda do_create=False: stored)
    obj.fluidsim_domain = SimpleNamespace(pressure_preconditioner=preconditioner,
                                          id_data=obj)
    return obj, stored


def test_pre_4_domain_without_a_stored_preconditioner_keeps_jacobi(monkeypatch):
    fc = _file_compat(monkeypatch)
    old, old_raw = _domain({"domain_schema_version": 3, "time_scale": 1.0})
    unstamped, _ = _domain({"resolution_divisions": 64})

    assert fc.migrate_domain_preconditioner([old, unstamped], scenes=()) == 2
    assert old.fluidsim_domain.pressure_preconditioner == "JACOBI"
    assert unstamped.fluidsim_domain.pressure_preconditioner == "JACOBI"
    assert old_raw["fluidsim_domain"][fc.KEPT_JACOBI_KEY] == 1
    assert fc.shows_kept_jacobi_notice(old.fluidsim_domain)


def test_an_unwritten_designated_domain_keeps_jacobi(monkeypatch):
    # A 3.x scene's domain whose group was never written also ran Jacobi.
    fc = _file_compat(monkeypatch)
    bare, bare_raw = _domain(None)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(domain_object=bare))

    assert fc.migrate_domain_preconditioner(objects=[], scenes=[scene]) == 1
    assert bare.fluidsim_domain.pressure_preconditioner == "JACOBI"
    assert bare_raw["fluidsim_domain"][fc.KEPT_JACOBI_KEY] == 1


def test_stored_choices_4_0_files_and_linked_data_are_left_alone(monkeypatch):
    fc = _file_compat(monkeypatch)
    chose_mg, chose_raw = _domain({"domain_schema_version": 3,
                                   "pressure_preconditioner": 1})
    new, new_raw = _domain({"domain_schema_version": fc.MULTIGRID_DEFAULT_SCHEMA})
    linked, _ = _domain({"time_scale": 1.0}, library="lib.blend")

    assert fc.migrate_domain_preconditioner([chose_mg, new, linked], scenes=()) == 0
    for obj in (chose_mg, new, linked):
        assert obj.fluidsim_domain.pressure_preconditioner == "MULTIGRID"
    assert fc.KEPT_JACOBI_KEY not in chose_raw["fluidsim_domain"]
    assert fc.KEPT_JACOBI_KEY not in new_raw["fluidsim_domain"]


def test_migration_is_idempotent_once_the_choice_is_stored(monkeypatch):
    fc = _file_compat(monkeypatch)
    old, old_raw = _domain({"domain_schema_version": 3})
    assert fc.migrate_domain_preconditioner([old], scenes=()) == 1
    # The real RNA stores the explicit value; the double does not, so mimic it.
    old_raw["fluidsim_domain"]["pressure_preconditioner"] = 0
    assert fc.migrate_domain_preconditioner([old], scenes=()) == 0


def test_save_stamps_the_multigrid_schema(monkeypatch):
    fc = _file_compat(monkeypatch)
    assert fc.CURRENT_DOMAIN_SCHEMA >= fc.MULTIGRID_DEFAULT_SCHEMA == 4


def test_notice_hides_after_a_switch_or_a_keep(monkeypatch):
    fc = _file_compat(monkeypatch)
    old, old_raw = _domain({"domain_schema_version": 3})
    fc.migrate_domain_preconditioner([old], scenes=())

    # Switching by hand hides it even before the flag is cleared.
    old.fluidsim_domain.pressure_preconditioner = "MULTIGRID"
    assert not fc.shows_kept_jacobi_notice(old.fluidsim_domain)
    old.fluidsim_domain.pressure_preconditioner = "JACOBI"
    assert fc.shows_kept_jacobi_notice(old.fluidsim_domain)

    fc.clear_kept_jacobi_notice(old.fluidsim_domain)
    assert fc.KEPT_JACOBI_KEY not in old_raw["fluidsim_domain"]
    assert not fc.shows_kept_jacobi_notice(old.fluidsim_domain)
    assert not fc.shows_kept_jacobi_notice(None)


def test_load_and_append_handlers_run_the_migration(monkeypatch):
    fc = _file_compat(monkeypatch)
    calls = []
    monkeypatch.setattr(fc, "migrate_domain_resolution", lambda *a, **k: 0)
    monkeypatch.setattr(fc, "migrate_removed_particle_color_mode", lambda *a, **k: 0)
    monkeypatch.setattr(fc, "migrate_domain_preconditioner",
                        lambda *a, **k: calls.append(k) or 0)
    monkeypatch.setattr(fc.bpy, "data", SimpleNamespace(scenes=(), objects=()),
                        raising=False)
    fc._on_load_post(None)
    assert calls == [{}]


def test_notice_operator_switches_or_keeps(monkeypatch):
    _modules(monkeypatch)
    presets = importlib.import_module("fluidSim_warp_blender51.operator_domains.presets")
    fc = importlib.import_module("fluidSim_warp_blender51.blender.file_compat")
    scene_access = importlib.import_module("fluidSim_warp_blender51.blender.scene_access")
    old, old_raw = _domain({"domain_schema_version": 3})
    fc.migrate_domain_preconditioner([old], scenes=())
    monkeypatch.setattr(scene_access, "active_domain_settings",
                        lambda _scene: old.fluidsim_domain)
    context = SimpleNamespace(scene=SimpleNamespace())

    op = presets.FLUIDSIM_OT_preconditioner_notice()
    op.action = "KEEP"
    assert op.execute(context) == {"FINISHED"}
    assert old.fluidsim_domain.pressure_preconditioner == "JACOBI"
    assert fc.KEPT_JACOBI_KEY not in old_raw["fluidsim_domain"]

    old_raw["fluidsim_domain"][fc.KEPT_JACOBI_KEY] = 1
    op.action = "MULTIGRID"
    assert op.execute(context) == {"FINISHED"}
    assert old.fluidsim_domain.pressure_preconditioner == "MULTIGRID"
    assert fc.KEPT_JACOBI_KEY not in old_raw["fluidsim_domain"]
