"""3.0 default Transfer = APIC for new domains; pre-3.0 domains stay FLIP/JACOBI."""

import importlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from test_pending_settings_registry import _modules  # noqa: E402


class _Obj(dict):
    """An object double: custom properties are the dict, stored raw group is `stored`."""

    def __init__(self, stored=None):
        super().__init__()
        self.stored = {} if stored is None else {"fluidsim_domain": stored}
        self.fluidsim_domain = SimpleNamespace()
        self.bl_system_properties_get = lambda: self.stored


def _compat(monkeypatch):
    _modules(monkeypatch)
    return importlib.import_module("fluidSim_warp_blender51.blender.file_compat")


def _scene(domain=None):
    return SimpleNamespace(fluidsim=SimpleNamespace(domain_object=domain))


def test_rna_defaults_are_the_3_0_defaults(monkeypatch):
    import re

    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "fluidSim_warp_blender51", "properties.py")
    text = open(path, encoding="utf-8").read()

    def default_of(field):
        block = text[text.index(f"    {field}: EnumProperty("):]
        return re.search(r'default="(\w+)"', block).group(1)

    assert default_of("sample_type") == "APIC"
    # Multigrid missed its per-solve speed gate on dam_break, so it stays opt-in.
    assert default_of("pressure_preconditioner") == "JACOBI"


def test_stored_domain_without_stamp_is_pinned_to_pre_3_0_defaults(monkeypatch):
    fc = _compat(monkeypatch)
    old = _Obj({"time_scale": 1.0})
    assert fc.migrate_domain_defaults([old], scenes=()) == 1
    assert old.fluidsim_domain.sample_type == "FLIP"
    assert old.fluidsim_domain.pressure_preconditioner == "JACOBI"
    assert old[fc.DEFAULTS_EPOCH_KEY] == fc.DEFAULTS_EPOCH


def test_untouched_designated_domain_is_old_too(monkeypatch):
    fc = _compat(monkeypatch)
    old = _Obj(None)  # nothing stored: an all-default 2.x domain
    assert fc.migrate_domain_defaults([old], scenes=(_scene(old),)) == 1
    assert old.fluidsim_domain.sample_type == "FLIP"


def test_stored_values_are_kept(monkeypatch):
    fc = _compat(monkeypatch)
    old = _Obj({"sample_type": "PIC"})
    old.fluidsim_domain.sample_type = "PIC"
    fc.migrate_domain_defaults([old], scenes=())
    assert old.fluidsim_domain.sample_type == "PIC"
    assert old.fluidsim_domain.pressure_preconditioner == "JACOBI"


def test_new_domain_is_stamped_and_not_migrated(monkeypatch):
    fc = _compat(monkeypatch)
    new = _Obj(None)
    assert fc.stamp_new_domain(new) is True
    assert fc.migrate_domain_defaults([new], scenes=(_scene(new),)) == 0
    assert not hasattr(new.fluidsim_domain, "sample_type")  # APIC stays the RNA default


def test_stamp_skips_objects_with_stored_settings(monkeypatch):
    fc = _compat(monkeypatch)
    old = _Obj({"time_scale": 1.0})
    assert fc.stamp_new_domain(old) is False
    assert fc.DEFAULTS_EPOCH_KEY not in old


def test_migration_is_idempotent_and_ignores_plain_objects(monkeypatch):
    fc = _compat(monkeypatch)
    old, plain = _Obj({"time_scale": 1.0}), _Obj(None)
    assert fc.migrate_domain_defaults([old, plain], scenes=()) == 1
    assert fc.migrate_domain_defaults([old, plain], scenes=()) == 0
    assert fc.DEFAULTS_EPOCH_KEY not in plain


def test_gallery_and_append_paths_run_the_defaults_migration(monkeypatch):
    fc = _compat(monkeypatch)
    calls = []
    monkeypatch.setattr(fc, "migrate_domain_resolution", lambda **k: calls.append("res"))
    monkeypatch.setattr(fc, "migrate_domain_defaults", lambda **k: calls.append("def"))
    fc._on_blend_import()
    assert calls == ["res", "def"]
    src = open(os.path.join(os.path.dirname(fc.__file__), "demo_gallery.py"),
               encoding="utf-8").read()
    assert "migrate_loaded_domains(objects=tuple(scene.objects)" in src
