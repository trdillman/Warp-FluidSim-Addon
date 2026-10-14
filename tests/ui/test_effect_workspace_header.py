"""The kernel header lists the parameters execution actually binds.

``_generate_kernel_boilerplate`` used to fall back to *every* parameter when
the parameter-keys CSV was empty, while ``kernel_definition_from_pg`` binds
none once the CSV has been authored empty, so the header disagreed with what
the kernel receives.
"""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

from test_operator_panel_behavior import _install_bpy_stub, _reload_module
from fluidSim_warp_blender51.sim.effects.builtins import builtin_effect_definitions


def _vortex():
    return next(item for item in builtin_effect_definitions()
                if item.effect_id == "fluidsim.vortex_force")


def _item_and_kernel(csv, authored, *, saved_keys):
    definition = _vortex()
    base_kernel = replace(definition.kernels[0], parameter_keys=saved_keys)
    definition = replace(definition, kernels=(base_kernel,))
    parameters = [
        SimpleNamespace(key=p.key, kind=p.kind.value, label=p.label, unit="NONE",
                        description="")
        for p in definition.parameters
    ]
    kernel = SimpleNamespace(
        kernel_id=base_kernel.kernel_id, label="K", domain="PARTICLES",
        stage="BEFORE_PRESSURE", cadence="EVERY_SUBSTEP", fields=[],
        parameter_keys_csv=csv, parameter_keys_authored=authored)
    item = SimpleNamespace(
        parameters=parameters,
        definition_json=json.dumps(definition.to_dict(), sort_keys=True))
    return item, kernel


def _bound_section(header: str) -> list[str]:
    lines = header.splitlines()
    start = lines.index("# BOUND PARAMETERS:") + 1
    return [line for line in lines[start:] if not line.startswith("# ====")]


def _workspace(monkeypatch):
    _install_bpy_stub(monkeypatch)
    return _reload_module("fluidSim_warp_blender51.blender.effect_workspace")


def test_header_shows_none_bound_when_keys_were_authored_empty(monkeypatch):
    workspace = _workspace(monkeypatch)
    keys = tuple(p.key for p in _vortex().parameters)
    item, kernel = _item_and_kernel("", True, saved_keys=keys)

    assert _bound_section(workspace._generate_kernel_boilerplate(kernel, item)) == [
        "#   (None bound)"]


def test_header_uses_saved_keys_for_legacy_unauthored_kernel(monkeypatch):
    workspace = _workspace(monkeypatch)
    saved = (_vortex().parameters[0].key,)
    item, kernel = _item_and_kernel("", False, saved_keys=saved)

    bound = _bound_section(workspace._generate_kernel_boilerplate(kernel, item))
    assert len(bound) == 1 and bound[0].startswith(f"#   {saved[0]} (")


def test_header_follows_csv(monkeypatch):
    workspace = _workspace(monkeypatch)
    second = _vortex().parameters[1].key
    item, kernel = _item_and_kernel(second, True, saved_keys=())

    bound = _bound_section(workspace._generate_kernel_boilerplate(kernel, item))
    assert len(bound) == 1 and bound[0].startswith(f"#   {second} (")
