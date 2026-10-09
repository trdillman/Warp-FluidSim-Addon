"""Narrow-band admission rules: pure, bpy-free, one answer for every caller."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim import narrow_band as nb
from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings


def _settings(**overrides):
    values = dict(
        species_enabled=False,
        viscosity_enabled=False,
        viscosity_from_particles=False,
        export_attributes=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _object(role, field_operation="PRESSURE"):
    return SimpleNamespace(
        fluidsim=SimpleNamespace(role=role, field_operation=field_operation))


def test_clean_setup_has_no_reasons():
    assert nb.blocking_reasons(_settings()) == ()


def test_color_mixing_blocks():
    assert nb.blocking_reasons(_settings(species_enabled=True)) == (
        nb.REASON_COLOR_MIXING,)


def test_viscosity_from_particles_blocks_only_with_viscosity_on():
    only_flag = _settings(viscosity_from_particles=True)
    assert nb.blocking_reasons(only_flag) == ()
    both = _settings(viscosity_enabled=True, viscosity_from_particles=True)
    assert nb.blocking_reasons(both) == (nb.REASON_VISCOSITY_FROM_PARTICLES,)
    assert nb.blocking_reasons(_settings(viscosity_enabled=True)) == ()


def test_scalar_field_source_blocks():
    assert nb.blocking_reasons(
        _settings(), scalar_field_source=True) == (nb.REASON_SCALAR_FIELD_SOURCE,)


def test_export_attributes_blocks():
    assert nb.blocking_reasons(_settings(export_attributes=True)) == (
        nb.REASON_EXPORT_ATTRIBUTES,)


def test_combination_lists_every_reason_in_a_stable_order():
    reasons = nb.blocking_reasons(
        _settings(
            species_enabled=True, viscosity_enabled=True,
            viscosity_from_particles=True, export_attributes=True),
        scalar_field_source=True)
    assert reasons == (
        nb.REASON_COLOR_MIXING,
        nb.REASON_VISCOSITY_FROM_PARTICLES,
        nb.REASON_SCALAR_FIELD_SOURCE,
        nb.REASON_EXPORT_ATTRIBUTES,
    )


def test_missing_attributes_read_as_not_blocking():
    assert nb.blocking_reasons(SimpleNamespace()) == ()


@pytest.mark.parametrize(
    ("objects", "expected"),
    [
        ([], False),
        ([_object("SOURCE")], False),
        ([_object("FIELD", "PRESSURE")], False),
        ([_object("FIELD", "VECTOR")], False),
        ([_object("SOURCE", "SCALAR")], False),
        ([_object("FIELD", "SCALAR")], True),
        ([_object("SOURCE"), _object("FIELD", "SCALAR")], True),
        ([SimpleNamespace()], False),
    ],
)
def test_only_a_field_role_with_scalar_target_counts(objects, expected):
    assert nb.has_scalar_field_source(objects) is expected


def test_scene_blocking_reasons_reads_field_sources_from_objects():
    reasons = nb.scene_blocking_reasons(
        _settings(), [_object("FIELD", "SCALAR")])
    assert reasons == (nb.REASON_SCALAR_FIELD_SOURCE,)
    assert nb.scene_blocking_reasons(_settings(), [_object("SOURCE")]) == ()


def test_refusal_message_names_every_reason():
    message = nb.refusal_message(
        (nb.REASON_COLOR_MIXING, nb.REASON_EXPORT_ATTRIBUTES))
    assert nb.REASON_COLOR_MIXING in message
    assert nb.REASON_EXPORT_ATTRIBUTES in message
    assert "Narrow-Band" in message


def test_solver_settings_default_off_with_three_voxel_band():
    defaults = SolverSettings()
    assert defaults.narrow_band_enabled is False
    assert defaults.narrow_band_voxels == 3.0


def test_module_does_not_import_bpy_or_warp():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(nb))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"bpy", "warp", "mathutils"}
