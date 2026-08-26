"""Truthful clear-dye output and Blender-adapter contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap
import types
from types import SimpleNamespace
import json

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.frame_output import (
    HostFrame,
    host_output_operation_counts,
)
from fluidSim_warp_blender51.sim.attributes import derive_species_preview_colors


class _Array:
    def __init__(self, values, calls, name):
        self.values = np.asarray(values, dtype=np.float32)
        self.calls = calls
        self.name = name

    def __getitem__(self, item):
        return _Array(self.values[item], self.calls, self.name)

    def numpy(self):
        self.calls[self.name] = self.calls.get(self.name, 0) + 1
        return self.values


def _species_solver(calls, *, enabled=True):
    concentration = _Array([
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
    ], calls, "species")
    particles = SimpleNamespace(
        refresh_host_count=lambda: 3,
        position=_Array([(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0),
                         (2.0, 0.0, 0.0)], calls, "positions"),
        attr=lambda name: concentration if name == "species_concentration" else None,
    )
    return SimpleNamespace(
        frame=4,
        particles=particles,
        species=(SimpleNamespace() if enabled else None),
        settings=SimpleNamespace(
            species_enabled=enabled,
            species_absorption_rgb=((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)),
            species_reference_path_length=1.0,
            voxel_size=0.5,
            frame_rate=24.0,
            time_scale=1.0,
        ),
    )


def test_species_host_accessors_are_lazy_aligned_and_share_one_readback():
    calls = {}
    host_output_operation_counts(reset=True)
    frame = HostFrame(_species_solver(calls))

    assert calls == {}
    np.testing.assert_array_equal(
        frame.species_concentrations(stride=2),
        np.array([(0.0, 0.0, 0.0, 0.0),
                  (0.5, 0.5, 0.0, 0.0)], dtype=np.float32),
    )
    frame.species_concentrations(stride=2)
    preview = frame.species_preview_colors(stride=2)
    frame.species_preview_colors(stride=2)

    assert calls == {"species": 1}
    assert host_output_operation_counts()["device_readbacks"] == 1
    np.testing.assert_allclose(preview[0], (1.0, 1.0, 1.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(
        preview[1],
        (np.exp(-2.0), np.exp(-2.0), np.exp(-2.0), 1.0),
        rtol=1e-6,
    )


def test_disabled_species_accessors_do_not_read_particle_state():
    calls = {}
    host_output_operation_counts(reset=True)
    frame = HostFrame(_species_solver(calls, enabled=False))

    assert frame.species_concentrations().shape == (0, 4)
    assert frame.species_preview_colors().shape == (0, 4)
    assert calls == {}
    assert host_output_operation_counts()["device_readbacks"] == 0


def test_disabled_species_ignores_stale_species_output_toggles():
    """A reinitialized solver may outlive authored Clear Dye UI choices."""
    calls = {}
    solver = _species_solver(calls, enabled=False)
    settings = SimpleNamespace(
        particles_attribute_species=True,
        surface_attribute_species=True,
    )
    host = HostFrame(solver)

    assert host.output_attributes(settings, "particles") == {}
    assert host.output_attributes(
        settings,
        "surface",
        positions=np.array([(0.5, 0.0, 0.0)], dtype=np.float32),
    ) == {}
    assert calls == {}


def test_authoritative_species_and_derived_color_share_particle_and_surface_state():
    calls = {}
    solver = _species_solver(calls)
    solver.settings.particles_attribute_species = True
    solver.settings.particles_attribute_color = True
    solver.settings.surface_attribute_species = True
    solver.settings.surface_attribute_color = True
    host = HostFrame(solver)

    particles = host.output_attributes(solver.settings, "particles")
    surface = host.output_attributes(
        solver.settings,
        "surface",
        positions=np.array([(0.1, 0.0, 0.0), (1.9, 0.0, 0.0)], np.float32),
    )

    assert set(particles) == {"species", "color"}
    assert set(surface) == {"species", "color"}
    np.testing.assert_array_equal(surface["species"], particles["species"][[0, 2]])
    np.testing.assert_allclose(
        surface["color"],
        derive_species_preview_colors(
            surface["species"],
            solver.settings.species_absorption_rgb,
            solver.settings.species_reference_path_length,
        ),
        rtol=1e-6,
    )
    assert calls["species"] == 1


def _publication_scene(**overrides):
    values = dict(
        show_particles=True,
        show_surface=True,
        show_whitewater=False,
        whitewater_enabled=False,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        whitewater_display_scale=1.0,
        particle_attribute_mode="CLEAR_DYE",
        surface_attribute_mode="CLEAR_DYE",
        whitewater_attribute_mode="NONE",
        surface_enabled=True,
        voxel_size=0.5,
        particles_attribute_species=True,
        surface_attribute_species=True,
    )
    values.update(overrides)
    return SimpleNamespace(fluidsim=SimpleNamespace(**values))


def test_live_publication_does_not_request_species_when_unselected():
    from fluidSim_warp_blender51.blender.output_publication import prepare_live_payload

    calls = {}
    solver = _species_solver(calls)
    scene = _publication_scene(
        show_surface=False,
        particle_attribute_mode="HEIGHT",
        particles_attribute_species=False,
        particles_attribute_color=False,
        surface_attribute_species=False,
        surface_attribute_color=False,
    )

    prepare_live_payload(scene, HostFrame(solver), publish_surface=False)

    assert "species" not in calls


def test_live_publication_falls_back_when_species_is_disabled_but_clear_dye_stays_authored():
    from fluidSim_warp_blender51.blender.output_publication import prepare_live_payload

    calls = {}
    solver = _species_solver(calls, enabled=False)
    scene = _publication_scene(
        show_surface=False,
        particles_attribute_species=True,
        particles_attribute_color=False,
        surface_attribute_species=True,
        surface_attribute_color=False,
    )

    payload = prepare_live_payload(
        scene, HostFrame(solver), publish_surface=False)

    assert payload.particle_attribute_mode == "NONE"
    assert payload.particle_attributes == {}
    assert payload.particles.shape == (3, 3)
    assert calls == {"positions": 1}


def test_live_publication_uses_current_output_only_species_toggle():
    from fluidSim_warp_blender51.blender.output_publication import prepare_live_payload

    calls = {}
    solver = _species_solver(calls)
    solver.settings.particles_attribute_species = False
    solver.settings.particles_attribute_color = False
    scene = _publication_scene(
        show_surface=False,
        particle_attribute_mode="NONE",
        particles_attribute_species=True,
        particles_attribute_color=False,
        surface_attribute_species=False,
        surface_attribute_color=False,
    )

    payload = prepare_live_payload(
        scene, HostFrame(solver), publish_surface=False)

    np.testing.assert_array_equal(
        payload.particle_attributes["species"],
        np.array([
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0),
        ], dtype=np.float32),
    )


def test_cache_playback_derives_species_presentation_without_solver_policy(
        monkeypatch, blender_adapters):
    (_properties, _panels, pending_settings, solver_settings,
     output_publication) = blender_adapters

    monkeypatch.setattr(
        solver_settings, "build_settings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache playback must not build solver settings")))
    monkeypatch.setattr(
        pending_settings, "apply_scene_settings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache playback must not mutate solver policy")),
        raising=False,
    )
    palette = {
        "ids": ["dye_a", "dye_b"],
        "absorption_rgb": [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        "reference_path_length": 1.0,
    }
    concentrations = np.array([
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
    ], dtype=np.float32)
    frame_data = {
        "positions": np.array([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], np.float32),
        "velocities": np.zeros((2, 3), np.float32),
        "surface_verts": np.array([(0.1, 0.0, 0.0), (1.9, 0.0, 0.0)], np.float32),
        "surface_tris": np.zeros((0, 3), np.int32),
        "particle_species_concentration": concentrations,
        "species_schema_version": np.array(1, np.int32),
        "species_palette_json": np.array(json.dumps(palette)),
        "voxel_size": np.array(0.5, np.float32),
    }

    payload = output_publication.prepare_cache_payload(
        _publication_scene(), frame_data)

    np.testing.assert_array_equal(
        payload.particle_attributes["species"], concentrations)
    np.testing.assert_array_equal(
        payload.surface_attributes["species"], concentrations)
    np.testing.assert_allclose(
        payload.surface_attributes["color"],
        derive_species_preview_colors(
            concentrations, palette["absorption_rgb"], 1.0),
        rtol=1e-6,
    )


@pytest.fixture
def blender_adapters(monkeypatch):
    bpy = types.ModuleType("bpy")

    class _Operator:
        @classmethod
        def poll_message_set(cls, _message):
            pass

        def report(self, *_args, **_kwargs):
            pass

    bpy.types = SimpleNamespace(
        PropertyGroup=object, Text=object, Object=object, Material=object,
        Image=object, Operator=_Operator, Panel=object, AddonPreferences=object,
    )
    bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(addons={}), screen=SimpleNamespace(areas=[]))
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(frame_change_post=[]))
    bpy.utils = SimpleNamespace(register_class=lambda _cls: None,
                                unregister_class=lambda _cls: None)
    bpy.props = types.ModuleType("bpy.props")
    for name in (
            "BoolProperty", "CollectionProperty", "EnumProperty",
            "FloatProperty", "FloatVectorProperty", "IntProperty",
            "PointerProperty", "StringProperty"):
        setattr(bpy.props, name, lambda **kwargs: kwargs.get("default"))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", bpy.props)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    names = (
        "fluidSim_warp_blender51.properties",
        "fluidSim_warp_blender51.panels",
        "fluidSim_warp_blender51.blender.pending_settings",
        "fluidSim_warp_blender51.blender.solver_settings",
        "fluidSim_warp_blender51.blender.output_publication",
    )
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in names}
    original_package_attrs = {}
    for name in names:
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        original_package_attrs[name] = (
            parent, attribute, getattr(parent, attribute, missing))
    modules = tuple(importlib.import_module(name) for name in names)
    for module in modules:
        if hasattr(module, "bpy"):
            monkeypatch.setattr(module, "bpy", bpy)
    try:
        yield modules
    finally:
        for name in reversed(names):
            original = original_modules[name]
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
            parent, attribute, package_value = original_package_attrs[name]
            if package_value is missing:
                if hasattr(parent, attribute):
                    delattr(parent, attribute)
            else:
                setattr(parent, attribute, package_value)


def _declared_fields(cls):
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    return {
        node.target.id
        for node in tree.body[0].body
        if (isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Call))
    }


def test_species_rna_schema_and_truthful_panel_labels(blender_adapters):
    properties, panels, _pending, _projection, _publication = blender_adapters
    domain_fields = _declared_fields(properties.FLUIDSIM_PG_settings)
    object_fields = _declared_fields(properties.FLUIDSIM_PG_object)
    assert {
        "species_enabled", "species_count", "species_diffusivity",
        "species_max_diffusion_substeps", "species_id_a", "species_id_b",
        "species_preview_swatch_a", "species_preview_swatch_b",
        "species_absorption_a", "species_absorption_b",
        "species_reference_path_length", "surface_attribute_species",
        "particles_attribute_species",
    }.issubset(domain_fields)
    assert "species_concentration" in object_fields
    settings_source = inspect.getsource(properties.FLUIDSIM_PG_settings)
    assert 'species_count: IntProperty(' in settings_source
    assert 'default=2, min=2, max=2' in settings_source
    panel_source = inspect.getsource(panels.FLUIDSIM_PT_clear_dye_species)
    assert "Clear Dye Species" in panel_source
    assert "Effective Diffusivity" in panel_source
    assert not any(term in panel_source for term in ("Paint", "Pigment", "Water/Oil"))


def test_build_settings_copies_exact_species_palette_values(
        monkeypatch, blender_adapters):
    _properties, _panels, _pending, projection, _publication = blender_adapters
    from fluidSim_warp_blender51 import preferences

    monkeypatch.setattr(preferences, "configured_device", lambda: "cpu")
    fs = SimpleNamespace(
        species_enabled=True,
        species_count=2,
        species_diffusivity=0.0125,
        species_max_diffusion_substeps=7,
        species_id_a="cyan_dye",
        species_id_b="amber_dye",
        species_preview_swatch_a=(0.1, 0.2, 0.3),
        species_preview_swatch_b=(0.4, 0.5, 0.6),
        species_absorption_a=(0.7, 0.8, 0.9),
        species_absorption_b=(1.1, 1.2, 1.3),
        species_reference_path_length=0.75,
        surface_attribute_species=True,
        particles_attribute_species=False,
        open_boundaries=False,
        domain_object=None,
    )
    scene = SimpleNamespace(
        fluidsim=fs,
        render=SimpleNamespace(fps=30, fps_base=1.0),
    )

    settings = projection.build_settings(scene)

    assert settings.species_enabled is True
    assert settings.species_count == 2
    assert settings.species_diffusivity == 0.0125
    assert settings.species_max_diffusion_substeps == 7
    assert settings.species_ids == ("cyan_dye", "amber_dye")
    assert settings.species_preview_swatches == (
        (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert settings.species_absorption_rgb == (
        (0.7, 0.8, 0.9), (1.1, 1.2, 1.3))
    assert settings.species_reference_path_length == 0.75
    assert settings.surface_attribute_species is True
    assert settings.particles_attribute_species is False


def test_species_pending_policy_is_exhaustive_and_uses_existing_source_reset(
        blender_adapters):
    _properties, _panels, pending, _projection, _publication = blender_adapters
    reinitialize = {
        "species_enabled", "species_count", "species_id_a", "species_id_b",
        "species_preview_swatch_a", "species_preview_swatch_b",
        "species_absorption_a", "species_absorption_b",
        "species_reference_path_length",
    }
    reset = {"species_diffusivity", "species_max_diffusion_substeps"}
    output = {"surface_attribute_species", "particles_attribute_species"}

    assert all(pending.SETTING_TRANSITION_REGISTRY[name]
               is pending.SettingTier.REINITIALIZE for name in reinitialize)
    assert all(pending.SETTING_TRANSITION_REGISTRY[name]
               is pending.SettingTier.RESET for name in reset)
    assert all(pending.SETTING_TRANSITION_REGISTRY[name]
               is pending.SettingTier.OUTPUT for name in output)
    assert "species_concentration" in pending._RESET_OBJECT_FIELDS
    assert reset.issubset(pending._RESET_SETTING_FIELDS)
    assert output.isdisjoint(pending._RESET_SETTING_FIELDS)
