"""Exhaustive/disjoint Pending Settings transition registry contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import sys
import textwrap
import types
from types import SimpleNamespace


def _modules(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(frame_change_post=[]))

    class _Operator:
        @classmethod
        def poll_message_set(cls, _message):
            pass

        def report(self, *_args, **_kwargs):
            pass

    bpy.types = SimpleNamespace(
        PropertyGroup=object,
        Text=object,
        Object=object,
        Material=object,
        Image=object,
        Operator=_Operator,
        Panel=object,
        AddonPreferences=object,
    )
    bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None, unregister_class=lambda _cls: None
    )
    bpy.props = types.ModuleType("bpy.props")
    for name in (
        "BoolProperty",
        "CollectionProperty",
        "EnumProperty",
        "FloatProperty",
        "FloatVectorProperty",
        "IntProperty",
        "PointerProperty",
        "StringProperty",
    ):
        setattr(bpy.props, name, lambda **kwargs: kwargs.get("default"))
    monkeypatch.setitem(sys.modules, "bpy.props", bpy.props)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    for name in (
        "fluidSim_warp_blender51.properties",
        "fluidSim_warp_blender51.blender.pending_settings",
        "fluidSim_warp_blender51.blender.scene_bridge",
    ):
        sys.modules.pop(name, None)
    properties = importlib.import_module("fluidSim_warp_blender51.properties")
    pending = importlib.import_module(
        "fluidSim_warp_blender51.blender.pending_settings"
    )
    bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    return properties, pending, bridge


def _declared_fields(settings_cls):
    tree = ast.parse(textwrap.dedent(inspect.getsource(settings_cls)))
    return {
        node.target.id
        for node in tree.body[0].body
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Call)
        )
    }


def _declared_settings_fields(properties):
    return (_declared_fields(properties.FLUIDSIM_PG_scene)
            | _declared_fields(properties.FLUIDSIM_PG_domain))


def test_transition_registry_is_exhaustive_and_disjoint(monkeypatch):
    properties, pending, _bridge = _modules(monkeypatch)
    declared = _declared_settings_fields(properties)

    pending.validate_transition_registry(declared)

    assert set(pending.SETTING_TRANSITION_REGISTRY) == declared
    assert all(
        isinstance(tier, pending.SettingTier)
        for tier in pending.SETTING_TRANSITION_REGISTRY.values()
    )
    assert len(pending.SETTING_TRANSITION_REGISTRY) == len(declared)


def test_special_particle_capacity_policy_is_explicit(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    assert pending.SETTING_TRANSITION_REGISTRY["max_particles"] is (
        pending.SettingTier.REINITIALIZE
    )
    assert pending.SETTING_TRANSITION_REGISTRY["auto_grow_particles"] is (
        pending.SettingTier.LIVE
    )
    assert (
        pending.SETTING_TRANSITION_REGISTRY["particle_growth_failure_policy"]
        is pending.SettingTier.LIVE
    )
    # Preserve the characterized Phase-3 behavior: this authored value is not
    # pushed into an active solver by the surface refresh callback.
    assert (
        pending.SETTING_TRANSITION_REGISTRY["spacetime_resync_surface"]
        is pending.SettingTier.AUTHORED_NEXT_RUN
    )


def test_quality_tier_is_reinitialize_and_detected_by_reinit_signature(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    # The scene-level Quality dial rescales solver knobs, so a change must
    # re-derive grid allocation (rebuild), not read as a pure presentation edit.
    assert (
        pending.SETTING_TRANSITION_REGISTRY["quality"]
        is pending.SettingTier.REINITIALIZE
    )
    assert "quality" not in pending._PRESENTATION_FIELDS
    # The reinit signature (which drives auto-detection) reads scene.fluidsim,
    # exactly where quality is authored, so quality must be in its field set.
    assert "quality" in pending._REINIT_SETTING_FIELDS


def test_quality_property_wires_the_rebuild_update_callback(monkeypatch):
    """A REINITIALIZE tier classification is inert without an RNA callback.

    The stub in ``_modules`` discards every ``bpy.props.*`` keyword (it only
    ever returns ``default``), so a change to the ``update=`` argument cannot
    be observed by calling the property -- it has to be read out of source.
    Without this, a UI edit to Quality would only refresh cached presentation
    on the depsgraph handler and never reach ``request_scene_update()``,
    leaving an active solver running the previous voxel size/substeps/sampling
    until the user manually applies the pending action or rewinds.
    """

    properties, _pending, _bridge = _modules(monkeypatch)
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(properties.FLUIDSIM_PG_scene))
    )
    (quality_node,) = (
        node
        for node in tree.body[0].body
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "quality"
        )
    )
    update_kwargs = {
        kw.arg: kw.value
        for kw in quality_node.annotation.keywords
        if kw.arg == "update"
    }
    assert "update" in update_kwargs, (
        "quality's EnumProperty has no update= callback; UI edits to Quality "
        "will not trigger a REINITIALIZE rebuild"
    )
    assert (
        isinstance(update_kwargs["update"], ast.Name)
        and update_kwargs["update"].id == "_request_rebuild"
    )


def test_species_toggle_rebuilds_storage_but_rate_updates_live(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    assert (
        pending.SETTING_TRANSITION_REGISTRY["species_enabled"]
        is pending.SettingTier.RESET
    )
    assert (
        pending.SETTING_TRANSITION_REGISTRY["species_mixing_rate"]
        is pending.SettingTier.LIVE
    )


def test_scene_bridge_no_longer_owns_pending_policy(monkeypatch):
    properties, pending, bridge = _modules(monkeypatch)
    source = inspect.getsource(bridge)

    assert "class SceneUpdate" not in source
    assert "_INTERACTION_STATES" not in source
    assert "def pending_presentation" not in source
    assert "def request_scene_update" not in source
    assert "def apply_pending_" not in source
    assert "def build_snippets" not in source
    assert "def validate_snippet_pg" not in source
    assert "def gather_sources" not in source
    assert "def gather_colliders" not in source
    assert "def register_handlers" not in source
    assert pending.SceneUpdate.REBUILD.value == "REBUILD"
    assert "pending_settings.request_scene_update" in inspect.getsource(properties)


def test_adapters_delegate_pending_policy_to_deep_owner(monkeypatch):
    _properties, _pending, _bridge = _modules(monkeypatch)
    root = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
    sources = [
        (root / "operator_domains" / "common.py").read_text(encoding="utf-8"),
        (root / "panels.py").read_text(encoding="utf-8"),
        (root / "blender" / "ui_common.py").read_text(encoding="utf-8"),
        (root / "blender" / "ui_scene.py").read_text(encoding="utf-8"),
        (root / "blender" / "viewport_overlay.py").read_text(encoding="utf-8"),
        (root / "blender" / "playback.py").read_text(encoding="utf-8"),
    ]

    assert "def _pending" in sources[0]
    assert "pending_ui_state" in sources[1]
    assert "pending_settings import pending_presentation" in sources[2]
    assert "pending_ui_state" in sources[3]
    assert "pending_settings import pending_presentation" in sources[4]
    assert "pending_settings.apply_pending_on_rewind" in sources[5]


def _migrated_scene():
    """A post-migration scene, the only state in which the split is observable.

    ``active_domain_settings`` returns ``obj.fluidsim_domain`` only when
    ``schema_version == 1`` and a domain object is designated; before that both
    owners are the same datablock and every ownership question is trivially
    satisfied.
    """
    domain_group = SimpleNamespace(_label="domain")
    domain_object = SimpleNamespace(fluidsim_domain=domain_group)
    scene_group = SimpleNamespace(
        _label="scene", schema_version=1, domain_object=domain_object
    )
    return SimpleNamespace(fluidsim=scene_group), scene_group, domain_group


def test_setting_owner_classifies_every_setting_tier(monkeypatch):
    """Ownership must be total over the tier enum, not just over today's fields.

    ``setting_owner`` partitions by ``SettingTier``; a tier added later and left
    out of ``_SCENE_OWNED_TIERS`` would silently default to the domain group.
    On a migrated scene that means writing the live group while the UI and the
    consumers still read the frozen ghost, which is the defect-#5 failure mode.
    So the partition is asserted exhaustively over the enum itself rather than
    over the fields that happen to be registered today.
    """
    _properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, domain_group = _migrated_scene()

    # Total over the enum: every member is on exactly one side of the split.
    assert set(pending._TIER_FIELDS) == set(pending.SettingTier)
    assert pending._SCENE_OWNED_TIERS <= set(pending.SettingTier)

    # Total over the registry: every field resolves to a real datablock.
    for name in pending.SETTING_TRANSITION_REGISTRY:
        owner = pending.setting_owner(scene, name)
        assert owner is scene_group or owner is domain_group, name

    # And the classification is not vacuous: each tier's fields actually route
    # to the side its tier declares, modulo the documented field overrides.
    for tier, names in pending._TIER_FIELDS.items():
        for name in names:
            if (name in pending._SCENE_OWNED_FIELDS
                    or name in pending._DOMAIN_OWNED_FIELDS):
                continue  # field-level override, pinned separately below
            expected = (
                scene_group if tier in pending._SCENE_OWNED_TIERS else domain_group
            )
            assert pending.setting_owner(scene, name) is expected, (tier, name)


def test_setting_owner_pins_the_scene_owned_partition(monkeypatch):
    """Pin both halves of the declaration: the tier set and the field overrides."""
    _properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, domain_group = _migrated_scene()

    assert pending._SCENE_OWNED_TIERS == frozenset({
        pending.SettingTier.OUTPUT,
        pending.SettingTier.AUTHORED_NEXT_RUN,
        pending.SettingTier.ADAPTER,
        pending.SettingTier.PRESENTATION,
        pending.SettingTier.SYNC,
    })
    assert pending._SCENE_OWNED_FIELDS == frozenset({"quality", "domain_object"})
    assert pending._DOMAIN_OWNED_FIELDS == frozenset({
        "height_error", "spacetime_resync_surface"})
    # The two overrides must never claim the same field.
    assert not (pending._SCENE_OWNED_FIELDS & pending._DOMAIN_OWNED_FIELDS)

    # One representative key per scene-owned tier, through the public API.
    for name in ("liquid_view_mode", "cache_compress", "live", "ui_mode"):
        assert pending.setting_owner(scene, name) is scene_group, name
    # ... and one per domain-owned tier.
    for name in ("viscosity_enabled", "surface_enabled", "whitewater_enabled",
                 "height_enabled", "voxel_size"):
        assert pending.setting_owner(scene, name) is domain_group, name

    # An unregistered name has no tier. It must fall to the domain group, not
    # silently claim the scene, so a typo cannot quietly write the ghost.
    assert pending.setting_owner(scene, "not_a_registered_field") is domain_group


def test_setting_owner_routes_scene_read_fields_to_the_scene_group(monkeypatch):
    """``quality`` and ``domain_object`` are read off ``scene.fluidsim``, so the
    tier partition alone gives the wrong answer for them.

    Both are REINITIALIZE, which is not a scene-owned tier, yet:

    * ``solver_settings.scene_quality_fidelity()`` reads ``scene.fluidsim.quality``
      deliberately (see its docstring) to keep the read owner and the UI write
      owner identical on migrated and pre-migration scenes alike.
    * ``migration._MIGRATION_SKIP`` never copies ``domain_object`` onto the
      domain group at all, so that copy is permanently unset; ``domain_bounds``
      / ``domain_transform`` and the rendering and session operators all read it
      from ``scene.fluidsim``.

    Routing either through the domain group would write a value nothing reads.
    """
    _properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, _domain_group = _migrated_scene()

    for name in ("quality", "domain_object"):
        assert pending.SETTING_TRANSITION_REGISTRY[name] is (
            pending.SettingTier.REINITIALIZE
        ), name
        assert pending.setting_owner(scene, name) is scene_group, name


def test_setting_owner_routes_domain_read_fields_to_the_domain_group(monkeypatch):
    """The mirror of the ``quality``/``domain_object`` override.

    ``height_error`` and ``spacetime_resync_surface`` sit in scene-owned tiers,
    and their tiers are correct as *lifecycle* statements: a status string
    triggers no solver action (PRESENTATION), and the resync flag is not pushed
    into a live solver (AUTHORED_NEXT_RUN). But every consumer of both reads the
    domain group -- scene_inputs writes ``height_error`` there and ui_scene
    draws it from ``_domain_owner``; ui_scene draws ``spacetime_resync_surface``
    there and ``apply_scene_settings`` consumes it from the group its caller
    passes, always ``active_domain_settings``. Routing either to the scene group
    would write the frozen ghost that nothing reads.

    This also gates the Phase 2 split: the class partition is derived from this
    declaration, so a scene-owned answer here would put both fields on
    ``FLUIDSIM_PG_scene`` and strand all 11 of their call sites.
    """
    _properties, pending, _bridge = _modules(monkeypatch)
    scene, _scene_group, domain_group = _migrated_scene()

    assert pending.SETTING_TRANSITION_REGISTRY["height_error"] is (
        pending.SettingTier.PRESENTATION
    )
    assert pending.SETTING_TRANSITION_REGISTRY["spacetime_resync_surface"] is (
        pending.SettingTier.AUTHORED_NEXT_RUN
    )
    for name in ("height_error", "spacetime_resync_surface"):
        assert pending.setting_owner(scene, name) is domain_group, name


def test_authoring_owner_answers_a_different_question_than_setting_owner(monkeypatch):
    """The two owner functions are not redundant, and their divergence is pinned.

    ``_authoring_owner`` resolves where the *UI writes* a signature field, so a
    user edit is detectable as pending. ``setting_owner`` resolves where a value
    must *land to take effect*, so a preset write is not swallowed. They agree
    for most fields and legitimately diverge wherever the panel writes
    ``scene.fluidsim`` while the solver builds from the active domain settings.
    Pinning that divergence keeps it a reviewed list rather than an accident.
    """
    _properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, domain_group = _migrated_scene()

    signature_fields = set(pending._RESET_SCENE_FIELDS) | set(
        pending._REINIT_SETTING_FIELDS
    )
    diverging = {
        name
        for name in signature_fields
        if (pending.setting_owner(scene, name) is scene_group)
        != (pending._authoring_owner(scene_group, domain_group, name) is scene_group)
    }

    # Every diverging field is authored by its panel on scene.fluidsim but read
    # by its consumer from the domain group: height_* via
    # scene_inputs.gather_height_source, *_attribute_* via
    # solver_settings.build_settings, profiling_* via simulation_admission.
    expected = (
        {
            "height_enabled", "height_base", "height_back_mode",
            "height_profile_image", "height_profile_scale",
            "height_emission_mode", "height_velocity", "height_emit_particles",
        }
        | {"profiling_enabled", "profiling_sync"}
        | {
            f"surface_attribute_{suffix}"
            for suffix in (
                "velocity", "speed", "vorticity", "age", "lifetime", "color",
                "source_id", "viscosity", "density", "motion",
            )
        }
        | {
            f"particles_attribute_{suffix}"
            for suffix in (
                "velocity", "speed", "vorticity", "age", "lifetime", "color",
                "source_id", "uid", "viscosity", "density",
            )
        }
        | {
            f"whitewater_attribute_{suffix}"
            for suffix in ("velocity", "speed", "age", "lifetime", "type")
        }
    )
    assert diverging == expected

    # quality and domain_object are NOT in that list: their consumer reads the
    # scene too, so the field-level override makes both functions agree.
    for name in ("quality", "domain_object"):
        assert pending.setting_owner(scene, name) is scene_group
        assert pending._authoring_owner(scene_group, domain_group, name) is scene_group


def test_preset_scene_owned_settings_agree_with_setting_owner(monkeypatch):
    """``presets.schema`` restates the split for the pure (bpy-free) preset layer.

    ``presets/library.py`` must import without ``bpy``, so it cannot call
    ``setting_owner`` and falls back to ``SCENE_OWNED_SETTINGS`` when no
    ``field_target`` is supplied. That fallback is only safe while it agrees
    exactly with the owner function over the preset-eligible keys, which is what
    this pins — the third declaration is a layering artifact, not a third
    opinion.
    """
    _properties, pending, _bridge = _modules(monkeypatch)
    from fluidSim_warp_blender51.presets import schema

    scene, scene_group, _domain_group = _migrated_scene()

    assert set(schema.SETTING_SPECS) <= set(pending.SETTING_TRANSITION_REGISTRY)
    derived = {
        key
        for key in schema.SETTING_SPECS
        if pending.setting_owner(scene, key) is scene_group
    }
    assert derived == set(schema.SCENE_OWNED_SETTINGS)


def test_settings_groups_are_setting_owner_made_structural(monkeypatch):
    """Every registered field is declared on the group setting_owner names.

    The split turned the ownership declaration into two disjoint classes, so a
    field declared on the wrong class would raise on every read the declaration
    routes to it. Pinning the two against each other keeps them from drifting.
    """
    properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, _domain_group = _migrated_scene()
    scene_fields = _declared_fields(properties.FLUIDSIM_PG_scene)
    domain_fields = _declared_fields(properties.FLUIDSIM_PG_domain)

    assert not scene_fields & domain_fields
    for name in pending.SETTING_TRANSITION_REGISTRY:
        on_scene = pending.setting_owner(scene, name) is scene_group
        assert name in (scene_fields if on_scene else domain_fields), (name, on_scene)


def test_suppress_callbacks_is_nested_and_exception_safe(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)
    scene = SimpleNamespace()
    state = pending._interaction_state(scene)

    assert not state.suppress_callbacks
    with pending.suppress_callbacks(scene):
        assert state.suppress_callbacks
        with pending.suppress_callbacks(scene):
            assert state.suppress_callbacks
        assert state.suppress_callbacks
    assert not state.suppress_callbacks

    state.suppress_callbacks = True
    try:
        with pending.suppress_callbacks(scene):
            raise RuntimeError("sentinel")
    except RuntimeError as exc:
        assert str(exc) == "sentinel"
    assert state.suppress_callbacks
