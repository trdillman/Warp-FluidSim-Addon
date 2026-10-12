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


# LIVE settings that reach the solver through a path other than the hot-field
# projection: gravity through _apply_gravity, the growth policy through
# _request_particle_capacity.
_LIVE_APPLIED_ELSEWHERE = frozenset({
    "gravity", "use_scene_gravity",
    "auto_grow_particles", "particle_growth_failure_policy",
})


def test_every_tiered_setting_reaches_its_apply_or_signature_set(monkeypatch):
    """Review items B04/B05: a tier classification is inert without wiring.

    surface_tension_* and species_mixing_rate were classified LIVE but never
    projected into the running solver, and surface_tension_enabled /
    species_enabled were classified RESET but absent from the reset
    signature, so editing them did nothing and flagged nothing.
    """
    _properties, pending, _bridge = _modules(monkeypatch)

    assert sorted(pending._LIVE_REGISTRY_FIELDS - pending._LIVE_SETTING_FIELDS
                  - _LIVE_APPLIED_ELSEWHERE) == []
    assert sorted(pending._RESET_REGISTRY_FIELDS - pending._RESET_SCENE_FIELDS) == []
    assert sorted(pending._REINITIALIZE_REGISTRY_FIELDS
                  - pending._REINIT_SETTING_FIELDS) == []
    assert sorted(pending._SURFACE_REGISTRY_FIELDS
                  - pending._SURFACE_SETTING_FIELDS) == []


def test_live_surface_tension_and_mixing_edits_reach_the_running_solver(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)
    from fluidSim_warp_blender51.blender import solver_settings as ss
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    defaults = SolverSettings()
    values = {src: getattr(defaults, ss.SETTING_ALIASES.get(src, src))
              for src in ss.DIRECT_SETTING_FIELDS}
    for target, (src, enum_type) in ss.enum_setting_fields().items():
        values[src] = enum_type(getattr(defaults, ss.SETTING_ALIASES.get(target, target))).name
    values.update(gravity=(0.0, 0.0, -9.81), use_scene_gravity=False)
    domain = SimpleNamespace(**values)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(quality="PREVIEW",
                                 domain_object=SimpleNamespace(fluidsim_domain=domain)),
        use_gravity=True, gravity=(0.0, 0.0, -9.81))
    solver = SimpleNamespace(settings=SolverSettings())
    sess = SimpleNamespace(owns_scene=lambda _scene: True, solver=solver)

    domain.surface_tension_coefficient = 0.5
    domain.surface_tension_safety = 0.2
    domain.surface_tension_band_voxels = 2.5
    domain.species_mixing_rate = 30.0
    assert pending._apply_realtime_solver_settings(scene, sess)

    assert solver.settings.surface_tension_coefficient == 0.5
    assert solver.settings.surface_tension_safety == 0.2
    assert solver.settings.surface_tension_band_voxels == 2.5
    assert solver.settings.species_mixing_rate == 30.0


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
    root = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
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
    """A scene with a designated domain, where the two owners are distinct.

    ``active_domain_settings`` returns ``obj.fluidsim_domain`` only when a
    domain object is designated.
    """
    domain_group = SimpleNamespace(_label="domain")
    domain_object = SimpleNamespace(fluidsim_domain=domain_group)
    scene_group = SimpleNamespace(_label="scene", domain_object=domain_object)
    return SimpleNamespace(fluidsim=scene_group), scene_group, domain_group


def test_declaring_group_is_the_class_that_declares_the_field(monkeypatch):
    """The class that declares a setting owns it, for every setting.

    ``FLUIDSIM_PG_scene`` and ``FLUIDSIM_PG_domain`` are disjoint, so there is
    one answer to "which group holds this field". A group asked for a field it
    does not declare raises, so a lookup that named the wrong group would
    strand every read and write that followed it. Detection samples through the
    same rule: reading the wrong group gives ``None`` twice and an edit never
    shows as pending (defect #5).
    """
    properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, domain_group = _migrated_scene()
    scene_fields = _declared_fields(properties.FLUIDSIM_PG_scene)
    domain_fields = _declared_fields(properties.FLUIDSIM_PG_domain)

    assert scene_fields & domain_fields == set()
    assert "domain_object" in scene_fields
    assert "domain_object" not in domain_fields
    for name in sorted(scene_fields | domain_fields):
        expected = scene_group if name in scene_fields else domain_group
        assert pending.declaring_group(scene, name) is expected, name
        assert pending._declaring_group(scene_group, domain_group, name) is expected, name
    # The lifecycle registry covers exactly the declared fields, so it names a
    # lifecycle for every field the rule can be asked about.
    assert set(pending.SETTING_TRANSITION_REGISTRY) == scene_fields | domain_fields


def test_declaring_group_does_not_follow_the_lifecycle_tier(monkeypatch):
    """Ownership answers "which datablock holds it", the tier "what changing it needs".

    The two are orthogonal, and these fields are why the tier cannot stand in
    for the declaring class:

    * ``quality`` and ``domain_object`` are REINITIALIZE, which is mostly a
      domain tier, yet the scene group declares them. ``domain_object`` names
      which object is the domain, and ``scene_quality_fidelity()`` reads
      ``scene.fluidsim.quality``.
    * ``height_error`` is PRESENTATION, which is mostly a scene tier, yet the
      domain group declares it: ``scene_inputs.validate_height_source`` writes it
      there.
    """
    _properties, pending, _bridge = _modules(monkeypatch)
    scene, scene_group, domain_group = _migrated_scene()

    for name in ("quality", "domain_object"):
        assert pending.SETTING_TRANSITION_REGISTRY[name] is (
            pending.SettingTier.REINITIALIZE), name
        assert pending.declaring_group(scene, name) is scene_group, name
    assert pending.SETTING_TRANSITION_REGISTRY["height_error"] is (
        pending.SettingTier.PRESENTATION)
    assert pending.declaring_group(scene, "height_error") is domain_group

    # One representative per kind, through the public function.
    for name in ("liquid_view_mode", "cache_compress", "live", "ui_mode",
                 "checkpoint_interval", "stat_particles"):
        assert pending.declaring_group(scene, name) is scene_group, name
    for name in ("viscosity_enabled", "surface_enabled", "whitewater_enabled",
                 "height_enabled", "voxel_size", "export_attributes"):
        assert pending.declaring_group(scene, name) is domain_group, name

    # An undeclared name is not scene-owned. It falls to the domain group, where
    # reading or writing it raises, so a typo cannot quietly write the scene.
    assert pending.declaring_group(scene, "not_a_registered_field") is domain_group


def test_declaring_group_without_a_designated_domain_has_no_domain_group(monkeypatch):
    """No domain object means no domain group: ``None``, never ``scene.fluidsim``."""
    _properties, pending, _bridge = _modules(monkeypatch)
    scene_group = SimpleNamespace(domain_object=None)
    scene = SimpleNamespace(fluidsim=scene_group)

    assert pending.declaring_group(scene, "voxel_size") is None
    assert pending.declaring_group(scene, "cache_path") is scene_group


def test_preset_scene_owned_settings_are_the_scene_class_field_set(monkeypatch):
    """``presets.schema`` holds the one bpy-free copy of the scene class's fields.

    ``presets/library.py`` must import without ``bpy``, so it cannot read
    ``FLUIDSIM_PG_scene``. ``SCENE_OWNED_SETTINGS`` stands in for it, and this
    pins it equal to the class, so it is a copy and not a second opinion. Of the
    preset-eligible keys, exactly the three output and bake settings are
    scene-owned.
    """
    properties, pending, _bridge = _modules(monkeypatch)
    from fluidSim_warp_blender51.presets import schema

    assert schema.SCENE_OWNED_SETTINGS == _declared_fields(
        properties.FLUIDSIM_PG_scene)
    assert set(schema.SETTING_SPECS) <= set(pending.SETTING_TRANSITION_REGISTRY)
    assert {key for key in schema.SETTING_SPECS
            if key in schema.SCENE_OWNED_SETTINGS} == {
        "whitewater_output_mode", "cache_compress", "checkpoint_interval"}

    scene, scene_group, domain_group = _migrated_scene()
    for key in schema.SETTING_SPECS:
        expected = scene_group if key in schema.SCENE_OWNED_SETTINGS else domain_group
        assert pending.declaring_group(scene, key) is expected, key


def test_pending_detection_reads_every_signature_field_from_its_declaring_group(
        monkeypatch):
    """Defect #5 regression: a domain-group edit of any signature field is seen.

    height_*, *_attribute_* and profiling_* are declared only on
    ``FLUIDSIM_PG_domain``. Detection used to sample them on ``scene.fluidsim``,
    which lacks them, so the reset and reinitialize signatures held ``None`` and
    editing any of those 35 fields never flagged a pending Reset.
    """
    properties, pending, _bridge = _modules(monkeypatch)
    monkeypatch.setattr(pending, "_configured_device", lambda: "cuda:0")
    scene, scene_group, domain_group = _migrated_scene()
    # An ID signs by name; a nameless double would sign by repr and drag every
    # domain-group edit into the domain_object entry.
    scene_group.domain_object.name = "Domain"
    scene_fields = _declared_fields(properties.FLUIDSIM_PG_scene)
    domain_fields = _declared_fields(properties.FLUIDSIM_PG_domain)
    signature_fields = sorted(
        (set(pending._RESET_SCENE_FIELDS) | set(pending._REINIT_SETTING_FIELDS))
        - {"domain_object"}
    )
    # Model the split: each group carries only the fields its class declares.
    for index, name in enumerate(signature_fields):
        assert (name in scene_fields) != (name in domain_fields), name
        group = scene_group if name in scene_fields else domain_group
        setattr(group, name, f"applied-{index}")

    reset, reinit = pending.current_interaction_signatures(scene)
    unread = sorted(name for name, value in reset + reinit
                    if name in signature_fields and value is None)
    assert unread == []

    for name in signature_fields:
        group = scene_group if name in scene_fields else domain_group
        previous = getattr(group, name)
        setattr(group, name, "edited")
        try:
            new_reset, new_reinit = pending.current_interaction_signatures(scene)
        finally:
            setattr(group, name, previous)
        changed = (set(pending._signature_changes(reset, new_reset))
                   | set(pending._signature_changes(reinit, new_reinit)))
        assert changed == {name}, name


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
