from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_blender_effect_adapter_exposes_strict_snapshot_and_identity_seams():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    for token in (
        "effect_instance_from_pg",
        "scene_effect_instances",
        "effect_identity",
        "build_effect_candidate",
        "test_draft",
        "apply_stack",
        "revert_draft",
        "migrate_legacy_snippets",
        "TrustStore",
        "content_trust_digest",
        "stack_fingerprint",
    ):
        assert token in source


def test_scene_model_tracks_draft_tested_applied_and_trust_state():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    for token in (
        "draft_signature",
        "tested_signature",
        "applied_signature",
        "applied_inventory_json",
        "applied_snapshot_json",
        "trust_state",
        "blocking_reason",
        "runtime_error",
    ):
        assert token in source


def test_controller_snapshot_uses_matrix_and_stable_object_name_not_pointer():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    assert "matrix_world" in source
    assert "object_name" in source
    assert "as_pointer" not in source


def test_legacy_migration_disables_unknown_code_and_mentions_voxel_tile():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    assert "VOXEL_TILE" in source
    assert "UNTRUSTED_IMPORTED" in source
    assert "enabled=False" in source or '"enabled": False' in source


def test_root_registration_includes_effect_scene_model():
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "effects.register" in source
    assert "effects.unregister" in source


def test_admission_uses_effect_stack_and_fails_closed_before_cutover():
    admission = (ROOT / "blender" / "simulation_admission.py").read_text(encoding="utf-8")
    bridge = (ROOT / "blender" / "scene_bridge.py").read_text(encoding="utf-8")
    effects_source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    assert "effect_builder" in admission
    assert "snapshot.effects" in admission
    assert "build_candidate_effects" in admission
    assert "set_custom_effect_candidate" in admission
    assert "effect_builder=build_effect_stack" in bridge
    assert "return effects.build_effect_stack_for_admission(scene)" in bridge
    assert "def build_effect_stack_for_admission" in effects_source
    assert "production_blocking_reason" in effects_source.split(
        "def build_effect_stack_for_admission", 1
    )[1]


def test_legacy_snippets_are_migrated_before_v2_only_admission():
    admission = (ROOT / "blender" / "simulation_admission.py").read_text(encoding="utf-8")
    bridge = (ROOT / "blender" / "scene_bridge.py").read_text(encoding="utf-8")
    effects_source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    solver = (ROOT / "sim" / "solver.py").read_text(encoding="utf-8")
    frame_features = (ROOT / "sim" / "frame_features.py").read_text(encoding="utf-8")
    assert "set_custom_snippets" not in admission
    assert "snippet_builder" not in admission
    assert "set_custom_snippets" not in solver
    assert "SnippetManager" not in solver
    assert "set_legacy_snippets" not in frame_features
    assert "snippets.run" not in frame_features
    assert "migrate_legacy_snippets(scene)" in bridge
    assert "def _on_load_post" in effects_source
    assert "@persistent" in effects_source
    assert "register_handlers" in effects_source
    assert "effect_builder=build_effect_stack" in bridge
    migration = effects_source.split("def migrate_legacy_snippets", 1)[1].split(
        "def _migrate_loaded_scenes", 1
    )[0]
    assert "legacy_collection.clear" not in migration
    assert "originals preserved for recovery" in migration
    assert "migration_complete" in effects_source.split(
        "def build_effect_stack_for_admission", 1
    )[1].split("def trust_effect", 1)[0]


def test_scene_status_updates_match_instances_by_stable_id_not_collection_order():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    refresh_block = source.split("def refresh_stack_state", 1)[1].split(
        "def build_effect_candidate", 1
    )[0]
    apply_block = source.split("def apply_stack", 1)[1].split("def revert_draft", 1)[0]
    assert "instances_by_id" in refresh_block
    assert "instances_by_id" in apply_block
    assert "zip(stack.items, instances)" not in refresh_block
    assert "zip(stack.items, instances)" not in apply_block


def test_disabled_or_muted_effects_have_explicit_nonexecuting_status():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    refresh_block = source.split("def refresh_stack_state", 1)[1].split(
        "def build_effect_candidate", 1
    )[0]
    assert "instance.instance_id not in effective_ids" in refresh_block
    assert 'pg.status = "DISABLED"' in refresh_block


def test_effect_identity_recomputes_current_text_and_controller_snapshot_at_admission():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    identity = source.split("def effect_identity", 1)[1].split("def production_blocking_reason", 1)[0]
    assert "scene_effect_instances(scene)" in identity
    assert "execution_stack_fingerprint" in identity
    assert "stack.draft_signature" not in identity
    assert "current_signature != stack.applied_signature" in identity


def test_blender_diagnostics_are_mirrored_by_instance_not_shared_effect_id():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    mirror = source.split("def _mirror_issues", 1)[1].split("def test_draft", 1)[0]
    assert "by_instance" in mirror
    assert "issue.instance_id" in mirror
    assert "by_effect = {item.effect_id: item" not in mirror


def test_support_report_survives_malformed_embedded_effect_definitions():
    source = (ROOT / "blender" / "effects.py").read_text(encoding="utf-8")
    report = source.split("def effect_report_state", 1)[1].split(
        "def build_effect_stack_for_admission", 1
    )[0]
    assert '"definition_error"' in report
    assert "except (TypeError, ValueError, json.JSONDecodeError)" in report
