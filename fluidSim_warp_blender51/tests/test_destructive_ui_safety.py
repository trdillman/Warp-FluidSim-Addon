from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "operators.py"


def _block(source: str, class_name: str) -> str:
    start = source.index(f"class {class_name}")
    end = source.find("\nclass ", start + 1)
    return source[start:] if end < 0 else source[start:end]


def test_production_restart_requires_path_specific_confirmation_and_revalidation():
    block = _block(OPERATORS.read_text(encoding="utf-8"), "FLUIDSIM_OT_bake_start")
    assert "expected_cache_path" in block
    assert "def invoke" in block
    assert "invoke_props_dialog" in block
    assert "Replace Existing Production Cache" in block
    assert "_resolved_cache_path" in block
    assert "changed after confirmation" in block


def test_free_cache_confirmation_displays_and_revalidates_the_resolved_path():
    block = _block(OPERATORS.read_text(encoding="utf-8"), "FLUIDSIM_OT_cache_free")
    assert "expected_cache_path" in block
    assert "invoke_props_dialog" in block
    assert "Free Entire Cache" in block
    assert "_resolved_cache_path" in block
    assert "changed after confirmation" in block


def test_orphan_removal_confirms_the_exact_current_orphan_inventory():
    block = _block(OPERATORS.read_text(encoding="utf-8"), "FLUIDSIM_OT_cache_remove_orphans")
    assert "expected_cache_path" in block
    assert "expected_orphans_json" in block
    assert "inspect_cache" in block
    assert "invoke_props_dialog" in block
    assert "orphan inventory changed" in block


def test_move_and_rename_dialogs_bind_to_the_source_path_reviewed_by_the_user():
    source = OPERATORS.read_text(encoding="utf-8")
    for class_name in ("FLUIDSIM_OT_cache_move", "FLUIDSIM_OT_cache_rename"):
        block = _block(source, class_name)
        assert "expected_source" in block
        assert "_resolved_cache_path" in block
        assert "changed after confirmation" in block
