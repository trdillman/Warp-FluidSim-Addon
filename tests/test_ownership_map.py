"""Pin the Phase 0 settings ownership map.

``docs/ownership-map.md`` is the gate deliverable of
``docs/PROPOSAL-split-settings-propertygroup.md``: one row per annotated field
on ``FLUIDSIM_PG_settings``, with the datablock each end of the field's data
flow actually touches, resolved from the AST rather than from the ownership
declarations -- so the declarations can be audited against it.

These tests exist so the map cannot silently rot. They deliberately pin the
exact ``BROKEN`` and ``PUBLISHED_BROKEN`` field sets: those name live defects,
and both *fixing* one and *introducing* one must be a reviewed edit to this
file rather than a quiet change in a generated document.

The generator is driven as a subprocess on purpose. ``tools/ownership_map.py``
installs a stub ``bpy`` into ``sys.modules`` so it can import the shipped
declarations headlessly; importing it in-process would leak that stub into
every later test in the same session.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / "docs" / "ownership-map.md"
GENERATOR = REPO_ROOT / "tools" / "ownership_map.py"

# The full annotated surface of the shared PropertyGroup.
EXPECTED_FIELD_COUNT = 232

# UI writes a datablock no consumer reads: on a migrated scene these edits
# never reach the solver. This is the standing live bug that Phase 1 fixes;
# shrinking this set is the point of the exercise, so the pin must be updated
# deliberately when it does.
EXPECTED_BROKEN = frozenset({
    "height_back_mode", "height_base", "height_emission_mode",
    "height_emit_particles", "height_profile_image", "height_profile_scale",
    "height_velocity", "particle_radius_scale", "profiling_enabled",
    "profiling_sync", "surface_build_lowres_sdf", "surface_dilate_iterations",
    "surface_dilate_strength", "surface_erode_iterations",
    "surface_erode_strength", "surface_filter_iterations",
    "surface_filter_strength", "surface_filter_type",
    "surface_leaf_capacity_scale", "surface_max_dense_dim",
    "surface_resolution_mode", "surface_resolution_multiplier",
    "surface_smooth_iterations", "surface_smooth_strength",
    "surface_smooth_type", "surface_splat_radius", "surface_update_cadence",
    "surface_voxel_size", "ww_bubble_buoyancy", "ww_bubble_drag",
    "ww_bubble_life_multiplier", "ww_churn_decay", "ww_churn_gain",
    "ww_churn_threshold", "ww_emission_rate", "ww_foam_advection",
    "ww_foam_band", "ww_foam_density_influence", "ww_foam_life_multiplier",
    "ww_foam_preservation", "ww_life_max", "ww_life_min", "ww_min_speed",
    "ww_spray_drag", "ww_spray_gravity_scale", "ww_spray_life_multiplier",
    "ww_spray_min_speed", "ww_surface_band", "ww_type_hysteresis",
})

# The telemetry-direction counterpart: the runtime publishes the value to one
# datablock and the panel displays the other, so the message never appears on
# a migrated scene.
EXPECTED_PUBLISHED_BROKEN = frozenset({"height_error"})


def _parse_map_rows():
    """Rows of the ``## The map`` table only, as field -> {column: value}.

    Scoped to that one section because the document also renders a
    ``## Fields with no resolved reader`` table whose rows share the leading
    backticked-field-name shape.
    """
    if not MAP_PATH.exists():  # pragma: no cover - guarded by its own test
        pytest.fail(f"{MAP_PATH} is missing; run python tools/ownership_map.py")
    text = MAP_PATH.read_text(encoding="utf-8")
    section = text.split("\n## The map\n", 1)
    assert len(section) == 2, "docs/ownership-map.md has no '## The map' section"
    body = section[1].split("\n## ", 1)[0]

    header = None
    rows = {}
    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue  # the markdown alignment row
        assert len(cells) == len(header), f"ragged row: {line}"
        field = cells[0].strip("`")
        assert field not in rows, f"duplicate row for {field}"
        rows[field] = dict(zip(header, cells))
    return rows


def _verdict(row):
    return row["verdict"].replace("*", "")


def _fields_with(rows, verdict):
    return {name for name, row in rows.items() if _verdict(row) == verdict}


def test_map_is_current():
    """The checked-in map matches what the generator produces right now."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "docs/ownership-map.md is stale; regenerate with "
        f"`python tools/ownership_map.py`.\n{result.stdout}{result.stderr}"
    )


def test_map_covers_every_annotated_field_exactly_once():
    """The gate: one row per field, no field counted twice, none missing."""
    rows = _parse_map_rows()
    assert len(rows) == EXPECTED_FIELD_COUNT


def test_every_field_declares_exactly_one_scope():
    """No field may be left without a resolved effective/authoring owner.

    This is the Phase 0 gate condition from the proposal: no later phase
    starts until every field has exactly one declared scope.
    """
    rows = _parse_map_rows()
    undeclared = {
        name: row["declared eff/auth"]
        for name, row in rows.items()
        if "?" in row["declared eff/auth"]
    }
    assert not undeclared, f"fields with no declared scope: {undeclared}"

    malformed = {
        name: row["declared eff/auth"]
        for name, row in rows.items()
        if not re.fullmatch(r"(scene|domain)/(scene|domain)",
                            row["declared eff/auth"])
    }
    assert not malformed, f"unparseable declared scope: {malformed}"


def test_broken_field_set_is_pinned():
    """The authoring-direction live bug: UI writes where no consumer reads."""
    rows = _parse_map_rows()
    broken = _fields_with(rows, "BROKEN")
    assert broken == set(EXPECTED_BROKEN), {
        "newly broken": sorted(broken - EXPECTED_BROKEN),
        "no longer broken": sorted(EXPECTED_BROKEN - broken),
    }


def test_published_broken_field_set_is_pinned():
    """The telemetry-direction counterpart: runtime writes, panel reads other."""
    rows = _parse_map_rows()
    published_broken = _fields_with(rows, "PUBLISHED_BROKEN")
    assert published_broken == set(EXPECTED_PUBLISHED_BROKEN), {
        "newly broken": sorted(published_broken - EXPECTED_PUBLISHED_BROKEN),
        "no longer broken": sorted(EXPECTED_PUBLISHED_BROKEN - published_broken),
    }


def test_verdict_summary_agrees_with_the_table():
    """The summary table at the top must count the same rows as the body."""
    text = MAP_PATH.read_text(encoding="utf-8")
    body = text.split("\n## Verdicts\n", 1)[1].split("\n## ", 1)[0]
    summary = {}
    for line in body.splitlines():
        match = re.match(r"^\| ([A-Z_]+) \| (\d+) \|", line)
        if match:
            summary[match.group(1)] = int(match.group(2))
    assert summary, "no verdict summary table found"

    rows = _parse_map_rows()
    counted = {}
    for row in rows.values():
        counted[_verdict(row)] = counted.get(_verdict(row), 0) + 1
    assert summary == counted
    assert sum(summary.values()) == EXPECTED_FIELD_COUNT


def test_mirror_redirect_inventory_is_recorded():
    """The map must state which mirrored panels lack owner_redirect.

    Those panels are why a field can be written to the scene group from two
    different tabs and still miss a domain-group consumer; the split cannot be
    reasoned about without the list.
    """
    text = MAP_PATH.read_text(encoding="utf-8")
    assert "## Mirror redirection" in text
    assert re.search(r"`owner_redirect=True` \(\d+ source panels\)", text)
    assert re.search(r"Mirrored \*\*without\*\* redirect \(\d+\)", text)
