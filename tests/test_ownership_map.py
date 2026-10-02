"""Pin the Phase 0 settings ownership map.

``docs/ownership-map.md`` is the gate deliverable of
``docs/PROPOSAL-split-settings-propertygroup.md``: one row per annotated field
on ``FLUIDSIM_PG_scene`` and ``FLUIDSIM_PG_domain``, with the datablock each
end of the field's data
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

# The full annotated surface of the shared PropertyGroup. 2.0 removed the six
# dead snippet-library/shortcut UI fields (233 -> 227), the pre-1.1
# ``schema_version`` stamp (227 -> 226) and the legacy ``snippets`` store (226 -> 225),
# then added the ``stat_native_solver_message`` readout (225 -> 226, REVIEW-016).
EXPECTED_FIELD_COUNT = 226

# UI writes a datablock no consumer reads: on a migrated scene those edits
# never reach the solver. Phase 1 emptied this set by repointing the contested
# panels at the domain owner. It stays pinned in the *empty* direction: any
# field that reappears here is a newly introduced live bug, and un-emptying it
# must be a reviewed edit to this file rather than a quiet change in a
# generated document.
EXPECTED_BROKEN = frozenset()

# The telemetry-direction counterpart: the runtime publishes the value to one
# datablock and the panel displays the other, so the message never appears on
# a migrated scene. Emptied by the Phase 1 `height_error` fix.
EXPECTED_PUBLISHED_BROKEN = frozenset()


# Fields whose write and read do meet, but where a second path exists. This is
# the residual hazard class and the one a regression actually lands in: when a
# panel drifts back to `scene.fluidsim` while a *sibling* panel still writes the
# domain group, the field goes OK -> OK_MULTI rather than OK -> BROKEN, so the
# empty BROKEN pin above would not notice.
#
# Phase 2 step 1 repointed the six two-path fields that were the pre-Phase-1
# shape -- the UI wrote the domain group while a consumer still read the scene
# group. The split emptied the set: `domain_object`'s second path was the
# Physics mirror rewriting `scene.fluidsim` to the domain group, and the domain
# group no longer declares the field, so the mirror now draws the scene group's.
#
# The pin now works in the *shrunk* direction: any field reappearing here is a
# consumer drifting back to `scene.fluidsim`, which the empty BROKEN pin above
# cannot catch.
EXPECTED_OK_MULTI = frozenset()

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
    before = MAP_PATH.read_bytes()
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "docs/ownership-map.md is stale; regenerate with "
        f"`python tools/ownership_map.py`.\n{result.stdout}{result.stderr}"
    )
    assert MAP_PATH.read_bytes() == before


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


def test_ok_multi_field_set_is_pinned():
    """Pin the residual two-path set, which is where a regression lands.

    A panel drifting back to the scene group while a sibling panel still writes
    the domain group produces OK_MULTI, not BROKEN, so this is the pin that
    catches it. Regenerating the map alone must not be able to grow this set.
    """
    rows = _parse_map_rows()
    ok_multi = _fields_with(rows, "OK_MULTI")
    assert ok_multi == set(EXPECTED_OK_MULTI), {
        "newly two-path": sorted(ok_multi - EXPECTED_OK_MULTI),
        "no longer two-path": sorted(EXPECTED_OK_MULTI - ok_multi),
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
    # The legend lists every verdict the analyzer can emit, including the ones
    # now at zero, so compare only the rows the table actually contains -- and
    # assert the zeros really are absent from it.
    assert {k: v for k, v in summary.items() if v} == counted
    assert not {k for k, v in summary.items() if not v} & set(counted)
    assert sum(summary.values()) == EXPECTED_FIELD_COUNT


def test_fault_verdicts_are_still_reachable():
    """Guard the empty pins above against a classifier that stopped emitting.

    ``EXPECTED_BROKEN`` and ``EXPECTED_PUBLISHED_BROKEN`` are both empty after
    Phase 1, so they would also pass if ``classify`` lost the ability to report
    those verdicts at all. Assert the generator still documents both, so the
    empty pins keep meaning "no field is in this state" rather than "the state
    no longer exists".
    """
    lines = MAP_PATH.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Verdicts")
    legend = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        legend.append(line)
    legend = " ".join(legend)
    for verdict in ("BROKEN", "PUBLISHED_BROKEN"):
        assert verdict in legend, (
            f"{verdict} is no longer described in the map's verdict legend; "
            "the empty pin above has stopped meaning anything."
        )


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


def test_stage_helper_writes_count_as_ui_writes():
    """The Bake/Setup/Simulate/Look stages draw through ``_prop`` and must be seen.

    ``_prop(layout, scene, "field")`` draws ``setting_owner(scene, "field")``.
    The generator binds that write to the field's declared effective owner;
    before it did, removing the Scene-tab cache panels left six cache fields
    looking as if no panel could edit them.
    """
    rows = _parse_map_rows()
    for field in (
        "cache_path", "cache_compress", "cache_use_custom_range",
        "cache_frame_start", "cache_frame_end", "checkpoint_interval",
    ):
        assert rows[field]["UI writes"] == "scene", field
    assert rows["flip_ratio"]["UI writes"] == "domain"
    assert rows["export_attributes"]["UI writes"] == "domain"
