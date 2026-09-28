"""Every Blender acceptance script is run by the production gate or excused.

REVIEW-009: thirteen ``tests/blender_*.py`` scripts had drifted out of the gate;
five had been broken since 1.0.255 without anyone noticing. This pins the
inventory so a new script must either be wired into
``tools/run_production_gate.py`` or listed in its ``MANUAL_BLENDER_PROBES``
table with a reason.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "run_production_gate.py"
SPEC = importlib.util.spec_from_file_location("run_production_gate_coverage", GATE_PATH)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _scripts_launched_by_blender_calls():
    """``tests/blender_*.py`` literals passed straight to a ``_blender(...)`` call."""
    tree = ast.parse(GATE_PATH.read_text(encoding="utf-8"))
    launched = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_blender":
            for arg in ast.walk(node):
                if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and arg.value.startswith("tests/blender_")
                        and arg.value.endswith(".py")):  # skip f-string fragments
                    launched.add(arg.value)
    return launched


def _gate_run_scripts():
    return (
        {script for script, _ in gate.BACKGROUND_ACCEPTANCE}
        | {f"tests/blender_{cap}_acceptance.py" for cap, _ in gate.SOURCE_REPORT_ACCEPTANCE}
        | _scripts_launched_by_blender_calls()
    )


def test_every_blender_script_is_gated_or_a_documented_manual_probe():
    scripts = {f"tests/{path.name}" for path in (ROOT / "tests").glob("blender_*.py")}
    run = _gate_run_scripts()
    manual = set(gate.MANUAL_BLENDER_PROBES)
    assert not run & manual, f"both run and excused: {sorted(run & manual)}"
    assert scripts - run - manual == set(), (
        "wire these into tools/run_production_gate.py or add them to "
        f"MANUAL_BLENDER_PROBES with a reason: {sorted(scripts - run - manual)}")
    assert (run | manual) - scripts == set(), f"missing scripts: {sorted((run | manual) - scripts)}"


def test_manual_probes_give_a_reason():
    for script, reason in gate.MANUAL_BLENDER_PROBES.items():
        assert len(reason.split()) >= 4, script


def test_gate_markers_are_printed_by_their_scripts():
    """A marker the script never prints would fail every gate run."""
    common = (ROOT / "tests" / "_phase2_source_acceptance_common.py").read_text(encoding="utf-8")
    for script, marker in gate.BACKGROUND_ACCEPTANCE:
        assert marker in (ROOT / script).read_text(encoding="utf-8"), (script, marker)
    for capability, marker in gate.SOURCE_REPORT_ACCEPTANCE:
        stub = (ROOT / f"tests/blender_{capability}_acceptance.py").read_text(encoding="utf-8")
        assert f'main("{capability}")' in stub
        assert f'"{capability}": "{marker}"' in common


def test_every_release_blender_validator_is_gated():
    """tools/release/validate_*_blender.py drifted unrun until 2.0; pin them all."""
    scripts = {path.name for path in (ROOT / "tools" / "release").glob("validate_*_blender.py")}
    gated = {f"validate_{check}_blender.py" for check, _, _ in gate.RELEASE_BLENDER_VALIDATORS}
    assert scripts == gated


def test_release_blender_validator_markers_and_arguments_match_their_scripts():
    for check, inputs, marker in gate.RELEASE_BLENDER_VALIDATORS:
        path = ROOT / "tools" / "release" / f"validate_{check}_blender.py"
        source = path.read_text(encoding="utf-8")
        assert marker in source, (check, marker)
        command = gate._release_blender_validator_command("blender", check, inputs, "warp")
        assert "--python-exit-code" in command and "--factory-startup" in command
        for flag in (arg for arg in command[command.index("--") + 1:] if arg.startswith("--")):
            assert f'"{flag}"' in source, (check, flag)


def test_gated_scripts_are_syntax_checked():
    """The py_compile step covers every script the gate launches."""
    tree = ast.parse(GATE_PATH.read_text(encoding="utf-8"))
    compiled_source = next(
        ast.get_source_segment(GATE_PATH.read_text(encoding="utf-8"), node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_run"
        and node.args and getattr(node.args[0], "value", None) == "python syntax"
    )
    assert "BACKGROUND_ACCEPTANCE" in compiled_source
    assert "SOURCE_REPORT_ACCEPTANCE" in compiled_source
    assert "RELEASE_BLENDER_VALIDATORS" in compiled_source
    for script in _scripts_launched_by_blender_calls():
        assert script in compiled_source, script
