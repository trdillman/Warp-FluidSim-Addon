#!/usr/bin/env python3
"""Fail closed when policy-drift reports lose their runtime authority."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "production_gate_manifest.json"
# A commit that only records gate evidence (this report, its promoted
# metrics, and the demo artifacts the gate regenerates) does not invalidate
# the gate's proof. Anything else changing after the proof commit does.
ALLOWED_EVIDENCE_PATTERNS = (
    "POLICY_DRIFT_REPORT.md",
    "demos/POLICY_DRIFT_REPORT.md",
    "fluidSim_warp_blender51/POLICY_DRIFT_REPORT.md",
    "demos/ANALYSIS.md",
    "fluidSim_warp_blender51/docs/technical_report.md",
    "demos/analysis_report.json",
    "demos/*.blend",
    "demos/*_preview.png",
)
REQUIRED_BINDINGS = (
    "generated_at",
    "git_head",
    "audit_unit",
    "gate_run_at",
    "gate_verdict",
    "demo_artifact",
    "demo_artifact_sha256",
)
METRIC_KEYS = (
    "front_speed_over_c0",
    "column_height_end",
    "crown_sqrt_t_corr",
    "max_crown_height",
)
ACTIVE_CARD_STATES = {"ready-for-agent", "in-progress", "active", "blocked"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _mapping(lines: list[str]) -> dict[str, str]:
    values = {}
    for line in lines:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*?)\s*$", line)
        if match:
            values[match.group(1)] = _unquote(match.group(2))
    return values


def _parse_report(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: missing frontmatter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ValueError(f"{path}: unterminated frontmatter") from exc
    footer_start = next(
        (i for i, line in enumerate(lines)
         if line.strip() == "## Machine-readable audit footer"),
        None,
    )
    if footer_start is None:
        raise ValueError(f"{path}: missing machine-readable footer")
    footer = _mapping(lines[footer_start + 1:])
    active = []
    for i, line in enumerate(lines[footer_start + 1:], footer_start + 1):
        if line.strip() == "active_backlog_tasks:":
            for item in lines[i + 1:]:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:", item):
                    break
                match = re.match(r"^\s*-\s*(.*?)\s*$", item)
                if match:
                    active.append(_unquote(match.group(1)))
            break
    return _mapping(lines[1:end]), footer, active


def _same_path(left: str, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(str(right)))


def _is_evidence_path(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in ALLOWED_EVIDENCE_PATTERNS)


def _non_evidence_drift(proof_head: str, current_head: str) -> list[str]:
    """Commits between the gate's proof HEAD and current HEAD.

    A commit that only touches gate evidence (the reports, promoted docs,
    and regenerated demo artifacts) does not need a fresh gate run to stay
    truthful; a report can never assert its own committed hash, since a
    commit's hash depends on the tree that contains that assertion.
    Anything else in that range means the proof is stale.
    """
    if proof_head == current_head:
        return []
    try:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", proof_head, current_head],
            cwd=ROOT, capture_output=True).returncode == 0
        if not ancestor:
            return ["gate manifest HEAD is not an ancestor of current HEAD"]
        commits = subprocess.check_output(
            ["git", "rev-list", f"{proof_head}..{current_head}"],
            cwd=ROOT, text=True).split()
        errors = []
        for sha in commits:
            parents = subprocess.check_output(
                ["git", "rev-parse", f"{sha}^@"], cwd=ROOT, text=True).split()
            if len(parents) > 1:
                errors.append(
                    f"commit {sha} after the gate proof is a merge; "
                    "re-run the gate")
                continue
            changed = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
                cwd=ROOT, text=True).splitlines()
            errors.extend(
                f"commit {sha} changed non-evidence file {path} after the "
                "gate proof; re-run the gate"
                for path in changed if path and not _is_evidence_path(path))
        return errors
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"cannot verify commit history since the gate proof: {exc}"]


def _card_state(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## State\s*$.*?^([^\r\n]+)\s*$", text,
                      re.MULTILINE | re.DOTALL)
    return match.group(1).strip().lower() if match else "missing"


def _active_cards(unit: Path) -> list[str]:
    task_dir = unit / "backlog" / "tasks"
    if not task_dir.is_dir():
        return []
    return sorted(
        f"backlog/tasks/{path.name}"
        for path in task_dir.glob("TASK-POLICY-DRIFT-*.md")
        if _card_state(path) in ACTIVE_CARD_STATES
    )


def _check_doc_metrics(path: Path, artifact: dict) -> list[str]:
    metrics = artifact.get("metrics", {})
    values = {
        key: metrics["dam_break" if key in {"front_speed_over_c0", "column_height_end"}
                      else "crown_splash"][key]
        for key in METRIC_KEYS
    }
    text = path.read_text(encoding="utf-8")
    errors = []
    for key, value in values.items():
        token = f"{key}={float(value):.10f}"
        if token not in text:
            errors.append(f"{path}: missing promoted metric {token}")
    return errors


def validate_report_authority(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    errors = []
    if not manifest_path.is_file():
        return [f"missing successful full-gate manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid gate manifest: {exc}"]
    if manifest.get("gate_verdict") != "FULL-PRODUCTION-GATE-PASS":
        errors.append("manifest is not a full-production-gate pass")
    if manifest.get("include_gui") is not True:
        errors.append("manifest does not prove GUI-inclusive acceptance")
    try:
        current_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cannot resolve current HEAD: {exc}")
        current_head = ""
    proof_head = manifest.get("git_head")
    if not isinstance(proof_head, str) or not proof_head:
        errors.append("gate manifest is missing git_head")
    elif current_head:
        errors.extend(_non_evidence_drift(proof_head, current_head))
    artifact_rel = manifest.get("demo_artifact")
    artifact = ROOT / artifact_rel if isinstance(artifact_rel, str) else None
    if artifact is None or not artifact.is_file():
        errors.append("gate manifest demo artifact is missing")
        return errors
    artifact_hash = _sha256(artifact)
    if manifest.get("demo_artifact_sha256") != artifact_hash:
        errors.append("gate manifest demo artifact hash is stale")
    try:
        generated = json.loads(artifact.read_text(encoding="utf-8"))
        checked_in_path = ROOT / "demos" / "analysis_report.json"
        checked_in = json.loads(checked_in_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"demo evidence is unreadable: {exc}")
        generated = checked_in = {}
    if generated != checked_in:
        errors.append("checked-in demo JSON does not match the gate artifact")

    for report_path in (ROOT / "POLICY_DRIFT_REPORT.md",
                        ROOT / "demos" / "POLICY_DRIFT_REPORT.md"):
        try:
            front, footer, active = _parse_report(report_path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        for key in REQUIRED_BINDINGS + ("status",):
            if front.get(key) != footer.get(key):
                errors.append(f"{report_path}: frontmatter/footer {key} mismatch")
        expected_unit = report_path.parent
        if not _same_path(footer.get("audit_unit", ""), expected_unit):
            errors.append(f"{report_path}: audit_unit is misbound")
        if footer.get("git_head") != manifest.get("git_head"):
            errors.append(f"{report_path}: report HEAD is stale")
        if footer.get("gate_run_at") != manifest.get("gate_run_at"):
            errors.append(f"{report_path}: gate timestamp is stale")
        if footer.get("generated_at") != manifest.get("gate_run_at"):
            errors.append(f"{report_path}: generated timestamp is stale")
        if footer.get("gate_verdict") != manifest.get("gate_verdict"):
            errors.append(f"{report_path}: gate verdict is not authoritative")
        if footer.get("demo_artifact_sha256") != artifact_hash:
            errors.append(f"{report_path}: artifact provenance is stale")
        expected_active = _active_cards(expected_unit)
        if sorted(active) != expected_active:
            errors.append(f"{report_path}: active card list is stale")

    for doc in (ROOT / "demos" / "ANALYSIS.md",
                ROOT / "fluidSim_warp_blender51" / "docs" / "technical_report.md"):
        if doc.is_file():
            errors.extend(_check_doc_metrics(doc, generated))
        else:
            errors.append(f"missing promoted evidence document: {doc}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    errors = validate_report_authority(args.manifest)
    if errors:
        print("POLICY-DRIFT-REPORT-VALIDATION-FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("POLICY-DRIFT-REPORT-VALIDATION-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
