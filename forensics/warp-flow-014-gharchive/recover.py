from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

TARGET_REPO = "trdillman/liquid_warp_blender51_port"
TARGET_ISSUE = 1
EXPECTED_SIZE = 221_174
EXPECTED_SHA256 = "9e14e030a74d9774bc5d8df455ea1f855b9045bd06d619cdfe04334ed1db511e"
HOURS = (
    "2026-08-22-22",
    "2026-08-22-23",
    "2026-08-23-0",
    "2026-08-23-1",
    "2026-08-23-2",
    "2026-08-23-3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_base64(body: str) -> str:
    candidates: list[str] = []
    for value in re.findall(
        r"```(?:[A-Za-z0-9_-]+)?\s*([A-Za-z0-9+/=\s]{1000,})```",
        body,
        flags=re.S,
    ):
        candidates.append(re.sub(r"\s+", "", value))
    lines = [
        line.strip()
        for line in body.splitlines()
        if len(line.strip()) >= 80
        and re.fullmatch(r"[A-Za-z0-9+/=]+", line.strip())
    ]
    if lines:
        candidates.append("".join(lines))
    compact = re.sub(r"\s+", "", body)
    if len(compact) >= 1000 and re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        candidates.append(compact)
    return max(candidates, key=len, default="")


def main() -> int:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "evidence").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comments").mkdir(exist_ok=True)
    repo_events: list[dict] = []
    downloads: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="warp-flow-gharchive-") as temp_name:
        temp = Path(temp_name)
        for stamp in HOURS:
            url = f"https://data.gharchive.org/{stamp}.json.gz"
            archive = temp / f"{stamp}.json.gz"
            print(f"downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "warp-flow-forensics/1"})
            with urllib.request.urlopen(request, timeout=600) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            downloads.append(
                {
                    "hour": stamp,
                    "size": archive.stat().st_size,
                    "sha256": sha256_file(archive),
                }
            )
            with gzip.open(archive, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if event.get("repo", {}).get("name") != TARGET_REPO:
                        continue
                    event["_gharchive_hour"] = stamp
                    repo_events.append(event)

    issue_events: list[dict] = []
    comments: list[dict] = []
    for event in repo_events:
        payload = event.get("payload") or {}
        issue = payload.get("issue") or {}
        if issue.get("number") != TARGET_ISSUE:
            continue
        issue_events.append(event)
        if event.get("type") != "IssueCommentEvent":
            continue
        comment = payload.get("comment") or {}
        body = comment.get("body") or ""
        comments.append(
            {
                "event_id": event.get("id"),
                "created_at": event.get("created_at"),
                "hour": event.get("_gharchive_hour"),
                "action": payload.get("action"),
                "comment_id": comment.get("id"),
                "comment_url": comment.get("html_url"),
                "author": (comment.get("user") or {}).get("login"),
                "body_length": len(body),
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "body": body,
            }
        )

    ordered = sorted(comments, key=lambda item: (item["created_at"] or "", item["comment_id"] or 0))
    for index, record in enumerate(ordered):
        name = f"{index:03d}_{record['created_at'] or 'unknown'}_{record['comment_id'] or 'unknown'}_{record['action'] or 'unknown'}.txt".replace(":", "-")
        (output / "comments" / name).write_text(record["body"], encoding="utf-8")

    with (output / "repo-events.jsonl").open("w", encoding="utf-8") as handle:
        for event in repo_events:
            handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
    with (output / "issue-1-events.jsonl").open("w", encoding="utf-8") as handle:
        for event in issue_events:
            handle.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")

    segments: list[str] = []
    segment_records: list[dict] = []
    for record in ordered:
        segment = extract_base64(record["body"])
        if not segment:
            continue
        segments.append(segment)
        segment_records.append(
            {
                key: record[key]
                for key in ("created_at", "action", "comment_id", "body_length", "body_sha256")
            }
            | {"base64_length": len(segment)}
        )

    attempts: list[dict] = []
    if segments:
        for label, values in (("concatenated", ["".join(segments)]), ("independent", segments)):
            try:
                if label == "concatenated":
                    value = values[0]
                    payload = base64.b64decode(value + "=" * ((-len(value)) % 4), validate=False)
                else:
                    payload = b"".join(
                        base64.b64decode(value + "=" * ((-len(value)) % 4), validate=False)
                        for value in values
                    )
                target = output / f"reconstructed-{label}.bin"
                target.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                attempts.append(
                    {
                        "strategy": label,
                        "size": len(payload),
                        "sha256": digest,
                        "matches_expected_size": len(payload) == EXPECTED_SIZE,
                        "matches_expected_sha256": digest == EXPECTED_SHA256,
                        "path": target.name,
                    }
                )
            except Exception as exc:
                attempts.append({"strategy": label, "error": repr(exc)})

    actions: dict[str, int] = {}
    for record in comments:
        action = record["action"] or "unknown"
        actions[action] = actions.get(action, 0) + 1
    summary = {
        "downloads": downloads,
        "repository_event_count": len(repo_events),
        "issue_1_event_count": len(issue_events),
        "issue_comment_event_count": len(comments),
        "comment_actions": actions,
        "comments": [{key: value for key, value in record.items() if key != "body"} for record in ordered],
        "base64_segments": segment_records,
        "reconstruction_attempts": attempts,
        "expected_archive": {"size": EXPECTED_SIZE, "sha256": EXPECTED_SHA256},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SUMMARY.md").write_text(
        "# Warp Flow GH Archive forensic result\n\n"
        f"- Repository events: {len(repo_events)}\n"
        f"- Issue #1 events: {len(issue_events)}\n"
        f"- Issue-comment events: {len(comments)}\n"
        f"- Extracted Base64 segments: {len(segments)}\n"
        f"- Expected archive: {EXPECTED_SIZE:,} bytes, `{EXPECTED_SHA256}`\n\n"
        "## Reconstruction attempts\n\n```json\n"
        + json.dumps(attempts, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(value for value in output.rglob("*") if value.is_file() and value.name != "SHA256SUMS.txt"):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
