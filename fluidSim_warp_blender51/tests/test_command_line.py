from __future__ import annotations

import json
from pathlib import Path

import pytest
from fluidsim_testpkg.blender.command_line import (
    EXIT_ARGUMENTS,
    CliOptions,
    parse_args,
    write_json_atomic,
)


def test_parse_complete_command(tmp_path: Path):
    cache = tmp_path / "cache"
    report = tmp_path / "report.json"
    options = parse_args([
        "--scene", "Scene", "--start", "1", "--end", "20",
        "--cache", str(cache), "--device", "cuda:0",
        "--resume", "auto", "--checkpoint-interval", "2",
        "--report-json", str(report), "--save-after-bake",
    ])
    assert options == CliOptions(
        scene="Scene",
        frame_start=1,
        frame_end=20,
        cache=cache.resolve(),
        device="cuda:0",
        resume="auto",
        checkpoint_interval=2,
        save_after_bake=True,
        report_json=report.resolve(),
    )


def test_end_before_start_is_rejected(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        parse_args([
            "--start", "10", "--end", "2", "--cache", str(tmp_path / "cache")
        ])
    assert exc.value.code == EXIT_ARGUMENTS


def test_relative_cache_path_is_rejected():
    with pytest.raises(SystemExit) as exc:
        parse_args(["--cache", "relative/cache"])
    assert exc.value.code == EXIT_ARGUMENTS


def test_checkpoint_interval_must_be_positive(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        parse_args([
            "--cache", str(tmp_path / "cache"), "--checkpoint-interval", "0"
        ])
    assert exc.value.code == EXIT_ARGUMENTS


def test_atomic_report_round_trip_has_no_temp_file(tmp_path: Path):
    path = tmp_path / "report.json"
    write_json_atomic(path, {"status": "success", "frames": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "success", "frames": [1, 2]
    }
    assert list(tmp_path.glob("report.json.*.tmp")) == []
