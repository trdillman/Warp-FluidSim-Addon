from __future__ import annotations

import json
from pathlib import Path

import pytest
from fluidsim_testpkg.blender.command_line import (
    EXIT_ARGUMENTS,
    EXIT_RESUME_INCOMPATIBLE,
    CliOptions,
    _is_legacy_2x_cache,
    parse_args,
    run,
    write_json_atomic,
)
from fluidsim_testpkg.io.cache_schema import LEGACY_2X_CACHE_MESSAGE


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


def _write_2x_cache(cache: Path) -> None:
    """A minimal FluidSim 2.x (format 2) cache: manifest plus one frame file."""
    (cache / "frames").mkdir(parents=True)
    (cache / "frames" / "frame_000001.npz").write_bytes(b"frame-2x")
    (cache / "fluidsim_cache.json").write_text(
        json.dumps({"format_version": 2, "frame_start": 1, "frame_end": 5,
                    "frames": {}}),
        encoding="utf-8")


def _tree(base: Path) -> dict[str, bytes]:
    return {
        item.relative_to(base).as_posix(): item.read_bytes()
        for item in sorted(base.rglob("*")) if item.is_file()
    }


@pytest.mark.parametrize("resume", ["never", "auto", "required"])
def test_a_2x_cache_is_refused_under_every_resume_mode(
        tmp_path: Path, capsys, resume: str):
    cache = tmp_path / "cache"
    report = tmp_path / "report.json"
    _write_2x_cache(cache)
    before = _tree(cache)

    code = run(parse_args([
        "--cache", str(cache), "--resume", resume, "--report-json", str(report),
    ]))

    assert code == EXIT_RESUME_INCOMPATIBLE == 14
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["message"] == LEGACY_2X_CACHE_MESSAGE
    assert payload["exit_code"] == EXIT_RESUME_INCOMPATIBLE
    assert payload["status"] == "failed"
    assert payload["frames_committed"] == 0
    assert _tree(cache) == before, "a refused 2.x cache must be left untouched"
    finished = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert finished["event"] == "bake_finished"
    assert finished["exit_code"] == EXIT_RESUME_INCOMPATIBLE
    assert finished["message"] == LEGACY_2X_CACHE_MESSAGE


def test_only_a_2x_manifest_counts_as_a_2x_cache(tmp_path: Path):
    legacy = tmp_path / "legacy"
    _write_2x_cache(legacy)
    current = tmp_path / "current"
    current.mkdir()
    (current / "fluidsim_cache.json").write_text(
        json.dumps({"format_version": 3}), encoding="utf-8")
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "fluidsim_cache.json").write_text("{not json", encoding="utf-8")

    assert _is_legacy_2x_cache(legacy) is True
    assert _is_legacy_2x_cache(current) is False
    assert _is_legacy_2x_cache(corrupt) is False
    assert _is_legacy_2x_cache(tmp_path / "missing") is False


def test_the_resume_never_refusal_does_not_name_a_cache_version():
    source = (Path(__file__).resolve().parents[1] / "blender"
              / "command_line.py").read_text(encoding="utf-8")

    assert "--resume=never will not overwrite it" in source
    assert "cache-v2" not in source.lower()


def test_atomic_report_round_trip_has_no_temp_file(tmp_path: Path):
    path = tmp_path / "report.json"
    write_json_atomic(path, {"status": "success", "frames": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "success", "frames": [1, 2]
    }
    assert list(tmp_path.glob("report.json.*.tmp")) == []
