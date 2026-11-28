"""Native two-view proof must compare pixels, not only camera labels."""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def harness():
    path = Path(__file__).resolve().parents[1] / "acceptance/blender_fspt_async_viewport_acceptance.py"
    spec = importlib.util.spec_from_file_location("async_view_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_view_oracle_rejects_identical_images_with_different_dimensions():
    with pytest.raises(AssertionError, match="different completed pixels"):
        harness().compare_view_pixels(np.ones((80, 100, 4)), np.ones((80, 101, 4)))


def test_two_view_oracle_reports_actual_rgb_change_on_common_grid():
    first, second = np.ones((80, 100, 4)), np.ones((80, 101, 4))
    first[:, :50, :3] = 0
    assert harness().compare_view_pixels(first, second) == pytest.approx(0.5)


def test_lifecycle_count_cli_survives_blenders_shared_cycles_argument_parser(monkeypatch, tmp_path):
    # Blender 5.2's installed cycles/engine.py parses this same post-"--"
    # tail with abbreviations enabled, before running the user script.
    # A bare --cycles dies there as an ambiguous prefix of these two options.
    tail = ["--json-out", str(tmp_path / "report.json"), "--lifecycle-cycles", "7"]
    cycles = argparse.ArgumentParser(add_help=False)
    cycles.add_argument("--cycles-print-stats", action="store_true")
    cycles.add_argument("--cycles-device")
    parsed, untouched = cycles.parse_known_args(tail)
    assert untouched == tail and not parsed.cycles_print_stats and parsed.cycles_device is None
    monkeypatch.setattr(sys, "argv", ["blender", "--", *tail])
    args = harness().arguments()
    assert args.cycles == 7 and args.json_out == tmp_path / "report.json"
