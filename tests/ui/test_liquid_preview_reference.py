"""Liquid Preview pixel references: real Blender, offscreen, SSIM >= 0.97.

Renders the seeded synthetic sphere + sheet payload in ``blender --background``
and compares each view with ``tests/references/liquid_preview/<view>.png``.

Regenerate the references only on purpose, and add a changelog line saying why:

    FLUIDSIM_REGENERATE_LIQUID_PREVIEW=1 pytest tests/ui/test_liquid_preview_reference.py

``BLENDER_EXE`` overrides the Blender executable. The test skips (with the
verbatim error) when Blender is missing or this machine cannot draw offscreen.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fluidSim_warp_blender51.blender.liquid_preview import reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "tests" / "references" / "liquid_preview"
DRIVER = Path(__file__).with_name("liquid_preview_render_driver.py")
REGENERATE_ENV = "FLUIDSIM_REGENERATE_LIQUID_PREVIEW"
SSIM_THRESHOLD = 0.97
DEFAULT_BLENDER = Path(
    "C:/Users/Tyler/Downloads/blender-5.2.0-windows-x64/"
    "blender-5.2.0-windows-x64/blender.exe")


def _blender_exe():
    configured = os.environ.get("BLENDER_EXE")
    candidate = Path(configured) if configured else DEFAULT_BLENDER
    if not candidate.is_file():
        pytest.skip(f"Blender executable not found: {candidate} (set BLENDER_EXE)")
    return candidate


def _render(tmp_path):
    out_dir = tmp_path / "render"
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    # A private extensions directory: --factory-startup must never touch the
    # real profile's installed extensions.
    env["BLENDER_USER_EXTENSIONS"] = str(tmp_path / "user_extensions")
    command = [str(_blender_exe()), "--background", "--factory-startup",
               "--python-exit-code", "1", "--python", str(DRIVER),
               "--", str(out_dir)]
    completed = subprocess.run(
        command, env=env, cwd=str(ROOT), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=600, check=False)
    status_path = out_dir / "status.json"
    if not status_path.is_file():
        pytest.fail(f"Blender wrote no status (exit {completed.returncode}):\n"
                    f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] == "unavailable":
        pytest.skip(f"offscreen GPU drawing unavailable: {status['error']}")
    assert status["status"] == "ok", status.get("error")
    return out_dir, status


@pytest.mark.blender
def test_liquid_preview_matches_references(tmp_path):
    out_dir, status = _render(tmp_path)
    if os.environ.get(REGENERATE_ENV) == "1":
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        for view in reference.REFERENCE_VIEWS:
            shutil.copyfile(out_dir / f"{view}.png", REFERENCE_DIR / f"{view}.png")
    scores = {}
    for view in reference.REFERENCE_VIEWS:
        expected_path = REFERENCE_DIR / f"{view}.png"
        assert expected_path.is_file(), (
            f"missing reference {expected_path}; set {REGENERATE_ENV}=1 to create it")
        expected = reference.read_png(expected_path)[..., :3]
        actual = reference.read_png(out_dir / f"{view}.png")[..., :3]
        assert actual.shape == (reference.REFERENCE_SIZE[1],
                                reference.REFERENCE_SIZE[0], 3)
        scores[view] = reference.ssim(actual, expected)
    print(f"Liquid Preview SSIM on {status['backend']}: {scores}")
    failing = {view: score for view, score in scores.items()
               if score < SSIM_THRESHOLD}
    assert not failing, f"SSIM below {SSIM_THRESHOLD}: {failing}"
