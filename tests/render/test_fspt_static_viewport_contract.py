"""CPU control for the static viewport harness's asynchronous resize wake-up."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_static_viewport_redraws_after_changing_resolution_at_sample_goal(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "acceptance/blender_fspt_viewport_acceptance.py"
    report = tmp_path / "viewport.json"
    settings = SimpleNamespace()
    scene = SimpleNamespace(fspt=settings, camera=SimpleNamespace(location=SimpleNamespace(x=0)))
    node = SimpleNamespace(inputs={"Color": SimpleNamespace()})
    space = SimpleNamespace(region_3d=SimpleNamespace(), overlay=SimpleNamespace(),
                            shading=SimpleNamespace())
    region = SimpleNamespace(type="WINDOW", x=0, y=0, width=40, height=40)
    redraws, timers = [], []
    area = SimpleNamespace(type="VIEW_3D", regions=[region], spaces=SimpleNamespace(active=space),
                           tag_redraw=lambda: redraws.append(settings.resolution_percentage))
    window = SimpleNamespace(screen=SimpleNamespace(areas=[area]), width=40, height=40)
    pixels = np.linspace(0, 0.7, 40 * 40 * 4, dtype=np.float32)

    def load_screenshot(*_args, **_kwargs):
        # Keep enough contrast and inter-stage variation for the real screenshot checks.
        return SimpleNamespace(size=(40, 40), pixels=SimpleNamespace(
            foreach_get=lambda out: np.copyto(out, pixels + 0.05 * (len(pixels_drawn) - 1))))

    pixels_drawn = []

    def screenshot(filepath):
        Path(filepath).write_bytes(b"Blender screenshot stub")
        pixels_drawn.append(filepath)
        return {"FINISHED"}

    bpy = SimpleNamespace(
        app=SimpleNamespace(background=False, timers=SimpleNamespace(
            register=lambda callback, **_kwargs: timers.append(callback))),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=[window]),
                                temp_override=lambda **_kwargs: nullcontext()),
        ops=SimpleNamespace(screen=SimpleNamespace(screenshot=screenshot)),
        data=SimpleNamespace(images=SimpleNamespace(load=load_screenshot, remove=lambda _: None)),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=object))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--json-out", str(report)])
    monkeypatch.delenv("FLUIDSIM_TEST_DEVICE", raising=False)

    spec = importlib.util.spec_from_file_location("static_viewport_acceptance", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.os = SimpleNamespace(_exit=lambda code: pytest.fail(report.read_text()))
    monkeypatch.setattr(harness.fixture, "load_addon", lambda *_args: object())
    monkeypatch.setattr(harness.fixture, "runtime_info", lambda _: {})
    monkeypatch.setattr(harness.fixture, "fixture", lambda: (scene, node, None))
    receipts = iter([
        ("scene", "camera", 1),
        ("scene", "edited_camera", 2),
        ("edited_scene", "edited_camera", 3),
    ])

    def latest_draw(*_args):
        geometry, camera, draw_count = next(receipts)
        return dict(status="COMPLETE", key=[geometry, camera, 10, 10, 6, 31],
                    device="cuda:0", backend="warp-cuda-async-v1", transport="host-pinned-rgba32f",
                    width=10, height=10, samples=8, draw_count=draw_count)

    monkeypatch.setattr(harness.fixture, "info", latest_draw)
    harness.main()
    tick, = timers
    for _ in range(3):
        assert tick() is not None

    assert json.loads(report.read_text())["stage"] == "resize"
    assert redraws == [25, 15], "Resize must wake the idle rendered area after editing its percentage"
