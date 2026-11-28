"""Windowed Blender rendered-viewport acceptance with a bounded timer."""
import hashlib
import importlib.util
import os
import time
import traceback
from pathlib import Path

import bpy
import numpy as np

sibling = Path(__file__).with_name("blender_fspt_acceptance.py")
spec = importlib.util.spec_from_file_location("fspt_acceptance_fixture", sibling)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def main():
    args = fixture.arguments("artifacts/fspt-p1/viewport.json")
    path = Path(args.json_out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "verifier_sha256": fixture.sha(Path(__file__)),
                  "fixture_sha256": fixture.sha(sibling), "stages": {}}
    started = time.monotonic()

    def failed():
        report.update(status="FAILED", error=traceback.format_exc())
        fixture.write_report(path, report)
        print("FSPT-VIEWPORT-FAIL", report["error"], flush=True)
        os._exit(1)

    try:
        fixture.checkpoint(path, report, "register")
        assert not bpy.app.background, "Requires a real Blender window"
        addon = fixture.load_addon(args, report)
        fixture.checkpoint(path, report, "runtime_probe")
        report["runtime"] = fixture.runtime_info(addon)
        fixture.checkpoint(path, report, "fixture_setup")
        scene, node, _ = fixture.fixture()
        window = bpy.context.window_manager.windows[0]
        area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
        region = next(r for r in area.regions if r.type == "WINDOW")
        space = area.spaces.active
        # Keep real window size, reduce traced pixels in the diagnostic fixture.
        scene.fspt.resolution_percentage = 25
        scene.fspt.viewport_samples = 8
        space.region_3d.view_perspective = "CAMERA"
        space.overlay.show_overlays = False
        fixture.checkpoint(path, report, "initial")
        space.shading.type = "RENDERED"
        area.tag_redraw()
    except BaseException:  # noqa: BLE001 - persist failure and quit the owned process
        failed()

    stages = iter(("initial", "camera_edit", "material_edit", "resize"))
    stage = next(stages)
    prior = {}
    observations = []

    prior_patch = None

    def tick():
        nonlocal stage, prior, observations, prior_patch
        try:
            if time.monotonic() - started >= 120:
                raise TimeoutError(f"Viewport stage {stage}: last observations {observations[-3:]}")
            frame = fixture.info(addon, "LAST_VIEWPORT_INFO")
            assert frame.get("status") != "ERROR", frame
            if not frame or not frame.get("key") or frame["key"] == prior.get("key"):
                return 0.03
            fixture.validate_info(frame)
            key = frame["key"]
            if stage == "camera_edit" and key[1] == prior["key"][1]:
                return 0.03
            if stage == "material_edit" and key[0] == prior["key"][0]:
                return 0.03
            if stage == "resize" and not (frame["width"] < prior["width"] and
                                          frame["height"] < prior["height"]):
                return 0.03
            assert frame["draw_count"] > prior.get("draw_count", 0), frame
            observation = {"key": frame["key"], "samples": frame["samples"], "draws": frame["draw_count"]}
            if not observations or observations[-1] != observation:
                observations.append(observation)
            if frame["samples"] < scene.fspt.viewport_samples:
                return 0.03
            report["stages"][stage] = {"frame": frame, "observations": observations}
            # Read an actual screen capture. Window decorations & panels alone
            # cannot satisfy the central viewport patch variation assertion.
            fixture.checkpoint(path, report, stage + "_screenshot")
            screenshot = path.with_name(path.stem + "-" + stage + "-screenshot.png")
            if screenshot.exists():
                screenshot.unlink()
            with bpy.context.temp_override(window=window, area=area, region=region):
                result = bpy.ops.screen.screenshot(filepath=str(screenshot))
            assert result == {"FINISHED"}
            image = bpy.data.images.load(str(screenshot), check_existing=False)
            try:
                w, h = image.size
                rgba = np.empty(w * h * 4, np.float32)
                image.pixels.foreach_get(rgba)
                sx, sy = w / window.width, h / window.height
                x0, x1 = (round((region.x + region.width * t) * sx) for t in (0.35, 0.65))
                y0, y1 = (round((region.y + region.height * t) * sy) for t in (0.35, 0.65))
                patch = rgba.reshape(h, w, 4)[y0:y1, x0:x1, :3].copy()
                assert patch.size and np.isfinite(patch).all()
                contrast = float(np.percentile(patch.mean(axis=2), 95) -
                                 np.percentile(patch.mean(axis=2), 5))
                assert contrast > 0.01, f"Viewport center lacks image contrast: {contrast}"
            finally:
                bpy.data.images.remove(image)
            capture = {"screenshot": str(screenshot), "screenshot_sha256": fixture.sha(screenshot),
                       "center_patch_sha256": hashlib.sha256(patch.tobytes()).hexdigest(),
                       "center_patch_contrast": contrast, "region_size": [region.width, region.height]}
            report["stages"][stage].update(capture)
            report.update(capture)
            if stage in ("camera_edit", "material_edit"):
                assert patch.shape == prior_patch.shape
                delta = float(np.abs(patch - prior_patch).mean())
                report["stages"][stage]["center_patch_mean_abs_delta"] = delta
                assert delta > 0.001, f"{stage} screenshot change too small: {delta}"
            prior_patch = patch
            if stage == "initial":
                fixture.checkpoint(path, report, "camera_edit")
                scene.camera.location.x += 0.8
            elif stage == "camera_edit":
                fixture.checkpoint(path, report, "material_edit")
                node.inputs["Color"].default_value = (0.02, 0.9, 0.07, 1)
            elif stage == "material_edit":
                fixture.checkpoint(path, report, "resize")
                scene.fspt.resolution_percentage = 15
                # The async viewport is idle at its sample goal. A scripted
                # settings edit needs a redraw to deliver the new request.
                area.tag_redraw()
            else:
                for name, item in report["stages"].items():
                    final = item["frame"]
                    assert any(obs["key"] == final["key"] and obs["draws"] < final["draw_count"]
                               and obs["samples"] < final["samples"] for obs in item["observations"]), (
                                   f"No progressive sample increase observed for {name}", item)
                fixture.checkpoint(path, report, "unregister")
                space.shading.type = "SOLID"
                scene.render.engine = "BLENDER_WORKBENCH"
                addon.unregister()
                report.update(status="VERIFIED", stage="complete", elapsed_seconds=time.monotonic() - started)
                fixture.write_report(path, report)
                print("FSPT-VIEWPORT-PASS", path, flush=True)
                bpy.ops.wm.quit_blender()
                return None
            prior = frame
            observations = []
            stage = next(stages)
            return 0.03
        except BaseException:  # noqa: BLE001 - persist failure and quit the owned process
            failed()

    bpy.app.timers.register(tick, first_interval=0.03)


if __name__ == "__main__":
    main()
