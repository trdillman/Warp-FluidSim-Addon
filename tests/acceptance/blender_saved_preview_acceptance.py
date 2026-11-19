"""Parent-run installed CUDA save/reopen probe; run each phase in a fresh Blender.

This script is NOT part of the CPU evidence. See the adjacent PR72 evidence
report for the broker, isolated-profile and visible-playback prerequisites.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import addon_utils
import bpy
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _require_cuda_session(sess, wp):
    assert sess.active and wp.get_device(sess.solver.device).is_cuda, "CUDA solver required"


def _enable_installed_runtime(module, installed):
    assert not any(os.environ.get(key) for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")), (
        "Installed acceptance must discard developer Python import overrides"
    )
    assert "warp" not in sys.modules, "Use a fresh Blender process without preloaded Warp"
    extensions = Path(os.environ["BLENDER_USER_EXTENSIONS"]).resolve()
    owned_site = Path(bpy.utils.extension_path_user(module, path=".local", create=False)).resolve()
    assert owned_site.is_relative_to(extensions), owned_site
    assert (owned_site / "warp" / "__init__.py").is_file(), (
        "Provision the locked wheel into this extension's owned user directory"
    )
    # Match the canonical installed smoke bootstrap, before registration can
    # prewarm/import Blender's bundled Warp. The parent provisions this directory
    # from the retained checksummed wheel, never from the development overlay.
    sys.path.insert(0, str(owned_site))
    addon_utils.enable(module, default_set=True, persistent=True)
    addon = importlib.import_module(module)
    assert Path(addon.__file__).resolve().parent == installed, addon.__file__
    wp = importlib.import_module("warp")
    assert Path(wp.__file__).resolve().is_relative_to(owned_site), wp.__file__
    return addon, wp, owned_site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("produce", "reopen"), required=True)
    parser.add_argument("--case", choices=("surface", "particles"), required=True)
    parser.add_argument("--addon-module", required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-blend", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source_root.resolve()
    installed = args.installed_root.resolve()
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip(), "Freeze and commit the reviewed source/verifier before GPU acceptance"
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    addon, wp, owned_site = _enable_installed_runtime(args.addon_module, installed)
    version = importlib.import_module(args.addon_module + ".version")
    assert wp.__version__ == version.CERTIFIED_WARP_VERSION
    with zipfile.ZipFile(args.archive) as archive:
        for relative in ("blender/output_publication.py", "blender/liquid_preview/handler.py"):
            data = (installed / relative).read_bytes()
            assert data == (source / "fluidSim_warp_blender51" / relative).read_bytes(), relative
            assert data == archive.read(relative), f"archive mismatch: {relative}"
    output = importlib.import_module(args.addon_module + ".blender.output_publication")
    bridge = importlib.import_module(args.addon_module + ".blender.scene_bridge")
    display = importlib.import_module(args.addon_module + ".blender.liquid_display")
    blend = args.output / (args.case + ".blend")
    producer = args.output / (args.case + "-produce.json")
    identity = {"source_head": head, "addon_file": addon.__file__,
                "archive_sha256": digest(args.archive.read_bytes()), "blender": bpy.app.version_string,
                "warp": wp.__version__, "warp_file": wp.__file__,
                "warp_owned_site": str(owned_site)}
    if args.phase == "produce":
        assert args.input_blend and args.input_blend.is_file(), "Supply a retained real demo"
        bpy.ops.wm.open_mainfile(filepath=str(args.input_blend.resolve()))
        scene = bpy.context.scene
        fs = scene.fluidsim
        fs.liquid_view_mode = "PARTICLES" if args.case == "particles" else "SURFACE"
        fs.liquid_display_mode = "PREVIEW"
        ds = bridge.active_domain_settings(scene)
        assert ds is not None, "The demo must designate a real domain"
        ds.surface_enabled = True
        fs.live = True
        bridge.force_release_session()
        scene.frame_set(scene.frame_start)
        scene.frame_set(scene.frame_start + 1)
        sess = bridge.session()
        _require_cuda_session(sess, wp)
        before = int(sess.solver.frame)
        scene.frame_set(scene.frame_start + 2)
        assert int(sess.solver.frame) > before, "real timeline did not advance"
        bridge.reset_simulation(scene)
        reset_frame = int(sess.solver.frame)
        scene.frame_set(scene.frame_start + 3)
        assert int(sess.solver.frame) > reset_frame, "reset/playback did not advance"
        output.update_outputs(scene, sess, force_surface=True)
        record = output.latest_published_points()
        assert record is not None and record.particle_count > 0
        assert np.isfinite(record.positions).all()
        if args.case == "surface":
            surface = output.output_object(scene, output.SURFACE_NAME)
            assert surface is not None and len(surface.data.vertices) > 0
            assert not surface.hide_render
        points_hash = digest(np.asarray(record.positions, dtype=np.float32).tobytes())
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        assert output.live_point_count(output.output_object(scene, output.PARTICLES_NAME)) > 0
        result = identity | {"phase": args.phase, "case": args.case, "points_sha256": points_hash,
                             "particle_count": record.particle_count, "radius": record.particle_radius,
                             "scene_frame": scene.frame_current, "device": str(sess.solver.device),
                             "reset_frame": reset_frame, "advanced_frame": int(sess.solver.frame),
                             "blend_sha256": digest(blend.read_bytes())}
        producer.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        expected = json.loads(producer.read_text(encoding="utf-8"))
        assert expected["source_head"] == head
        assert expected["archive_sha256"] == identity["archive_sha256"]
        assert expected["blend_sha256"] == digest(blend.read_bytes())
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        scene = bpy.context.scene
        assert not bridge.session().active, "reopen must restore without running a solver"
        record = output.latest_published_points()
        assert record is not None and record.particle_count == expected["particle_count"]
        assert digest(np.asarray(record.positions, dtype=np.float32).tobytes()) == expected["points_sha256"]
        assert abs(record.particle_radius - expected["radius"]) < 1e-6
        assert scene.frame_current == expected["scene_frame"]
        fs = scene.fluidsim
        assert fs.liquid_display_mode == "PREVIEW" and fs.liquid_preview_enabled
        assert fs.show_particles == (args.case == "particles")
        assert fs.show_surface == (args.case == "surface")
        display.apply_viewport(scene, deferred=False)
        obj = output.output_object(scene, output.PARTICLES_NAME)
        assert obj.hide_render == (args.case == "surface")
        assert obj.hide_get(), "Liquid Preview must own the viewport representation"
        result = identity | {"phase": args.phase, "case": args.case, "restored": True,
                             "particle_count": record.particle_count, "solver_active": False}
    (args.output / (args.case + "-" + args.phase + ".json")).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SAVED-PREVIEW-ACCEPTANCE-PASS", args.case, args.phase)


if __name__ == "__main__":
    main()
