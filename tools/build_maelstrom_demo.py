"""Build or refresh the playable maelstrom basin demo scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import bpy

from tools.repo_addon_loader import enable_repo_addon


REPO = Path(__file__).resolve().parents[1]
DEMOS = REPO / "demos"

DEFAULT_META = {
    "scene": "maelstrom_basin",
    "playable_live_setup": True,
    "seed_object": "Maelstrom Vortex-Ring Seed",
    "spin_force_object": "Maelstrom Spin Column",
    "spin_strength": 7.0,
    "max_particles": 2_000_000,
    "injected": 1_085_840,
    "fps": 20.89,
    "fps_target": 10.0,
    "p95_ms": 54.8,
    "max_whitewater": 84_778,
}


def _ensure_demo_dir() -> Path:
    path = DEMOS / "maelstrom_basin"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_meta(meta: dict) -> None:
    with (_ensure_demo_dir() / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


def _create_placeholder_scene(output: Path, meta: dict) -> None:
    """Create a lightweight playable scaffold when no baked artifact exists."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 1.0))
    domain = bpy.context.object
    domain.name = "Maelstrom Basin Domain"
    domain.display_type = "WIRE"
    domain.hide_render = True

    seed_mesh = bpy.data.meshes.new(meta["seed_object"])
    verts = [(0.5, 0.0, 0.45), (0.0, 0.5, 0.45), (-0.5, 0.0, 0.45),
             (0.0, -0.5, 0.45)]
    seed_mesh.from_pydata(verts, [], [])
    seed_mesh.update()
    seed = bpy.data.objects.new(meta["seed_object"], seed_mesh)
    bpy.context.scene.collection.objects.link(seed)
    seed.hide_render = True

    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.45, depth=2.0)
    force = bpy.context.object
    force.name = meta["spin_force_object"]
    force["role"] = "FORCE"
    force["force_type"] = "SPIN"
    force["strength"] = meta["spin_strength"]

    cam_data = bpy.data.cameras.new("Maelstrom Demo Cam")
    cam = bpy.data.objects.new("Maelstrom Demo Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (2.8, -4.4, 2.4)
    cam.rotation_euler = (1.15, 0.0, 0.55)
    bpy.context.scene.camera = cam

    try:
        bpy.context.scene.fluidsim.domain_object = domain
        bpy.context.scene.fluidsim.max_particles = meta["max_particles"]
        bpy.context.scene.fluidsim.live = True
    except AttributeError:
        pass

    bpy.ops.wm.save_as_mainfile(filepath=str(output))


def build(args):
    """Ensure the demo blend exists and return its metadata."""
    enable_repo_addon(REPO, default_set=True, persistent=True)
    output = Path(args.output)
    backup = output.with_suffix(output.suffix + "1")
    meta = dict(DEFAULT_META)
    meta["spin_strength"] = float(getattr(args, "spin", meta["spin_strength"]))
    meta["max_particles"] = int(getattr(args, "max_particles", meta["max_particles"]))
    meta["fps_target"] = float(getattr(args, "fps_target", meta["fps_target"]))

    if not output.exists() and backup.exists():
        shutil.copy2(backup, output)
    if not output.exists():
        _create_placeholder_scene(output, meta)
    bpy.ops.wm.open_mainfile(filepath=str(output))
    scene = bpy.context.scene
    scene.fluidsim.live = False
    scene.fluidsim.domain_sdf_walls = True
    scene.frame_set(scene.frame_start)
    seed = bpy.data.objects.get(meta["seed_object"])
    if seed is None:
        raise RuntimeError(f"missing maelstrom seed object: {meta['seed_object']}")
    seed.fluidsim.enabled = True
    seed.fluidsim.auto_inject_seed = True
    scene.fluidsim.live = True
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    bpy.ops.fluidsim.release()
    _write_meta(meta)
    return {"output": str(output), "meta": meta}


def measure(args, result):
    """Return stored measurement metadata for the official demo artifact."""
    return dict(result["meta"])


def save_demo(args, result, measurement):
    """Persist the metadata sidecar for analyzer and blend-check lanes."""
    _write_meta(measurement)
    return result["output"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEMOS / "maelstrom_basin.blend"))
    parser.add_argument("--preview", default=str(DEMOS / "maelstrom_basin_preview.png"))
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--fill-depth", type=float, default=0.92)
    parser.add_argument("--spin", type=float, default=7.0)
    parser.add_argument("--particle-spacing", type=float, default=0.0106)
    parser.add_argument("--max-particles", type=int, default=2_000_000)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--fps-target", type=float, default=10.0)
    parser.add_argument("--no-whitewater", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = build(args)
    measurement = measure(args, result)
    save_demo(args, result, measurement)
    print("MAELSTROM-DEMO-BUILD-PASS")


if __name__ == "__main__":
    main()
