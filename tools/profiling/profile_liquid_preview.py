"""Time Liquid Preview redraws offscreen in background Blender; prints JSON.

    blender --background --factory-startup --python-exit-code 1 \\
        --python tools/profiling/profile_liquid_preview.py -- \\
        [--width 1920] [--height 1080] [--particles 600000] [--scale 0.5] \\
        [--redraws 60] [--out result.json]

Each redraw renders the full pass chain into a 1080p offscreen and is closed
by a one-pixel colour read, which waits for the GPU; the median of that
read alone is measured and subtracted. ``redraw_ms`` is the steady state (same
frame id, no upload); ``upload_redraw_ms`` is a redraw that also uploads a new
frame. Budget (SPEC-liquid-preview): <= 4 ms per redraw at these defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUDGET_MS = 4.0


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--particles", type=int, default=600_000)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--redraws", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out")
    return parser.parse_args(argv)


def _sync(framebuffer):
    framebuffer.read_color(0, 0, 1, 1, 4, 0, "FLOAT")


def run(args):
    import gpu

    sys.path.insert(0, ROOT)
    from dataclasses import replace

    from fluidSim_warp_blender51.blender.liquid_preview import passes, reference

    gpu.init()
    width, height = args.width, args.height
    record = reference.reference_record(count=args.particles)
    settings = passes.PreviewSettings(scale=args.scale)
    view, projection = reference.reference_camera(width, height)
    renderer = passes.LiquidPreviewRenderer()
    target = gpu.types.GPUOffScreen(width, height, format="RGBA8")
    redraw, upload_redraw, sync_only = [], [], []
    try:
        with target.bind():
            framebuffer = gpu.state.active_framebuffer_get()
            gpu.state.viewport_set(0, 0, width, height)

            def draw(current):
                framebuffer.clear(color=reference.BACKGROUND, depth=1.0)
                _sync(framebuffer)
                start = time.perf_counter()
                renderer.draw(current, view_matrix=view,
                              projection_matrix=projection,
                              viewport=(0, 0, width, height), settings=settings)
                _sync(framebuffer)
                return (time.perf_counter() - start) * 1000.0

            for _ in range(args.warmup):
                draw(record)
            for _ in range(args.redraws):
                redraw.append(draw(record))
                start = time.perf_counter()
                _sync(framebuffer)
                sync_only.append((time.perf_counter() - start) * 1000.0)
            for index in range(args.redraws):
                fresh = replace(record, frame_id=10_000 + index)
                upload_redraw.append(draw(fresh))
    finally:
        target.free()
        renderer.free()

    sync_ms = statistics.median(sync_only)
    redraw_ms = max(statistics.median(redraw) - sync_ms, 0.0)
    return {
        "backend": f"{gpu.platform.backend_type_get()} {gpu.platform.renderer_get()}",
        "width": width, "height": height, "scale": args.scale,
        "particles": record.particle_count,
        "whitewater": record.whitewater_count,
        "redraws": args.redraws,
        "redraw_ms": round(redraw_ms, 3),
        "redraw_ms_p90": round(
            statistics.quantiles(redraw, n=10)[-1] - sync_ms, 3),
        "upload_redraw_ms": round(
            max(statistics.median(upload_redraw) - sync_ms, 0.0), 3),
        "sync_read_ms": round(sync_ms, 3),
        "budget_ms": BUDGET_MS,
        "within_budget": redraw_ms <= BUDGET_MS,
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = _parse(argv)
    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
