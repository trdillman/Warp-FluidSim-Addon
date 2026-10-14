"""Blender-side driver for ``test_liquid_preview_reference.py``.

Run as ``blender --background --factory-startup --python <this> -- <out_dir>``.
Renders every Liquid Preview reference view into ``<out_dir>`` and writes
``status.json``. ``status`` is ``"unavailable"`` when this machine cannot draw
offscreen at all (the test then skips with the verbatim error) and ``"error"``
when our own rendering code fails (the test then fails).
"""

from __future__ import annotations

import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_status(out_dir, **status):
    with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        import gpu

        gpu.init()
        probe = gpu.types.GPUOffScreen(8, 8, format="RGBA8")
        probe.free()
        backend = f"{gpu.platform.backend_type_get()} {gpu.platform.renderer_get()}"
    except Exception as exc:  # noqa: BLE001 - reported verbatim by the test
        _write_status(out_dir, status="unavailable",
                      error=f"{type(exc).__name__}: {exc}")
        return
    try:
        sys.path.insert(0, ROOT)
        from fluidSim_warp_blender51.blender.liquid_preview import reference

        for view in reference.REFERENCE_VIEWS:
            image = reference.render_reference(view)
            reference.write_png(os.path.join(out_dir, f"{view}.png"), image)
    except Exception:  # noqa: BLE001 - reported verbatim by the test
        _write_status(out_dir, status="error", backend=backend,
                      error=traceback.format_exc())
        return
    _write_status(out_dir, status="ok", backend=backend,
                  views=list(reference.REFERENCE_VIEWS))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv[0])
