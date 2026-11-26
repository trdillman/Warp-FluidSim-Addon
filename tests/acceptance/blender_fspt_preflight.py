"""Real Blender/CUDA capability probe; routed by blender-gate, no fake bpy."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fluidSim_warp_blender51.runtime.warp_runtime import _import_warp, probe_runtime

wp = _import_warp()


@wp.kernel
def fspt_capability_probe(out: wp.array(dtype=float)):
    out[0] = 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="artifacts/fspt-p1/preflight.json")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAILED", "blender": bpy.app.version_string,
                  "blender_build": bpy.app.build_hash.decode(), "python": sys.version,
                  "os": platform.platform(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "warp": wp.__version__, "warp_path": wp.__file__, "addon_root": str(ROOT)}
    try:
        state = probe_runtime(force=True)
        assert state.available, state.message
        device = wp.get_device(state.cuda_devices[0].alias)
        assert device.is_cuda, "CPU is forbidden for FSPT"
        output = wp.zeros(1, dtype=float, device=device)
        wp.launch(fspt_capability_probe, dim=1, inputs=[output], device=device)
        wp.synchronize_device(device)
        assert float(output.numpy()[0]) == 3.0
        report.update(status="VERIFIED", device=str(device), gpu=device.name,
                      cuda_driver=state.cuda_driver_version, cuda_toolkit=state.cuda_toolkit_version,
                      runtime_warning=state.warning, executed_kernel_value=3.0,
                      render_engine_methods={x: x in bpy.types.RenderEngine.bl_rna.functions for x in
                          ("begin_result", "end_result", "bind_display_space_shader", "tag_redraw")})
        assert all(report["render_engine_methods"].values())
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FSPT-PREFLIGHT-PASS " + json.dumps(report))


if __name__ == "__main__":
    main()
