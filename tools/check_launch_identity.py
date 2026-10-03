#!/usr/bin/env python3
"""Prove a solver change leaves a bake bitwise identical.

Runs the profile harness for a scene preset twice in separate processes of the
same code, once with the cached-launch path disabled (``--plain``) and once
enabled, and prints a SHA-256 over the final particle state. The two digests
must match.

    python tools/check_launch_identity.py --scene-preset dam_break --frames 10
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(ns):
    import warp as wp
    from tools import profile_gpu_solver as pgs

    if ns.plain:
        original = wp.launch
        wp.launch = lambda *a, **k: original(*a, **k)

    captured = {}
    from fluidSim_warp_blender51.sim import solver as solver_module
    real_solver = solver_module.FluidSolver

    class Capturing(real_solver):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["solver"] = self

    solver_module.FluidSolver = Capturing
    args = argparse.Namespace(
        device="cuda:0", frames=ns.frames, warmup=0, emit_frames=4,
        voxel_size=None, max_particles=None, max_whitewater=200000,
        min_substeps=None, max_substeps=None, whitewater=False,
        no_surface=False, sync=False, no_phase_timers=True,
        scene_preset=ns.scene_preset, no_whitewater=False, viscosity=False,
        viscosity_constant=1.0, viscosity_iterations=None,
        viscosity_tolerance=None, surface_tension=False,
        surface_tension_coefficient=0.072)
    state = {}

    def hook(index, _stats):
        if index == ns.frames - 1:
            solver = captured["solver"]
            n = solver.particles.refresh_host_count()
            h = hashlib.sha256()
            h.update(solver.particles.position[:n].numpy().tobytes())
            h.update(solver.particles.velocity[:n].numpy().tobytes())
            state["digest"] = h.hexdigest()
            state["count"] = n

    pgs.run_profile(args, on_frame=hook)
    print(f"DIGEST {state['digest']} particles={state['count']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-preset", default="dam_break")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--child", action="store_true")
    ns = parser.parse_args(argv)
    if ns.child:
        return _digest(ns)
    digests = []
    for plain in (True, False):
        cmd = [sys.executable, __file__, "--child",
               "--scene-preset", ns.scene_preset, "--frames", str(ns.frames)]
        if plain:
            cmd.append("--plain")
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if l.startswith("DIGEST")]
        digests.append(line[-1] if line else "MISSING")
        print(("plain  " if plain else "cached ") + digests[-1])
    ok = digests[0] == digests[1] and digests[0] != "MISSING"
    print("LAUNCH-IDENTITY-" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
