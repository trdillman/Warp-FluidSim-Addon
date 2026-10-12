"""Blender/CUDA acceptance: keyframed Custom Effect values follow the frame in a bake.

Preview re-read keyframed effect values every step (B66), but a Production Bake
kept the values it was admitted with, and its effect identity was taken at
wherever the playhead stood. Checks, in a real Blender with a CUDA bake:

  1. A keyframed Vortex Force strength reaches the baking solver frame by frame.
  2. A partial bake resumes with the playhead on another frame.
  3. Re-keying one baked frame's value is caught by resume as changed animation
     from exactly that frame.

Prints ``ANIMATED-EFFECT-BAKE-ACCEPTANCE-PASS`` and exits 0 when every check passes.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/blender_animated_effect_bake_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from _test_device import script_device  # noqa: E402

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []
START, PARTIAL_END, END = 1, 3, 6
KEYED = {frame: float(frame) for frame in range(START, END + 1)}


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _parameter(scene, key):
    return next(item for item in scene.fluidsim_effects.items[0].parameters
                if item.key == key)


def _key_strength(scene, values):
    strength = _parameter(scene, "strength")
    for frame, value in values.items():
        strength.float_value = value
        strength.keyframe_insert("float_value", frame=frame)


class _Recorder:
    """Records the strength the solver runs each step, by effect frame."""

    def __init__(self):
        from fluidSim_warp_blender51.sim.solver import FluidSolver

        self.cls = FluidSolver
        self.original = FluidSolver.step_frame
        self.rows = []
        recorder = self

        def step_frame(solver):
            from fluidSim_warp_blender51.blender import production_bake
            from fluidSim_warp_blender51.sim.effects.source import parameter_runtime_inputs

            runtime = solver.features.effects
            (instance,) = runtime.applied_instances()
            keys = instance.definition.kernels[0].parameter_keys
            values = dict(zip(keys, parameter_runtime_inputs(instance, 0), strict=True))
            recorder.rows.append((
                getattr(solver, "effect_frame", None), float(values["strength"]),
                production_bake.any_bake_active()))
            return recorder.original(solver)

        FluidSolver.step_frame = step_frame

    def take(self):
        rows, self.rows = self.rows, []
        return rows

    def close(self):
        self.cls.step_frame = self.original


def main(cache_root):
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import production_bake, scene_bridge
    from fluidSim_warp_blender51.blender.effects import add_builtin_effect

    bpy.context.preferences.edit.keyframe_new_interpolation_type = "CONSTANT"
    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    domain = scene_bridge.active_domain_settings(scene)
    domain.voxel_size = 0.12
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    fs.cache_path = str(cache_root)
    fs.live = False
    scene.frame_start, scene.frame_end = START, END
    fs.bake_start, fs.bake_end = START, END

    add_builtin_effect(scene, "fluidsim.vortex_force")
    _key_strength(scene, KEYED)
    scene.frame_set(START)
    check("effect applied",
          bpy.ops.fluidsim.effect_apply_stack(device=script_device()) == {"FINISHED"},
          scene.fluidsim_effects.blocking_reason)

    recorder = _Recorder()
    try:
        # A partial bake started with the playhead away from the start frame.
        scene.frame_set(5)
        result = production_bake.run_synchronous(scene, start=START, end=PARTIAL_END)
        check("partial bake completes",
              result.last_committed_frame == PARTIAL_END, result)
        rows = recorder.take()
        check("steps ran inside a Production Bake", rows and all(r[2] for r in rows), rows)
        check("keyframed strength follows each baked frame",
              [(r[0], r[1]) for r in rows]
              == [(f, KEYED[f]) for f in range(START, PARTIAL_END + 1)], rows)

        effect = scene.fluidsim_effects.items[0]
        check("effect telemetry reaches the effect panel",
              effect.calls > 0 and effect.elements > 0 and not effect.runtime_error,
              (effect.calls, effect.elements, effect.submission_ms, effect.runtime_error))
        scene.frame_set(END - 1)
        decision = production_bake.ProductionBakeBackend().resume_decision(scene, end=END)
        check("partial bake resumes with the playhead elsewhere",
              decision.allowed and decision.resume_frame == PARTIAL_END + 1, decision)

        _key_strength(scene, {2: 20.0})
        scene.frame_set(END - 1)
        decision = production_bake.ProductionBakeBackend().resume_decision(scene, end=END)
        check("re-keyed baked frame is changed animation from that frame",
              not decision.allowed and decision.code == "SOURCE_ANIMATION_CHANGED"
              and decision.first_changed_frame == 2, decision)
        _key_strength(scene, {2: KEYED[2]})

        scene.frame_set(END - 1)
        result = production_bake.run_synchronous(scene, end=END, resume=True)
        rows = recorder.take()
        check("resumed bake completes",
              result.last_committed_frame == END, (result, rows))
        check("resumed frames follow the keys",
              [(r[0], r[1]) for r in rows]
              == [(f, KEYED[f]) for f in range(PARTIAL_END + 1, END + 1)], rows)
    finally:
        recorder.close()
        if scene_bridge.session().active:
            scene_bridge.session().release(scene)


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp(prefix="fluidsim-animated-effect-bake-"))
    try:
        main(root / "cache")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print("ANIMATED-EFFECT-BAKE-ACCEPTANCE-FAIL", FAILURES, flush=True)
        sys.exit(1)
    print("ANIMATED-EFFECT-BAKE-ACCEPTANCE-PASS", flush=True)
    sys.exit(0)
