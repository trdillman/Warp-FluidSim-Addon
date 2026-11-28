"""Native Solver Session completion clock, independently before surface work."""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_acceptance as base
import blender_fspt_device_surface_acceptance as surface_fixture


def facts(clock):
    return {name: getattr(clock, name) for name in (
        "scene_token", "owner_token", "epoch", "solver_frame", "state_revision",
        "simulation_time", "source_timeline_frame", "ready_at")}


def require_clock(clock, solver, publication, before, extraction_started):
    """Independent ordering and literal state agreement, including a bad control."""
    assert clock is not None and clock.ready_at is not None
    assert before <= clock.ready_at <= publication.captured_at <= extraction_started
    assert extraction_started < publication.published_at
    assert publication.state_ready_at == clock.ready_at
    assert (clock.solver_frame, clock.state_revision, clock.simulation_time) == (
        solver.frame, solver.liquid_state_revision, solver.sim_time)
    assert clock.source_timeline_frame == publication.source_timeline_frame == 2
    assert publication.epoch == clock.epoch


def run_case(modules, device, report, path):
    bridge = modules("blender.scene_bridge")
    output = modules("blender.output_publication")
    publication = modules("blender.surface_publication")
    inputs = modules("blender.scene_inputs")
    access = modules("blender.scene_access")
    wp = modules("runtime.warp_runtime")._import_warp()
    origin = (3.0, -2.0, 1.0)
    scene, _ = surface_fixture.make_scene(modules, origin)
    solver = surface_fixture.make_solver(
        modules, device, origin, waves=True,
        cadence=modules("sim.enums").SurfaceUpdateCadence.FRAME)
    assert wp.get_device(solver.device).is_cuda, "The real solver must execute on CUDA"
    session = bridge.Session()
    session.install_solver(
        solver, profiler=None, device=solver.device, scene_inputs=inputs.SceneInputs(),
        solved_timeline_frame=1)
    session.bind_scene(access.scene_identity(scene), "fspt-clock-proof")
    report["module_sources"] = {name: {"path": str(Path(modules(name).__file__).resolve()),
                                      "sha256": base.sha(Path(modules(name).__file__))}
                                for name in ("blender.scene_bridge", "blender.surface_publication",
                                             "blender.output_publication", "sim.solver")}
    observed = []
    prepare = output.prepare_live_payload

    def delayed_publication(*args, **kwargs):
        entered = time.monotonic()
        clock = session.current_solver_completion(scene)
        assert clock is not None and clock.ready_at <= entered
        time.sleep(0.03)  # Test-only delay establishes a distinct optional boundary.
        result = prepare(*args, **kwargs)
        observed.append({"entered_at": entered, "returned_at": time.monotonic(),
                         "clock": facts(clock)})
        return result

    try:
        assert session.current_solver_completion(scene) is None, "Admission invented a solve clock"
        scene.frame_set(2)
        before = time.monotonic()
        with patch.object(output, "prepare_live_payload", delayed_publication):
            session.step(scene, write_cache=False, timeline_frame=2)
        completed = session.current_solver_completion(scene)
        stamp = publication.current_surface_publication(scene)
        assert len(observed) == 1
        require_clock(completed, solver, stamp, before, observed[0]["entered_at"])
        assert stamp.published_at - completed.ready_at >= 0.03
        # The ordering comparator must reject a publication-completion clock.
        from dataclasses import replace

        try:
            require_clock(replace(completed, ready_at=stamp.published_at), solver,
                          replace(stamp, state_ready_at=stamp.published_at),
                          before, observed[0]["entered_at"])
        except AssertionError:
            pass
        else:
            raise AssertionError("Publication time was accepted as solver completion")

        surface = output.output_object(scene, output.SURFACE_NAME, ensure=False)
        positions, triangles = surface_fixture.evaluated_surface(surface)
        assert len(triangles) and np.isfinite(positions).all()
        original = path.with_name(path.stem + "-surface.npz")
        assert not original.exists(), "Use a fresh evidence destination"
        np.savez(original, positions=positions, triangles=triangles)
        report["surface"] = {"vertices": len(positions), "triangles": len(triangles),
                             "original": str(original), "sha256": base.sha(original)}
        report["completed"] = facts(completed)
        report["publication"] = asdict(stamp)
        report["optional_work"] = observed

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Completion query performed GPU/readback/solver work")

        with (patch.object(solver, "step_frame", forbidden),
              patch.object(solver, "host_frame", forbidden),
              patch.object(wp.array, "numpy", forbidden),
              patch.object(wp, "synchronize_device", forbidden),
              patch.object(wp, "synchronize_stream", forbidden),
              patch.object(wp, "synchronize_event", forbidden)):
            for _ in range(10):
                assert session.current_solver_completion(scene) is completed
        time.sleep(0.03)
        output.update_outputs(scene, session, force_surface=True)
        refreshed = publication.current_surface_publication(scene)
        assert refreshed.generation > stamp.generation
        assert refreshed.state_ready_at == completed.ready_at
        assert session.current_solver_completion(scene) is completed
        report["forced_publication"] = asdict(refreshed)

        # A repeated timeline label with a new actual solve receives a new clock.
        session.step(scene, write_cache=False, timeline_frame=2, publish_outputs=False)
        repeated = session.current_solver_completion(scene)
        assert repeated.ready_at > completed.ready_at and repeated.solver_frame > completed.solver_frame
        assert repeated.state_revision > completed.state_revision
        assert repeated.source_timeline_frame == 2
        report["same_timeline_new_solve"] = facts(repeated)

        # Actual external stepping does not receive the Session's former receipt.
        solver.step_frame()
        assert session.current_solver_completion(scene) is None
        output.update_outputs(scene, session, force_surface=True)
        assert publication.current_surface_publication(scene).state_ready_at is None
        report["direct_step_clock_rejected"] = True

        session.step(scene, write_cache=False, timeline_frame=2, publish_outputs=False)
        assert session.current_solver_completion(scene) is not None
        with patch.object(solver, "step_frame", side_effect=RuntimeError("injected failed solve")):
            try:
                session.step(scene, write_cache=False, timeline_frame=2, publish_outputs=False)
            except RuntimeError as exc:
                assert str(exc) == "injected failed solve"
            else:
                raise AssertionError("Solve failure was swallowed")
        assert session.current_solver_completion(scene) is None
        report["failed_step_clock_rejected"] = True
        report["read_only_query_control"] = True
        report["wrong_clock_control"] = True
    finally:
        session._clear_runtime()
        assert session.current_solver_completion(scene) is None


def main():
    args = base.arguments("artifacts/fspt-p0/state-clock.json")
    path = Path(args.json_out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "verifier_sha256": base.sha(Path(__file__)),
              "fixture_sha256": base.sha(Path(surface_fixture.__file__)),
              "scope": "authoritative Session completion boundary before optional publication",
              "not_qualified": ["measured realtime freshness", "renderer throughput",
                                "joint overhead/quality gate", "installed route"]}
    addon = None
    try:
        addon = base.load_addon(args, report)
        report["runtime"] = base.runtime_info(addon)
        assert report["runtime"]["warp"] == "1.19.0"

        def modules(name):
            return importlib.import_module(addon.__name__ + "." + name)

        base.checkpoint(path, report, "completed-state")
        run_case(modules, base.script_device(), report, path)
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        try:
            if addon is not None:
                addon.unregister()
        except BaseException:
            report.update(status="FAILED", cleanup_error=traceback.format_exc())
            raise
        finally:
            base.write_report(path, report)
    print("FSPT-STATE-CLOCK-PASS", path, flush=True)


if __name__ == "__main__":
    main()
