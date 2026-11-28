"""Native two-view proof must compare pixels, not only camera labels."""
import argparse
import importlib.util
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from threading import local
from types import ModuleType, SimpleNamespace
from weakref import WeakSet

import numpy as np
import pytest


def harness():
    path = Path(__file__).resolve().parents[1] / "acceptance/blender_fspt_async_viewport_acceptance.py"
    spec = importlib.util.spec_from_file_location("async_view_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_view_oracle_rejects_identical_images_with_different_dimensions():
    with pytest.raises(AssertionError, match="different completed pixels"):
        harness().compare_view_pixels(np.ones((80, 100, 4)), np.ones((80, 101, 4)))


def test_two_view_oracle_reports_actual_rgb_change_on_common_grid():
    first, second = np.ones((80, 100, 4)), np.ones((80, 101, 4))
    first[:, :50, :3] = 0
    assert harness().compare_view_pixels(first, second) == pytest.approx(0.5)


def test_lifecycle_count_cli_survives_blenders_shared_cycles_argument_parser(monkeypatch, tmp_path):
    # Blender 5.2's installed cycles/engine.py parses this same post-"--"
    # tail with abbreviations enabled, before running the user script.
    # A bare --cycles dies there as an ambiguous prefix of these two options.
    tail = ["--json-out", str(tmp_path / "report.json"), "--lifecycle-cycles", "7"]
    cycles = argparse.ArgumentParser(add_help=False)
    cycles.add_argument("--cycles-print-stats", action="store_true")
    cycles.add_argument("--cycles-device")
    parsed, untouched = cycles.parse_known_args(tail)
    assert untouched == tail and not parsed.cycles_print_stats and parsed.cycles_device is None
    monkeypatch.setattr(sys, "argv", ["blender", "--", *tail])
    args = harness().arguments()
    assert args.cycles == 7 and args.json_out == tmp_path / "report.json"


@pytest.fixture
def instrumented_callbacks(monkeypatch):
    module = harness()
    run = module.AsyncViewportAcceptance.__new__(module.AsyncViewportAcceptance)
    run._phases = local()
    run.patchers, run.calls, run.times = [], Counter(), defaultdict(list)
    run.stage = "initial"
    run.backends, run.retirements = WeakSet(), []
    run.maximum_pending = run.maximum_capacity = run.maximum_active_backends = 0
    observed = []

    class Engine:
        def view_update(self, context, depsgraph):
            observed.append((run.phase, context, depsgraph))
            return "updated"

        def view_draw(self, context, depsgraph):
            observed.append((run.phase, context, depsgraph))
            if getattr(depsgraph, "fail", False):
                raise ValueError("draw control")
            return "drawn"

        def render(self, depsgraph):
            observed.append((run.phase, None, depsgraph))
            return "rendered"

    class Backend:
        def __init__(self):
            self.closed, self._capacity, self._job = False, 1, None
            self.completed = object()

        @property
        def pending(self):
            return self._job is not None

        def submit(self, *, final=False):
            self._job = SimpleNamespace(lease=SimpleNamespace(final=final))
            return True

        def collect(self, *, wait=False):
            self._job = None
            return self.completed

        def close(self):
            self.closed, self._capacity, self._job = True, 0, None

    run.engine = SimpleNamespace(FSPTRenderEngine=Engine)
    run.backend_module = SimpleNamespace(AsyncCudaBackend=Backend)
    run.addon = SimpleNamespace(__name__="callback_contract_addon")
    extraction = ModuleType("callback_contract_addon.blender.fspt_scene")
    extraction.extract_scene = lambda *args, **kwargs: None
    runtime = ModuleType("callback_contract_addon.runtime.warp_runtime")
    runtime._import_warp = lambda: SimpleNamespace(synchronize=lambda: None,
                                                   synchronize_device=lambda: None)
    monkeypatch.setitem(sys.modules, extraction.__name__, extraction)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    run.instrument()
    try:
        yield run, Engine, observed
    finally:
        for guard in reversed(run.patchers):
            guard.stop()


@pytest.mark.parametrize("name,phase,result", [
    ("view_update", "update", "updated"),
    ("view_draw", "draw", "drawn"),
    ("render", "final", "rendered"),
])
def test_native_rna_callback_arity_preserves_arguments(instrumented_callbacks, name, phase, result):
    run, engine_type, observed = instrumented_callbacks
    context, graph = object(), object()
    supplied = (graph,) if name == "render" else (context, graph)
    callback = getattr(engine_type, name)
    # Blender's RNA dispatcher uses the code object's explicit positional
    # count. A variadic wrapper declaring only `instance` receives no inputs.
    declared_inputs = callback.__code__.co_argcount - 1
    assert callback(engine_type(), *supplied[:declared_inputs]) == result
    assert observed == [(phase, None if name == "render" else context, graph)]
    assert run.phase is None and run.calls[phase] == 1 and len(run.times[phase]) == 1


def test_native_callback_failure_restores_prior_audit_phase(instrumented_callbacks):
    run, engine_type, observed = instrumented_callbacks
    run.phase = "outer"
    context, graph = object(), SimpleNamespace(fail=True)
    with pytest.raises(ValueError, match="draw control"):
        engine_type().view_draw(context, graph)
    assert observed == [("draw", context, graph)]
    assert run.phase == "outer" and run.calls["draw"] == 1 and len(run.times["draw"]) == 1


def transition_run(stage):
    """A completed, idle pair of views at the real timer-transition seam."""
    module = harness()
    run = module.AsyncViewportAcceptance.__new__(module.AsyncViewportAcceptance)
    run.stage, run.started = stage, time.monotonic()
    run.args = SimpleNamespace(deadline=360, cycles=50)
    run.completed_cycles = 0
    run.scene = SimpleNamespace(fspt=SimpleNamespace(viewport_samples=2),
                                render=SimpleNamespace(engine="FSPT"))
    run.engine = SimpleNamespace(viewport_observations=lambda: [])
    run.check_resources = lambda: None
    run.save_views = lambda *args: {}
    run.checkpoint = lambda stage: setattr(run, "stage", stage)
    frame = SimpleNamespace(key=("new",), pixels=np.ones((2, 2, 4)))
    run.ready_views = lambda count: {1: ({}, frame), 2: ({}, frame)}
    run.prior_keys = {1: ("old",), 2: ("old",)}
    run.prior_world = {1: np.zeros((2, 2, 4)), 2: np.zeros((2, 2, 4))}
    redraws = []
    areas = [SimpleNamespace(spaces=SimpleNamespace(active=SimpleNamespace(
        shading=SimpleNamespace(type="RENDERED"))), tag_redraw=lambda: redraws.append(
            (run.stage, run.scene.fspt.viewport_samples))) for _ in range(2)]
    run.areas = lambda: areas
    return run, redraws


@pytest.mark.parametrize("stage,next_stage", [
    ("world_reassignment", "final_pending"),
    ("after_final", "cycle_pending"),
    ("cycle_reopen", "cycle_pending"),
])
def test_scripted_refinement_wakes_every_idle_view(stage, next_stage):
    run, redraws = transition_run(stage)
    assert run.tick() == 0.01
    # A property assignment alone does not wake Blender's idle event/draw path.
    # Arm the pending observation before either draw can submit its next job.
    assert run.stage == next_stage and run.scene.fspt.viewport_samples == 64
    assert redraws == [(next_stage, 64), (next_stage, 64)]


def test_scripted_retirement_requests_real_view_redraw():
    run, redraws = transition_run("cycle_pending")
    run.backends = [SimpleNamespace(closed=False, pending=True)]
    run.retire_cycle()
    assert run.stage == "cycle_retire" and run.scene.render.engine == "BLENDER_WORKBENCH"
    assert all(area.spaces.active.shading.type == "SOLID" for area in run.areas())
    assert len(redraws) == 2


@pytest.mark.parametrize("stage", ["final_pending", "cycle_pending"])
def test_pending_observer_retains_real_job_but_allows_final_drain(instrumented_callbacks, stage):
    run, _, _ = instrumented_callbacks
    run.stage = stage
    backend = run.backend_module.AsyncCudaBackend()
    assert backend.submit()
    assert backend.collect(wait=False) is None
    assert backend.pending and run.calls["pending_observation_hold"] == 1
    run.phase = "final"
    assert backend.collect(wait=True) is backend.completed
    assert not backend.pending and run.calls["final_viewport_owned_wait"] == 1


def test_pending_observer_does_not_mask_teardown(instrumented_callbacks):
    run, _, _ = instrumented_callbacks
    run.stage = "cycle_pending"
    backend = run.backend_module.AsyncCudaBackend()
    backend.submit()
    assert backend.collect(wait=False) is None
    backend.close()
    assert backend.closed and not backend.pending and backend._capacity == 0


def test_final_render_waits_do_not_count_as_viewport_drain(instrumented_callbacks):
    run, _, _ = instrumented_callbacks
    run.phase = "final"
    backend = run.backend_module.AsyncCudaBackend()
    backend.submit(final=True)
    assert backend.collect(wait=True) is backend.completed
    assert run.calls["final_owned_wait"] == 1
    assert run.calls["final_viewport_owned_wait"] == 0


def test_ordinary_refinement_collection_is_not_held(instrumented_callbacks):
    run, _, _ = instrumented_callbacks
    backend = run.backend_module.AsyncCudaBackend()
    backend.submit()
    assert backend.collect(wait=False) is backend.completed
    assert not backend.pending and run.calls["pending_observation_hold"] == 0
