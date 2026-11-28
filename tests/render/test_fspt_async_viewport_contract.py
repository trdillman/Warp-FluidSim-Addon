"""Native two-view proof must compare pixels, not only camera labels."""
import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path
from threading import local
from types import ModuleType, SimpleNamespace

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
        def submit(self): pass
        def collect(self): pass
        def close(self): pass

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
