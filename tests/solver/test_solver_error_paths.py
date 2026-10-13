"""REVIEW-027: the solver's error paths, each asserted by the message it gives.

Four paths had no test at all, so a rewrite could swap an actionable message
for a bare ``RuntimeError`` (or drop the check) and nothing would fail:

1. a device string Warp cannot resolve,
2. a device that resolves but is not CUDA,
3. a native reduction launch that returns a CUDA error code,
4. a sourcing transaction whose device-reported counts disagree with the host.

Every test drives the real error-handling code. Nothing here needs a GPU: the
device tests use Warp's own device lookup (``"cpu"`` always exists, a made-up
name never does), and the two tests that need a launch to fail substitute only
the native entry point or the device-state readback, so the checks and messages
under test are the production ones.

Each message is checked for what a user can act on: what is wrong, the value
that caused it, and (where there is one) what to do about it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import cuda_reduction, source_execution  # noqa: E402
from fluidSim_warp_blender51.sim import solver as solver_module  # noqa: E402
from fluidSim_warp_blender51.sim.cuda_reduction import CudaDotProduct  # noqa: E402
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings  # noqa: E402
from fluidSim_warp_blender51.sim.source_contracts import DirectInjectionPacket  # noqa: E402
from fluidSim_warp_blender51.sim.source_execution import CompiledSourceExecution  # noqa: E402

REMEDY = "Set a valid CUDA device (e.g. 'cuda:0') in Add-on Preferences."


# --------------------------------------------------------------------------
# 1 + 2. FluidSolver rejects an unusable device before it allocates anything
# --------------------------------------------------------------------------

@pytest.fixture
def no_allocation(monkeypatch):
    """Fail the test if the solver gets past the device check.

    ``SparseGrid`` is the first allocation after the check. Reaching it means
    the solver accepted the device, which is what the check exists to prevent
    (the old symptom was a bare "Must be a CUDA device" from deep inside the
    first pressure solve).
    """
    def allocated(*_args, **_kwargs):
        raise AssertionError("the solver allocated GPU state on an unusable device")

    monkeypatch.setattr(solver_module, "SparseGrid", allocated)


@pytest.mark.parametrize("device", ["nonsense", "cuda:99"])
def test_unresolvable_device_string_names_the_value_and_the_fix(device, no_allocation):
    with pytest.raises(RuntimeError) as caught:
        FluidSolver(SolverSettings(device=device))

    message = str(caught.value)
    assert message.startswith(f"FluidSim: invalid Warp device {device!r} (")
    assert REMEDY in message
    # Warp's own reason is kept, both in the text and as the chained cause.
    assert caught.value.__cause__ is not None
    assert str(caught.value.__cause__) in message


def test_non_cuda_device_says_why_cpu_cannot_work_and_how_to_fix_it(no_allocation):
    with pytest.raises(RuntimeError) as caught:
        FluidSolver(SolverSettings(device="cpu"))

    message = str(caught.value)
    assert message.startswith("FluidSim requires a CUDA device; got 'cpu'.")
    assert "no CPU-resident pressure-projection path" in message
    assert REMEDY in message
    assert "invalid Warp device" not in message  # the two failures stay distinct


# --------------------------------------------------------------------------
# 3. The native reduction reports a failed launch with the CUDA result
# --------------------------------------------------------------------------

def _array(size=64):
    return SimpleNamespace(ptr=0x1000, size=size)


def _reduction(monkeypatch, result, arity):
    """A ``CudaDotProduct`` whose native launch returns ``result``.

    ``arity`` selects the Warp 1.15 (8 argument) or 1.16+ (11 argument) ABI.
    ``launch`` and ``_launch`` are the real methods.
    """
    calls = []

    def native_launch(*args):
        calls.append(args)
        return result

    native_launch.argtypes = [None] * arity
    monkeypatch.setattr(
        cuda_reduction.context,
        "runtime",
        SimpleNamespace(core=SimpleNamespace(wp_cuda_launch_kernel=native_launch)),
    )
    dot = CudaDotProduct.__new__(CudaDotProduct)
    dot.device = SimpleNamespace(context=1, stream=SimpleNamespace(cuda_stream=2))
    dot._kernel = 3
    dot._active_kernel = 4
    dot._max_blocks = 4
    dot._partials = SimpleNamespace(ptr=48)
    dot._ticket = SimpleNamespace(ptr=64)
    return dot, calls


def _launch(dot, entry):
    if entry == "launch":
        dot.launch(_array(), _array(), _array(1), 64)
    else:
        dot.launch_active(_array(), _array(), _array(1), _array(1), 64)


@pytest.mark.parametrize("arity", [8, 11])
@pytest.mark.parametrize("entry", ["launch", "launch_active"])
def test_a_failed_native_launch_raises_with_the_cuda_result(monkeypatch, entry, arity):
    dot, calls = _reduction(monkeypatch, result=700, arity=arity)

    with pytest.raises(RuntimeError) as caught:
        _launch(dot, entry)

    assert len(calls) == 1  # the failure comes from the launch, not a pre-check
    message = str(caught.value)
    assert "Native CUDA dot-product launch failed" in message
    assert message.endswith(": 700")  # the CUDA result code, for diagnosis


@pytest.mark.parametrize("entry", ["launch", "launch_active"])
def test_a_successful_native_launch_does_not_raise(monkeypatch, entry):
    """The control for the test above: result 0 is success, not an error."""
    dot, calls = _reduction(monkeypatch, result=0, arity=8)

    _launch(dot, entry)

    assert len(calls) == 1


def test_reduction_refuses_a_cpu_device_before_loading_a_kernel(monkeypatch):
    def loaded(*_args):
        raise AssertionError("the CPU device reached the CUDA kernel loader")

    monkeypatch.setattr(cuda_reduction, "_acquire_kernel", loaded)

    with pytest.raises(RuntimeError, match=r"^CudaDotProduct requires a CUDA device$"):
        CudaDotProduct("cpu")


# --------------------------------------------------------------------------
# 4. Sourcing: counts that disagree are an error, never silently accepted
# --------------------------------------------------------------------------

class _DeviceState:
    """A device array double: ``numpy()`` is the readback the checks parse."""

    def __init__(self, values):
        self.values = values

    def zero_(self):
        pass

    def numpy(self):
        return np.asarray(self.values, dtype=np.int64)


CAPACITY = 100
HOST_COUNT = 10  # particles the host believes are alive before the attempt


def _transaction(monkeypatch, state):
    """Run the real ``_direct_packet_attempt`` against a scripted readback.

    ``state`` is ``[demand, admitted, emitted, invariant, available]``, the
    layout the emission kernels write. The kernel launches are no-ops, so the
    only thing that varies is what the device "reports", and the host-side
    consistency checks that follow are the production ones.
    """
    monkeypatch.setattr(source_execution, "fast_launch", lambda *_a, **_k: None)
    particles = SimpleNamespace(
        capacity=CAPACITY,
        count=object(),
        next_uid=1,
        refresh_host_count=lambda: HOST_COUNT,
        has_optional_channel=lambda _name: False,
    )
    pipeline = SimpleNamespace(
        runtime=SimpleNamespace(
            particles=particles,
            invalidate_host_cache=lambda: None,
            invalidate_particle_surface=lambda: None,
        ),
        device="cuda:0",
        _emission_state=_DeviceState(state),
        _compiled_result=_DeviceState([]),
        _compiled_empty_scalar=object(),
    )
    return CompiledSourceExecution._direct_packet_attempt(pipeline, (), 3)


def test_a_consistent_device_report_is_admitted(monkeypatch):
    """The control: with matching counts the same call succeeds."""
    result = _transaction(monkeypatch, [3, 1, 3, 0, CAPACITY - HOST_COUNT])

    assert (result.admitted, result.emitted, result.required) == (True, 3, 3)


def test_a_drifting_initial_count_is_reported_as_a_transaction_error(monkeypatch):
    # The device says 20 particles were alive; the host counted 10.
    with pytest.raises(RuntimeError, match=r"direct particle transaction initial count drifted"):
        _transaction(monkeypatch, [3, 1, 3, 0, CAPACITY - 20])


@pytest.mark.parametrize(
    "state",
    [
        pytest.param([3, 1, 2, 0, CAPACITY - HOST_COUNT], id="emitted-fewer-than-demanded"),
        pytest.param([3, 1, 3, 1, CAPACITY - HOST_COUNT], id="invariant-flag-set"),
    ],
)
def test_emitting_a_different_count_than_admitted_is_a_transaction_error(monkeypatch, state):
    with pytest.raises(RuntimeError, match=r"direct particle transaction violated its admitted count"):
        _transaction(monkeypatch, state)


def test_mismatched_velocity_and_position_counts_are_rejected_when_the_packet_is_built():
    positions = np.zeros((3, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"per-particle velocities must match positions"):
        DirectInjectionPacket(positions=positions, velocities=velocities)

    # Matching counts are fine, so the rejection is about the mismatch alone.
    DirectInjectionPacket(positions=positions, velocities=np.zeros((3, 3), dtype=np.float32))
