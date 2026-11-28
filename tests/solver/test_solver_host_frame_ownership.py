"""Host snapshot identity is a read-only public solver responsibility."""
from types import SimpleNamespace

from fluidSim_warp_blender51.sim.frame_output import HostFrame, host_output_operation_counts
from fluidSim_warp_blender51.sim.solver import FluidSolver


def _solver():
    solver = FluidSolver.__new__(FluidSolver)
    solver.frame = 7
    solver._host_frame_cache = None
    return solver


def test_identity_query_rejects_empty_foreign_and_replaced_same_frame_snapshots():
    solver = _solver()
    assert not solver.owns_host_frame(None)
    assert not solver.owns_host_frame(HostFrame(solver))
    current = solver.host_frame()
    assert solver.owns_host_frame(current)
    assert not solver.owns_host_frame(HostFrame(solver))
    other = _solver()
    assert not other.owns_host_frame(current)
    solver._host_frame_cache = HostFrame(solver)
    assert not solver.owns_host_frame(current)
    assert solver.owns_host_frame(solver._host_frame_cache)


def test_identity_query_rejects_stale_frame_detached_and_rebound_owner():
    solver = _solver()
    current = solver.host_frame()
    solver.frame += 1
    assert not solver.owns_host_frame(current)
    solver.frame -= 1
    current.solver = None
    assert not solver.owns_host_frame(current)
    current.solver = _solver()
    assert not solver.owns_host_frame(current)


def test_same_frame_state_edit_invalidates_owned_snapshot_without_reacquisition():
    solver = _solver()
    current = solver.host_frame()
    solver.mark_liquid_state_changed()
    assert current.frame == solver.frame
    assert not solver.owns_host_frame(current)
    assert solver._host_frame_cache is None


def test_identity_query_never_constructs_or_materializes_a_host_snapshot(monkeypatch):
    solver = _solver()
    current = solver.host_frame()
    current._values["untouched"] = sentinel = object()
    before = host_output_operation_counts()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ownership query materialized solver output")

    monkeypatch.setattr(HostFrame, "__init__", forbidden)
    monkeypatch.setattr(solver, "host_frame", forbidden)
    solver.surface = SimpleNamespace(mesh=forbidden)
    assert solver.owns_host_frame(current)
    solver._host_frame_cache = None
    assert not solver.owns_host_frame(current)
    assert not solver.owns_host_frame(None)
    assert solver._host_frame_cache is None
    assert current._values == {"untouched": sentinel}
    assert host_output_operation_counts() == before
