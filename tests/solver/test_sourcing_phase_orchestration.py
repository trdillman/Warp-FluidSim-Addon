from contextlib import contextmanager
import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim.solver import FluidSolver


def _solver(events, *, emitted=False, growth=False, band=False, fail=False):
    solver = SimpleNamespace(bins=object())
    @contextmanager
    def phase(name):
        events.append(('begin', name))
        try:
            yield
        finally:
            events.append(('end', name))
    solver._phase = phase
    solver._index_particles = lambda: events.append('index')
    def source(dt):
        events.append(('source', dt))
        if fail:
            raise ValueError('source failed')
        if growth:
            solver.bins = object()
        return emitted
    solver._source = source
    solver.narrow_band = SimpleNamespace(
        credit_interior_counts=lambda: events.append('credit'),
        release_interior_counts=lambda: events.append('release'),
    ) if band else None
    return solver


@pytest.mark.parametrize('emitted,growth,band,body', [
    (False, False, False, [('source', 0.125)]),
    (True, False, True, ['credit', ('source', 0.125), 'release', 'index']),
    (False, True, True, ['credit', ('source', 0.125), 'release', 'index']),
])
def test_sourcing_phase_indexes_emission_and_pool_growth(emitted, growth, band, body):
    events = []
    solver = _solver(events, emitted=emitted, growth=growth, band=band)
    FluidSolver._run_sourcing_phase(solver, 0.125)
    assert events == [
        ('begin', 'indexing'), 'index', ('end', 'indexing'),
        ('begin', 'sourcing'), *body, ('end', 'sourcing'),
    ]


def test_source_error_propagates_without_releasing_narrow_band_credit():
    events = []
    with pytest.raises(ValueError, match='source failed'):
        FluidSolver._run_sourcing_phase(_solver(events, band=True, fail=True), 0.125)
    assert events == [
        ('begin', 'indexing'), 'index', ('end', 'indexing'),
        ('begin', 'sourcing'), 'credit', ('source', 0.125), ('end', 'sourcing'),
    ]


def test_frame_invokes_sourcing_after_effect_and_before_p2g():
    tree = ast.parse(textwrap.dedent(inspect.getsource(FluidSolver.step_frame)))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'substep_index')
    statements = [ast.unparse(n) for n in loop.body]
    before = next(i for i, s in enumerate(statements) if 'FrameFeaturePhase.BEFORE_SOURCING' in s)
    phase = statements.index('self._run_sourcing_phase(dt)')
    p2g = next(i for i, s in enumerate(statements) if s.startswith("with self._phase('p2g')"))
    assert before < phase < p2g
