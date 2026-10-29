"""Verifier regression controls on real CPU arrays; never CUDA certification.

Loads only the verifier functions by AST so no Blender runtime is impersonated.
Run with a Python environment containing real Warp. CPU memory observations are
explicitly unavailable; no device-memory assertion is exercised by these tests.
"""
import ast
import gc
from pathlib import Path
import sys
import traceback
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import warp as wp
from fluidSim_warp_blender51.sim.solver import FluidSolver as ProductionSolver, SolverSettings
from fluidSim_warp_blender51.sim.particles import ParticleSystem, ParticleBins
from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem
from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes

VERIFIER = ROOT / 'tests/acceptance/blender_allocation_accounting_acceptance.py'

class FluidSolver:
    """Verifier-only owner fixture with real component arrays and growth method.

    The real solver correctly refuses CPU. Do not bypass its CUDA guard.
    This fixture tests verifier lifecycle behavior, not solver initialization.
    """
    __module__ = ProductionSolver.__module__
    ensure_particle_capacity = ProductionSolver.ensure_particle_capacity
    def __init__(self, settings):
        from types import SimpleNamespace
        self.settings, self.device = settings, 'cpu'
        self.grid = SimpleNamespace(voxel_capacity=512)
        self.particles = ParticleSystem(settings.max_particles, 'cpu')
        self.bins = ParticleBins(settings.max_particles, 512, 'cpu')
        self.whitewater = WhitewaterSystem(settings.max_whitewater, 'cpu',
                                         candidate_capacity=settings.max_particles)
        self.narrow_band = None
        self.sourcing = None
        self._host_frame_cache = None
        self.last_particle_growth = None
    def release(self):
        pass

tree = ast.parse(VERIFIER.read_text())
functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
namespace = dict(wp=wp, gc=gc, traceback=traceback, weakref=weakref,
                 FluidSolver=FluidSolver, estimate_solver_bytes=estimate_solver_bytes)
exec(compile(functions, str(VERIFIER), 'exec'), namespace)
namespace['memory_sample'] = lambda device: {'observation': 'NOT_AVAILABLE_ON_CPU'}

class AcceptanceControls(unittest.TestCase):
    def setUp(self):
        self.settings = SolverSettings(domain_min=(0.,0.,0.), domain_max=(.8,.8,.8),
            voxel_size=.1, leaf_capacity=1, max_particles=32, max_whitewater=16,
            surface_enabled=False, device='cpu')
        self.saved = []
        # CPU cannot provide a CUDA budget. This synthetic budget only lets the
        # actual replacement implementation execute on CPU for verifier tests.
        self.budget = patch('fluidSim_warp_blender51.sim.memory_safety.vram_snapshot',
            return_value={'gpu_total_bytes': 16*1024**3, 'gpu_free_bytes': 8*1024**3})
        self.budget.start()

    def tearDown(self):
        self.budget.stop()
        self.saved.clear()
        gc.collect()

    def run_case(self):
        return namespace['run_case']('cpu_verifier_control', self.settings, 'cpu')

    def test_good_case(self):
        result = self.run_case()
        self.assertEqual(result['status'], 'PASS', result.get('error'))

    def test_initial_underestimate_is_rejected(self):
        with patch.dict(namespace, estimate_solver_bytes=lambda _: 0):
            result = self.run_case()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('AssertionError', result['error'])

    def test_retained_owned_array_is_rejected(self):
        original = FluidSolver.release
        def leak(owner):
            self.saved.append(owner.particles.position)
            original(owner)
        with patch.object(FluidSolver, 'release', leak):
            result = self.run_case()
        self.assertEqual(result['status'], 'FAIL', 'retained owned array falsely accepted')
        self.assertIn('owned Warp arrays retained', result['error'])

    def test_release_exception_is_reported(self):
        original = FluidSolver.release
        def broken(owner):
            original(owner)
            raise RuntimeError('release-control-sentinel')
        with patch.object(FluidSolver, 'release', broken):
            result = self.run_case()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('release-control-sentinel', result['error'])

    def test_growth_underestimate_is_rejected(self):
        from dataclasses import replace
        from fluidSim_warp_blender51.sim import particle_capacity
        original = particle_capacity.plan_particle_growth
        def undercount(*args, **kwargs):
            return replace(original(*args, **kwargs), peak_allocation_bytes=0)
        with patch.object(particle_capacity, 'plan_particle_growth', undercount):
            result = self.run_case()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('AssertionError', result['error'])

    def test_final_memory_sample_failure_is_reported(self):
        samples = 0
        def sample(device):
            nonlocal samples
            samples += 1
            # before, initial, replacement boundary, retained old, release
            if samples == 5:
                raise RuntimeError('final-memory-control-sentinel')
            return {'observation': 'NOT_AVAILABLE_ON_CPU'}
        with patch.dict(namespace, memory_sample=sample):
            result = self.run_case()
        self.assertEqual(samples, 5)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('final-memory-control-sentinel', result['error'])

    def test_budget_rejection_preserves_storage_without_candidate_allocation(self):
        owner = FluidSolver(self.settings)
        original = (owner.particles, owner.bins, owner.whitewater.candidates)
        with patch('fluidSim_warp_blender51.sim.memory_safety.vram_snapshot',
                   return_value={'gpu_total_bytes': 1024, 'gpu_free_bytes': 1}), \
             patch.object(owner.whitewater, 'allocate_candidate_workspace',
                          side_effect=AssertionError('must not allocate')) as allocate:
            result = owner.ensure_particle_capacity(33)
        self.assertEqual(result.status, 'BLOCKED')
        self.assertEqual(original, (owner.particles, owner.bins, owner.whitewater.candidates))
        self.assertEqual(owner.settings.max_particles, 32)
        allocate.assert_not_called()

    def test_precommit_failure_preserves_contents_and_allows_retry(self):
        import numpy as np
        owner = FluidSolver(self.settings)
        owner.particles.position.fill_(wp.vec3(1., 2., 3.))
        original = (owner.particles, owner.bins, owner.whitewater.candidates)
        content = owner.particles.position.numpy().copy()
        with patch.object(owner.whitewater, 'allocate_candidate_workspace',
                          side_effect=RuntimeError('precommit-control-sentinel')) as allocate:
            result = owner.ensure_particle_capacity(33)
        self.assertEqual(result.status, 'FAILED')
        self.assertIn('precommit-control-sentinel', result.reason)
        allocate.assert_called_once()
        self.assertEqual(original, (owner.particles, owner.bins, owner.whitewater.candidates))
        np.testing.assert_array_equal(content, owner.particles.position.numpy())
        self.assertEqual(owner.settings.max_particles, 32)
        self.assertTrue(owner.ensure_particle_capacity(33).grew)

if __name__ == '__main__':
    unittest.main(verbosity=2)
