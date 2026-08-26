"""Species composition contracts across every particle birth transaction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender.sources import PreparedParticleSeed
from fluidSim_warp_blender51.sim.enums import (
    SourceEmissionMode,
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan,
    MeshRegionPlan,
    PayloadPlan,
    PointRegionPlan,
    SourceKey,
    SourcePlan,
)
from fluidSim_warp_blender51.sim.sourcing import (
    DirectInjectionPacket,
    MeshSourceDesc,
    SourceShapeDesc,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings


DYE_A = (0.25, 0.5, 0.0, 0.0)
DYE_B = (0.6, 0.1, 0.0, 0.0)


def _settings(*, capacity=32, auto_grow=True):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.25,
        particles_per_voxel=8.0,
        max_particles=capacity,
        species_enabled=True,
        surface_enabled=False,
        whitewater_enabled=False,
        auto_grow_particles=auto_grow,
        device="cuda:0",
    )
    return settings


def _expected_volume(solver):
    return solver.settings.voxel_size ** 3 / solver.quality.rest_particle_count


def test_every_liquid_source_record_has_additive_composition_default():
    assert PayloadPlan().species_concentration == (0.0, 0.0, 0.0, 0.0)
    assert SourceShapeDesc().species_concentration == (0.0, 0.0, 0.0, 0.0)
    assert MeshSourceDesc().species_concentration == (0.0, 0.0, 0.0, 0.0)
    assert DirectInjectionPacket([[0, 0, 0]]).species_concentration == (
        0.0, 0.0, 0.0, 0.0)


def test_region_plan_families_carry_composition_through_payload():
    regions = (
        AnalyticRegionPlan(),
        MeshRegionPlan(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.asarray(((0, 1, 2),), np.int32),
        ),
        PointRegionPlan(np.asarray(((0.2, 0.2, 0.2),), np.float32)),
    )
    for index, region in enumerate(regions):
        plan = SourcePlan(
            SourceKey(f"species-region-{index}"),
            SourceOperation.LIQUID,
            SourceLifecycle.DIRECT_APPEND
            if isinstance(region, PointRegionPlan) else SourceLifecycle.TOP_UP,
            SourceEvaluationCadence.STATIC,
            region,
            PayloadPlan(
                kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                species_concentration=DYE_A,
            ),
        )
        assert plan.payload.species_concentration == DYE_A


def test_prepared_seed_schema_preserves_composition():
    seed = PreparedParticleSeed(
        stable_id="seed", positions=[[0.1, 0.2, 0.3]], velocity=(0, 0, 0),
        temperature=0.0, viscosity=0.0, density=1.0,
        color=(1, 1, 1, 1), source_id=7, life=0.0,
        use_lifespan=False, species_concentration=DYE_B,
    )
    assert seed.species_concentration == DYE_B


@pytest.mark.parametrize("composition", [
    (-0.1, 0.0, 0.0, 0.0),
    (0.8, 0.3, 0.0, 0.0),
    (0.2, 0.3, 0.1, 0.0),
    (float("nan"), 0.0, 0.0, 0.0),
])
def test_deep_direct_packet_seam_rejects_invalid_composition(composition):
    with pytest.raises(ValueError, match="species concentration"):
        DirectInjectionPacket(
            [[0.1, 0.1, 0.1]], species_concentration=composition)


def test_direct_birth_writes_volume_and_composition_atomically():
    solver = FluidSolver(_settings(capacity=4))
    try:
        result = solver.inject_particles_exact(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
            species_concentration=DYE_A,
        )
        assert result.admitted and result.emitted == 2
        np.testing.assert_allclose(
            solver.particles.attr("represented_volume").numpy()[:2],
            _expected_volume(solver),
        )
        np.testing.assert_allclose(
            solver.particles.attr("species_concentration").numpy()[:2],
            (DYE_A, DYE_A),
        )
    finally:
        solver.release()


def test_exact_fit_and_refused_batch_preserve_count_and_species_budget():
    solver = FluidSolver(_settings(capacity=2, auto_grow=False))
    try:
        exact = solver.inject_particles_exact(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
            species_concentration=DYE_A,
        )
        assert exact.admitted and exact.total == 2
        before_count = solver.particles.refresh_host_count()
        before_volume = solver.particles.attr("represented_volume").numpy().copy()
        before_species = solver.particles.attr("species_concentration").numpy().copy()

        refused = solver.inject_particle_packets_exact((
            DirectInjectionPacket(
                [[0.3, 0.3, 0.3]], species_concentration=DYE_B),
        ))

        assert not refused.admitted and refused.emitted == 0
        assert solver.particles.refresh_host_count() == before_count
        np.testing.assert_array_equal(
            solver.particles.attr("represented_volume").numpy(), before_volume)
        np.testing.assert_array_equal(
            solver.particles.attr("species_concentration").numpy(), before_species)
    finally:
        solver.release()


def test_species_disabled_direct_injection_keeps_old_payload_and_no_resources():
    settings = _settings(capacity=4)
    settings.species_enabled = False
    solver = FluidSolver(settings)
    try:
        result = solver.inject_particles_exact([[0.1, 0.1, 0.1]])
        assert result.admitted and result.emitted == 1
        assert not solver.particles.has_optional_channel("represented_volume")
        assert not solver.particles.has_optional_channel("species_concentration")
        assert not hasattr(solver, "species")
    finally:
        solver.release()


def test_source_receipt_transfer_does_not_cross_the_device_host_boundary(monkeypatch):
    solver = FluidSolver(_settings(capacity=4))
    try:
        solver.sourcing._reset_species_source_amount()
        solver.sourcing._species_source_amount.fill_(2.0)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("source receipt crossed the device/host boundary")

        monkeypatch.setattr(
            type(solver.sourcing._species_source_amount), "numpy", forbidden)

        solver.sourcing._record_species_sources()
    finally:
        solver.release()


def test_initial_fill_receipt_fires_once_and_reset_rearms():
    solver = FluidSolver(_settings(capacity=2048))
    source = SourceShapeDesc(
        shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.PARTICLES,
        position=(0.5, 0.5, 0.5), half_extents=(0.2, 0.2, 0.2),
        emit_rate=1.0, source_id="dye-fill",
        emission_mode=SourceEmissionMode.INITIAL_FILL,
        species_concentration=DYE_A,
    )
    try:
        solver.set_sources((source,))
        first = solver.step_frame().species_receipt
        second = solver.step_frame().species_receipt
        assert first.sources[0] > 0.0 and first.sources[1] > 0.0
        assert second.sources == pytest.approx((0.0, 0.0), abs=1e-7)
        solver.sourcing._species_source_amount.fill_(3.0)
        solver.reset()
        np.testing.assert_array_equal(
            solver.sourcing._species_source_amount.numpy(), (0.0, 0.0))
        solver.set_sources((source,))
        reset = solver.step_frame().species_receipt
        assert reset.sources == pytest.approx(first.sources, rel=1e-5)
    finally:
        solver.release()


def test_sink_receipt_closes_one_frame_conservation():
    solver = FluidSolver(_settings(capacity=16))
    try:
        solver.inject_particles_exact(
            [[0.5, 0.5, 0.5]], species_concentration=DYE_A)
        solver.set_sources((SourceShapeDesc(
            shape=SourceShapeShape.BOX,
            stype=SourceShapeType.SINK,
            mode=SourceShapeSourceMode.PARTICLES,
            position=(0.5, 0.5, 0.5), half_extents=(0.3, 0.3, 0.3),
            source_id="sink",
        ),))
        receipt = solver.step_frame().species_receipt
        assert receipt.sinks[0] > 0.0 and receipt.sinks[1] > 0.0
        assert receipt.residual == pytest.approx((0.0, 0.0), abs=1e-6)
    finally:
        solver.release()


def test_lifespan_removal_is_recorded_once_in_species_sink_budget():
    solver = FluidSolver(_settings(capacity=4))
    try:
        solver.inject_particles_exact(
            [[0.5, 0.5, 0.5]], life=0.01, use_lifespan=True,
            species_concentration=DYE_A)
        baseline = solver.species.finalize_frame_receipt()

        solver._lifecycle(0.02)
        receipt = solver.species.finalize_frame_receipt()

        assert solver.particles.refresh_host_count() == 0
        assert receipt.sinks == pytest.approx(baseline.after, abs=1e-7)
        assert receipt.after == pytest.approx((0.0, 0.0), abs=1e-7)
        assert receipt.residual == pytest.approx((0.0, 0.0), abs=1e-7)
    finally:
        solver.release()


def test_invalid_advection_state_removal_is_recorded_in_species_sink_budget():
    import warp as wp

    solver = FluidSolver(_settings(capacity=4))
    try:
        solver.inject_particles_exact(
            [[0.5, 0.5, 0.5]], species_concentration=DYE_B)
        baseline = solver.species.finalize_frame_receipt()
        wp.copy(
            solver.particles.position,
            wp.array([[float("nan"), 0.5, 0.5]], dtype=wp.vec3,
                     device=solver.device),
            count=1,
        )

        solver._sanitize_particles(0.01)
        solver._lifecycle(0.01)
        receipt = solver.species.finalize_frame_receipt()

        assert solver.particles.refresh_host_count() == 0
        assert receipt.sinks == pytest.approx(baseline.after, abs=1e-7)
        assert receipt.after == pytest.approx((0.0, 0.0), abs=1e-7)
        assert receipt.residual == pytest.approx((0.0, 0.0), abs=1e-7)
    finally:
        solver.release()


def test_species_lifecycle_accounts_removal_in_the_existing_sink_pass(monkeypatch):
    solver = FluidSolver(_settings(capacity=4))
    try:
        solver.inject_particles_exact(
            [[0.5, 0.5, 0.5]], species_concentration=DYE_A)

        def forbidden_extra_ledger_launch():
            raise AssertionError("species removal used an extra ledger launch")

        monkeypatch.setattr(
            solver.species, "record_removed_particles",
            forbidden_extra_ledger_launch, raising=False)
        solver._lifecycle(0.01)
    finally:
        solver.release()
