import numpy as np
import pytest
import warp as wp
from types import SimpleNamespace

from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.solver import SolverSettings
from fluidSim_warp_blender51.sim.species import SpeciesPalette, SpeciesTransport


SPECIES_CHANNELS = ("represented_volume", "species_concentration")


def _transport(particles):
    return SpeciesTransport(
        particles, SimpleNamespace(voxel_capacity=1, active_voxel_count=0),
        "cpu", enabled=True, diffusivity=0.0, max_diffusion_substeps=1,
        palette=SpeciesPalette.create(
            ids=("a", "b"),
            absorption_rgb=((0.1, 1.0, 2.0), (2.0, 1.0, 0.1)),
            reference_path_length=1.0))


def test_species_state_is_default_off_without_changing_old_inventory():
    particles = ParticleSystem(4, "cpu")

    assert particles.optional_channels == ()
    assert not particles.has_optional_channel("represented_volume")
    assert not particles.has_optional_channel("species_concentration")
    assert set(particles.vec4_attrs) == set()
    assert set(particles.float_attrs) == set()
    assert SolverSettings().species_enabled is False


def test_species_state_allocates_as_one_coupled_optional_inventory():
    particles = ParticleSystem(4, "cpu", SPECIES_CHANNELS)

    assert particles.optional_channels == SPECIES_CHANNELS
    assert particles.has_optional_channel("represented_volume")
    assert particles.has_optional_channel("species_concentration")
    assert particles.attr("represented_volume").dtype == wp.float32
    assert particles.attr("species_concentration").dtype == wp.vec4


@pytest.mark.parametrize("selection", [
    ("represented_volume",),
    ("species_concentration",),
])
def test_species_state_rejects_half_selected_inventory(selection):
    with pytest.raises(ValueError, match="allocated together"):
        ParticleSystem(4, "cpu", selection)


def test_species_state_resize_and_compaction_preserve_alignment():
    particles = ParticleSystem(4, "cpu", SPECIES_CHANNELS)
    wp.copy(particles.attr("represented_volume"), wp.array(
        np.asarray((0.25, 0.5, 0.75), np.float32), dtype=float, device="cpu"))
    wp.copy(particles.attr("species_concentration"), wp.array(
        np.asarray(((0.1, 0.2, 0.0, 0.0),
                    (0.3, 0.4, 0.0, 0.0),
                    (0.5, 0.25, 0.0, 0.0)), np.float32),
        dtype=wp.vec4, device="cpu"))
    wp.copy(particles.alive, wp.array((1, 0, 1, 0), dtype=wp.int32, device="cpu"))
    wp.copy(particles.count, wp.array((3,), dtype=wp.int32, device="cpu"))
    particles.host_count = 3

    grown = particles.resized_copy(8)
    grown.compact(count_hint=3)

    assert grown.refresh_host_count() == 2
    np.testing.assert_allclose(
        grown.attr("represented_volume").numpy()[:2], (0.25, 0.75))
    np.testing.assert_allclose(
        grown.attr("species_concentration").numpy()[:2],
        ((0.1, 0.2, 0.0, 0.0), (0.5, 0.25, 0.0, 0.0)))


def test_species_ledger_compaction_fuses_species_channels_and_keeps_peers_aligned(
        monkeypatch):
    """The specialized scatter owns species copies; peers retain generic copies."""
    from fluidSim_warp_blender51.sim import particles as particle_api
    from fluidSim_warp_blender51.sim.species import (
        AFTER, MAX_LOADING, MINIMUM, RECEIPT_SIZE, SINKS,
    )

    channels = (
        "represented_volume", "color", "species_concentration",
        "source_id", "uid")
    particles = ParticleSystem(4, "cpu", channels)
    wp.copy(particles.attr("represented_volume"), wp.array(
        np.asarray((0.25, 0.5, 0.75), np.float32), dtype=float, device="cpu"))
    wp.copy(particles.attr("species_concentration"), wp.array(
        np.asarray(((0.1, 0.2, 0.0, 0.0),
                    (0.3, 0.4, 0.0, 0.0),
                    (0.5, 0.25, 0.0, 0.0)), np.float32),
        dtype=wp.vec4, device="cpu"))
    colors = np.asarray(((1.0, 0.0, 0.0, 1.0),
                         (0.0, 1.0, 0.0, 1.0),
                         (0.0, 0.0, 1.0, 1.0)), np.float32)
    wp.copy(particles.attr("color"), wp.array(
        colors, dtype=wp.vec4, device="cpu"))
    wp.copy(particles.attr("source_id"), wp.array(
        (11, 22, 33), dtype=wp.int32, device="cpu"))
    wp.copy(particles.attr("uid"), wp.array(
        (101, 202, 303), dtype=wp.int32, device="cpu"))
    wp.copy(particles.alive, wp.array(
        (1, 0, 1, 0), dtype=wp.int32, device="cpu"))
    wp.copy(particles.count, wp.array((3,), dtype=wp.int32, device="cpu"))
    particles.host_count = 3
    transport = _transport(particles)
    launches = []
    original_launch = wp.launch

    def counted_launch(kernel, *args, **kwargs):
        launches.append(str(getattr(kernel, "key", kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", counted_launch)

    particles.compact(
        count_hint=3,
        species_sink_ledger=transport.compaction_sink_ledger())
    transport.capture_compacted_after()
    receipt = transport.finalize_frame_receipt()

    assert particles.refresh_host_count() == 2
    np.testing.assert_allclose(
        particles.attr("represented_volume").numpy()[:2], (0.25, 0.75))
    np.testing.assert_allclose(
        particles.attr("species_concentration").numpy()[:2],
        ((0.1, 0.2, 0.0, 0.0), (0.5, 0.25, 0.0, 0.0)))
    np.testing.assert_allclose(particles.attr("color").numpy()[:2], colors[[0, 2]])
    np.testing.assert_array_equal(
        particles.attr("source_id").numpy()[:2], (11, 33))
    np.testing.assert_array_equal(particles.attr("uid").numpy()[:2], (101, 303))
    assert receipt.sinks == pytest.approx((0.15, 0.2))
    assert receipt.after == pytest.approx((0.4, 0.2375))
    assert receipt.min_concentration == pytest.approx(0.1)
    assert receipt.max_loading == pytest.approx(0.75)
    assert receipt.residual == pytest.approx((0.0, 0.0), abs=1.0e-7)
    assert sum("species_compact_alive_scatter" in key for key in launches) == 1
    assert not any("compact_float_scatter" in key for key in launches)
    assert sum("compact_vec4_scatter" in key for key in launches) == 1


def test_stale_compaction_hint_does_not_recharge_dead_species_tail():
    """Repeated CFL upper bounds must not replace the authoritative device count."""
    from fluidSim_warp_blender51.sim.species import (
        AFTER, MAX_LOADING, MINIMUM, RECEIPT_SIZE, SINKS,
    )

    particles = ParticleSystem(4, "cpu", SPECIES_CHANNELS)
    wp.copy(particles.attr("represented_volume"), wp.array(
        (0.25, 0.5, 0.75), dtype=float, device="cpu"))
    wp.copy(particles.attr("species_concentration"), wp.array(
        ((0.1, 0.2, 0.0, 0.0),
         (0.3, 0.4, 0.0, 0.0),
         (0.5, 0.25, 0.0, 0.0)), dtype=wp.vec4, device="cpu"))
    wp.copy(particles.alive, wp.array(
        (1, 0, 1, 0), dtype=wp.int32, device="cpu"))
    wp.copy(particles.count, wp.array((3,), dtype=wp.int32, device="cpu"))
    particles.host_count = 3
    transport = _transport(particles)
    ledger = transport.compaction_sink_ledger()

    particles.compact(count_hint=3, species_sink_ledger=ledger)
    particles.compact(count_hint=3, species_sink_ledger=ledger)
    particles.compact(count_hint=3, species_sink_ledger=ledger)
    transport.capture_compacted_after()
    receipt = transport.finalize_frame_receipt()

    assert int(particles.count.numpy()[0]) == 2
    assert receipt.sinks == pytest.approx((0.15, 0.2), abs=1.0e-7)
    assert receipt.after == pytest.approx((0.4, 0.2375), abs=1.0e-7)
    np.testing.assert_allclose(
        particles.attr("represented_volume").numpy()[:2], (0.25, 0.75))
    np.testing.assert_allclose(
        particles.attr("species_concentration").numpy()[:2],
        ((0.1, 0.2, 0.0, 0.0), (0.5, 0.25, 0.0, 0.0)))


def test_postcompaction_capture_uses_device_count_and_ignores_poisoned_tail():
    particles = ParticleSystem(4, "cpu", SPECIES_CHANNELS)
    wp.copy(particles.attr("represented_volume"), wp.array(
        (0.25, 0.75, 1000.0, 1000.0), dtype=float, device="cpu"))
    wp.copy(particles.attr("species_concentration"), wp.array(
        ((0.1, 0.2, 0.0, 0.0),
         (0.5, 0.25, 0.0, 0.0),
         (100.0, 100.0, 0.0, 0.0),
         (100.0, 100.0, 0.0, 0.0)), dtype=wp.vec4, device="cpu"))
    wp.copy(particles.count, wp.array((2,), dtype=wp.int32, device="cpu"))
    particles.host_count = 4
    transport = _transport(particles)

    transport.capture_compacted_after()
    receipt = transport.finalize_frame_receipt()

    assert receipt.after == pytest.approx((0.4, 0.2375), abs=1.0e-7)
    assert receipt.min_concentration == pytest.approx(0.1)
    assert receipt.max_loading == pytest.approx(0.75)


def test_species_state_checkpoint_round_trip_preserves_both_channels():
    particles = ParticleSystem(3, "cpu", SPECIES_CHANNELS)
    volume = np.asarray((0.25, 0.5), np.float32)
    concentration = np.asarray(
        ((0.1, 0.2, 0.0, 0.0), (0.3, 0.4, 0.0, 0.0)), np.float32)
    wp.copy(particles.attr("represented_volume"), wp.array(
        volume, dtype=float, device="cpu"))
    wp.copy(particles.attr("species_concentration"), wp.array(
        concentration, dtype=wp.vec4, device="cpu"))

    arrays = {
        "particle_position": np.zeros((2, 3), np.float32),
        "particle_velocity": np.zeros((2, 3), np.float32),
        "particle_alive": np.ones((2,), np.int32),
        "particle_time_offset": np.zeros((2,), np.float32),
        **{
            f"particle_attr_{name}": np.zeros((2,), np.float32)
            for name in particles.ATTR_NAMES
        },
        **particles.checkpoint_arrays(2),
    }
    restored = ParticleSystem(3, "cpu", SPECIES_CHANNELS)

    restored.restore_checkpoint_prefix(arrays, 2)

    np.testing.assert_allclose(restored.attr("represented_volume").numpy()[:2], volume)
    np.testing.assert_allclose(
        restored.attr("species_concentration").numpy()[:2], concentration)


@pytest.mark.parametrize(("volume", "concentration", "message"), [
    (0.0, (0.2, 0.3, 0.0, 0.0), "positive"),
    (-1.0, (0.2, 0.3, 0.0, 0.0), "positive"),
    (1.0, (-0.1, 0.3, 0.0, 0.0), "nonnegative"),
    (1.0, (0.8, 0.3, 0.0, 0.0), "simplex"),
    (1.0, (0.2, 0.3, 0.1, 0.0), "unused lanes"),
])
def test_invalid_species_birth_state_is_rejected_before_count_mutation(
        volume, concentration, message):
    particles = ParticleSystem(4, "cpu", SPECIES_CHANNELS)
    particles.count.fill_(1)

    with pytest.raises(ValueError, match=message):
        particles.validate_species_birth_state(volume, concentration)

    assert int(particles.count.numpy()[0]) == 1


def test_solver_settings_fail_closed_for_invalid_species_configuration():
    with pytest.raises(ValueError, match="exactly 2"):
        SolverSettings(species_enabled=True, species_count=3)
    with pytest.raises(ValueError, match="diffusivity"):
        SolverSettings(species_enabled=True, species_diffusivity=-1.0)
    with pytest.raises(ValueError, match="absorption"):
        SolverSettings(
            species_enabled=True,
            species_absorption_rgb=((1.0, -0.1, 0.0), (0.0, 0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="reference path length"):
        SolverSettings(species_enabled=True, species_reference_path_length=0.0)
