"""Validate Phase-3 persistent channels using the supplied Warp package."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

VALIDATION_PACKAGE = "fluidsim_validation_source"


def _ensure_source_package(module_name: str) -> None:
    """Install a synthetic package namespace without executing Blender code."""

    from types import ModuleType

    if module_name in sys.modules:
        return
    if not module_name.isidentifier():
        raise ValueError(f"validation module must be a Python identifier: {module_name!r}")
    package = ModuleType(module_name)
    package.__file__ = str(ROOT / "__init__.py")
    package.__package__ = module_name
    package.__path__ = [str(ROOT)]
    sys.modules[module_name] = package


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _copy(target, values, dtype, wp, device, count=None):
    source = wp.array(values, dtype=dtype, device=device)
    wp.copy(target, source, count=len(values) if count is None else count)


def validate(expected_warp_root: Path, module_name: str = VALIDATION_PACKAGE) -> dict:
    import warp as wp

    wp.init()
    module_path = Path(wp.__file__).resolve()
    if expected_warp_root.resolve() not in module_path.parents:
        raise AssertionError(
            f"Warp loaded from {module_path}, expected under {expected_warp_root}"
        )
    if str(getattr(wp, "__version__", "")) != "1.16.0":
        raise AssertionError(f"unexpected Warp version {getattr(wp, '__version__', '')}")

    _ensure_source_package(module_name)
    frame_output = importlib.import_module(f"{module_name}.sim.frame_output")
    particles_module = importlib.import_module(f"{module_name}.sim.particles")
    sourcing_module = importlib.import_module(f"{module_name}.sim.sourcing")
    k_sample_particle_vorticity = frame_output.k_sample_particle_vorticity
    ParticleSystem = particles_module.ParticleSystem
    SourcePipeline = sourcing_module.SourcePipeline

    device = "cpu"
    particles = ParticleSystem(
        8, device, optional_channels=("color", "source_id", "uid")
    )
    owner = SimpleNamespace(
        device=device,
        particles=particles,
        grid=SimpleNamespace(voxel_capacity=1),
        _nvox=1,
        _host_frame_cache=None,
        settings=SimpleNamespace(
            auto_grow_particles=False,
            particle_growth_failure_policy="DEFER",
        ),
    )
    pipeline = SourcePipeline(owner)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float32,
    )
    injected = pipeline.inject_particles_exact(
        positions,
        velocities=velocities,
        temperature=0.25,
        viscosity=1.5,
        density=0.9,
        color=(0.2, 0.4, 0.8, 1.0),
        source_id=44,
        life=3.0,
        use_lifespan=True,
    )
    if not injected.admitted or injected.emitted != 3 or injected.total != 3:
        raise AssertionError(f"unexpected injection result {injected}")
    np.testing.assert_allclose(particles.position.numpy()[:3], positions)
    np.testing.assert_allclose(particles.velocity.numpy()[:3], velocities)
    np.testing.assert_allclose(
        particles.vec4_attrs["color"].numpy()[:3],
        np.tile(np.array([[0.2, 0.4, 0.8, 1.0]], dtype=np.float32), (3, 1)),
    )
    np.testing.assert_array_equal(
        particles.int_attrs["source_id"].numpy()[:3], np.array([44, 44, 44])
    )
    np.testing.assert_array_equal(
        particles.int_attrs["uid"].numpy()[:3], np.array([1, 2, 3])
    )
    if particles.next_uid != 4:
        raise AssertionError("UID counter did not advance transactionally")

    # Kill the middle particle and prove every optional channel follows scan
    # compaction in exactly the same order.
    _copy(particles.alive, np.array([1, 0, 1], dtype=np.int32), wp.int32, wp, device)
    particles.compact(count_hint=3)
    if particles.refresh_host_count() != 2:
        raise AssertionError("particle compaction produced the wrong count")
    np.testing.assert_allclose(particles.position.numpy()[:2], positions[[0, 2]])
    np.testing.assert_array_equal(
        particles.int_attrs["uid"].numpy()[:2], np.array([1, 3])
    )
    np.testing.assert_array_equal(
        particles.int_attrs["source_id"].numpy()[:2], np.array([44, 44])
    )
    np.testing.assert_allclose(
        particles.vec4_attrs["color"].numpy()[:2],
        np.array([[0.2, 0.4, 0.8, 1.0], [0.2, 0.4, 0.8, 1.0]], dtype=np.float32),
    )

    checkpoint = {
        "particle_position": particles.position.numpy()[:2],
        "particle_velocity": particles.velocity.numpy()[:2],
        "particle_alive": particles.alive.numpy()[:2],
        "particle_time_offset": particles.time_offset.numpy()[:2],
    }
    for name in particles.ATTR_NAMES:
        checkpoint[f"particle_attr_{name}"] = particles.attrs[name].numpy()[:2]
    checkpoint.update(particles.checkpoint_arrays(2))
    restored = ParticleSystem(
        8, device, optional_channels=("color", "source_id", "uid")
    )
    restored.restore_checkpoint_prefix(checkpoint, 2)
    restored.next_uid = particles.next_uid
    np.testing.assert_array_equal(
        restored.int_attrs["uid"].numpy()[:2], np.array([1, 3])
    )
    np.testing.assert_allclose(
        restored.vec4_attrs["color"].numpy()[:2],
        particles.vec4_attrs["color"].numpy()[:2],
    )

    # Exercise the compatibility packet-normalization path used by prepared
    # Blender source seeds. Per-particle velocities must survive conversion to
    # DirectInjectionPacket rather than silently falling back to one velocity.
    compat_particles = ParticleSystem(4, device, optional_channels=("uid",))
    compat_owner = SimpleNamespace(
        device=device, particles=compat_particles,
        grid=SimpleNamespace(voxel_capacity=1), _nvox=1,
        _host_frame_cache=None, settings=owner.settings,
    )
    compat_pipeline = SourcePipeline(compat_owner)
    compat_velocity = np.array(
        [[3.0, 0.0, 0.0], [0.0, -2.0, 1.0]], dtype=np.float32
    )
    compat_packet = SimpleNamespace(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        velocity=(9.0, 9.0, 9.0), velocities=compat_velocity,
        temperature=0.0, viscosity=0.0, density=1.0,
        color=(1.0, 1.0, 1.0, 1.0), source_id=7,
        life=0.0, use_lifespan=False,
    )
    compat_result = compat_pipeline.inject_particle_packets_exact((compat_packet,))
    if not compat_result.admitted or compat_result.emitted != 2:
        raise AssertionError(f"compatibility packet injection failed: {compat_result}")
    np.testing.assert_allclose(
        compat_particles.velocity.numpy()[:2], compat_velocity, rtol=0.0, atol=0.0
    )

    disabled = ParticleSystem(4, device, optional_channels=())
    if disabled.optional_channels or disabled.has_optional_channel("color"):
        raise AssertionError("disabled product channels were allocated")

    # Compile and execute the Phase-3 vorticity kernel against a valid sparse
    # single-leaf zero field.  Zero curl is exact and device-independent.
    nvox = 8 ** 3
    field = wp.zeros(nvox, dtype=float, device=device)
    weight = wp.ones(nvox, dtype=float, device=device)
    leaf_table = wp.array([0], dtype=wp.int32, device=device)
    sample_positions = wp.array(
        np.array([[3.5, 3.5, 3.5], [4.0, 2.0, 5.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    vorticity = wp.empty(2, dtype=wp.vec3, device=device)
    wp.launch(
        k_sample_particle_vorticity,
        dim=2,
        inputs=[
            sample_positions,
            2,
            wp.vec3(0.0, 0.0, 0.0),
            1.0,
            field,
            field,
            field,
            weight,
            weight,
            weight,
            leaf_table,
            wp.vec3i(8, 8, 8),
            wp.vec3i(1, 1, 1),
        ],
        outputs=[vorticity],
        device=device,
    )
    wp.synchronize_device(device)
    np.testing.assert_array_equal(vorticity.numpy(), np.zeros((2, 3), dtype=np.float32))

    cuda_devices = [str(item.alias) for item in wp.get_devices() if item.is_cuda]
    return {
        "status": "passed",
        "warp_version": str(wp.__version__),
        "warp_module": str(module_path),
        "module": module_name,
        "device_executed": device,
        "cuda_devices": cuda_devices,
        "cuda_certified": False,
        "checks": {
            "direct_source_color_source_id_uid": "passed",
            "optional_channel_compaction": "passed",
            "checkpoint_restore": "passed",
            "prepared_packet_velocity_preservation": "passed",
            "disabled_channel_allocation": "passed",
            "vorticity_kernel": "passed",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--module", default=VALIDATION_PACKAGE)
    parser.add_argument("--report-json", type=Path, required=True)
    options = parser.parse_args(argv)
    report = validate(options.expected_warp_root, options.module)
    _write_json(options.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
