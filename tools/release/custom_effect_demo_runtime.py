"""Deterministic supplied-Warp simulations used by Custom Effects demo scenes.

These are real Custom Effects v2 compilations and launches against bounded
particle fixtures. They are intentionally independent from the CUDA-only full
FluidSolver so the supplied Warp package can prove behavior on this CPU-only
validation host without overstating CUDA solver certification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from ..sim.effects.builtins import builtin_effect_definitions
from ..sim.effects.compiler import build_candidate_effects
from ..sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
    ParameterKind,
)
from ..sim.effects.runtime import EffectRuntimeManager
from ..sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)
from ..sim.effects.trust import TrustStore, content_trust_digest
from ..version import VERSION_STRING


@wp.kernel
def _integrate_demo_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    dt: float,
):
    tid = wp.tid()
    if tid >= count[0] or alive[tid] == 0:
        return
    positions[tid] = positions[tid] + velocities[tid] * dt


@dataclass
class _DemoParticles:
    position: object
    velocity: object
    alive: object
    count: object
    capacity: int
    host_count: int

    def refresh_host_count(self) -> int:
        return int(self.host_count)


@dataclass
class _DemoSolver:
    particles: _DemoParticles
    settings: object
    frame: int = 0
    sim_time: float = 0.0


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _definition(effect_id: str) -> EffectDefinitionV2:
    by_id = {item.effect_id: item for item in builtin_effect_definitions()}
    return by_id[effect_id]


def _instance(
    effect_id: str,
    instance_id: str,
    order: int,
    *,
    parameters: dict[str, object] | None = None,
    controllers: dict[str, dict[str, object]] | None = None,
) -> EffectInstanceV2:
    return EffectInstanceV2(
        instance_id=instance_id,
        definition=_definition(effect_id),
        order=order,
        parameter_values=parameters or {},
        controller_bindings=controllers or {},
        display_name=_definition(effect_id).name,
    )


def authoring_definition() -> EffectDefinitionV2:
    """Return the non-built-in source used by the Warp Code authoring demo."""

    return EffectDefinitionV2(
        effect_id="demo.lateral_wave",
        version="1.0.0",
        name="Lateral Wave",
        summary="Add a time-varying lateral wave to liquid particles.",
        description=(
            "A compact developer-authored example for the Warp Code workspace. "
            "It is deliberately packaged as untrusted local code in the demo scene."
        ),
        category="Demo",
        author="FluidSim Demo",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin=f"demo://custom-effects/{VERSION_STRING}",
        parameters=(
            ParameterDefinitionV2(
                "amplitude",
                "Amplitude",
                ParameterKind.FLOAT,
                2.0,
                minimum=0.0,
                maximum=100.0,
                unit="m/s²",
                description="Lateral acceleration magnitude.",
            ),
            ParameterDefinitionV2(
                "frequency",
                "Frequency",
                ParameterKind.FLOAT,
                2.5,
                minimum=0.001,
                maximum=1000.0,
                unit="1/m",
                description="Spatial wave frequency along Y.",
            ),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="lateral_wave",
                label="Apply Lateral Wave",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source=(
                    "wave = wp.sin(position[1] * frequency + time)\n"
                    "particle_velocity[particle_index] = particle_velocity[particle_index] + "
                    "wp.vec3(wave * amplitude * solver_time_step, 0.0, 0.0)"
                ),
                fields=(
                    FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),
                ),
                parameter_keys=("amplitude", "frequency"),
                description="Adds bounded, timestep-aware lateral acceleration.",
            ),
        ),
        tags=("demo", "wave", "developer"),
        cost_hint="LOW",
    )


def _particle_fixture(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    device: str,
    canary_count: int = 4,
) -> tuple[_DemoSolver, dict[str, np.ndarray]]:
    active = len(positions)
    capacity = active + int(canary_count)
    position_canary = np.array([91.0, 92.0, 93.0], dtype=np.float32)
    velocity_canary = np.array([81.0, 82.0, 83.0], dtype=np.float32)
    position_storage = np.vstack((
        np.asarray(positions, dtype=np.float32),
        np.tile(position_canary, (canary_count, 1)),
    ))
    velocity_storage = np.vstack((
        np.asarray(velocities, dtype=np.float32),
        np.tile(velocity_canary, (canary_count, 1)),
    ))
    alive_storage = np.concatenate((
        np.ones((active,), dtype=np.int32),
        np.zeros((canary_count,), dtype=np.int32),
    ))
    count_storage = np.array([active], dtype=np.int32)
    particles = _DemoParticles(
        position=wp.array(position_storage, dtype=wp.vec3, device=device),
        velocity=wp.array(velocity_storage, dtype=wp.vec3, device=device),
        alive=wp.array(alive_storage, dtype=wp.int32, device=device),
        count=wp.array(count_storage, dtype=wp.int32, device=device),
        capacity=capacity,
        host_count=active,
    )
    solver = _DemoSolver(particles=particles, settings=SimpleNamespace(voxel_size=0.1))
    canaries = {
        "position": position_storage[active:].copy(),
        "velocity": velocity_storage[active:].copy(),
        "active_count": np.array([active], dtype=np.int32),
    }
    return solver, canaries


def _run_stack(
    instances: tuple[EffectInstanceV2, ...],
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    steps: int,
    dt: float,
    device: str,
    trust_path: Path,
    trust_custom: bool = False,
) -> dict[str, object]:
    store = TrustStore(trust_path)
    if trust_custom:
        for instance in instances:
            if not instance.definition.built_in:
                store.trust(content_trust_digest(instance.definition), instance.definition.name)
    candidate = build_candidate_effects(instances, device=device, trust_store=store)
    if not candidate.success:
        raise AssertionError([issue.to_dict() for issue in candidate.issues])
    manager = EffectRuntimeManager(device=device)
    if not manager.swap_candidate(candidate):
        raise AssertionError("candidate runtime swap failed")
    solver, canaries = _particle_fixture(positions, velocities, device=device)
    active_count = int(canaries["active_count"][0])
    trajectory = [np.asarray(positions, dtype=np.float32).copy()]
    for step in range(int(steps)):
        frame = step + 1
        solver.frame = frame
        solver.sim_time = step * float(dt)
        time_ctx = {
            "frame": frame,
            "substep_index": 0,
            "time": solver.sim_time,
            "real_time_step": float(dt),
            "solver_time_step": float(dt),
            "dt": float(dt),
            "time_scale": 1.0,
            "substeps": 1,
        }
        manager.run(EffectStage.BEFORE_EMISSION, solver, time_ctx)
        manager.run(EffectStage.AFTER_PARTICLE_TRANSFER, solver, time_ctx)
        manager.run(EffectStage.BEFORE_PRESSURE, solver, time_ctx)
        wp.launch(
            _integrate_demo_particles,
            dim=solver.particles.capacity,
            inputs=[
                solver.particles.position,
                solver.particles.velocity,
                solver.particles.alive,
                solver.particles.count,
                float(dt),
            ],
            device=device,
        )
        manager.run(EffectStage.AFTER_PARTICLE_UPDATE, solver, time_ctx)
        wp.synchronize_device(device)
        trajectory.append(solver.particles.position.numpy()[:active_count].copy())
    if manager.runtime_errors:
        raise AssertionError([vars(item) for item in manager.runtime_errors])
    final_positions_storage = solver.particles.position.numpy()
    final_velocities_storage = solver.particles.velocity.numpy()
    canaries_intact = bool(
        np.array_equal(final_positions_storage[active_count:], canaries["position"])
        and np.array_equal(final_velocities_storage[active_count:], canaries["velocity"])
    )
    finite = bool(
        np.isfinite(final_positions_storage[:active_count]).all()
        and np.isfinite(final_velocities_storage[:active_count]).all()
    )
    return {
        "initial_position": np.asarray(positions, dtype=np.float32).copy(),
        "initial_velocity": np.asarray(velocities, dtype=np.float32).copy(),
        "final_position": final_positions_storage[:active_count].copy(),
        "final_velocity": final_velocities_storage[:active_count].copy(),
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "canaries_intact": canaries_intact,
        "finite": finite,
        "signature": manager.applied_signature,
        "inventory": manager.inventory,
        "runtime_stats": {key: vars(value).copy() for key, value in manager.stats.items()},
    }


def _vortex_inputs() -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, math.tau, 48, endpoint=False, dtype=np.float32)
    radii = 0.75 + 0.35 * ((np.arange(48) % 3) / 2.0)
    z = np.linspace(-0.35, 0.35, 48, dtype=np.float32)
    positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), z)).astype(np.float32)
    velocities = np.zeros_like(positions)
    return positions, velocities


def run_vortex_demo(*, device: str, trust_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    positions, velocities = _vortex_inputs()
    instance = _instance(
        "fluidsim.vortex_force",
        "demo-vortex",
        0,
        parameters={
            "controller": "Vortex Controller",
            "axis": (0.0, 0.0, 1.0),
            "strength": 6.0,
            "radius": 2.5,
            "suction": 0.12,
            "lift": 0.08,
        },
        controllers={
            "controller": {
                "object_name": "Vortex Controller",
                "position": (0.0, 0.0, 0.0),
                "matrix_world": (
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ),
            }
        },
    )
    first = _run_stack((instance,), positions, velocities, steps=18, dt=1.0 / 30.0, device=device, trust_path=trust_path)
    second = _run_stack((instance,), positions, velocities, steps=18, dt=1.0 / 30.0, device=device, trust_path=trust_path)
    final_position = first["final_position"]
    final_velocity = first["final_velocity"]
    radius = np.linalg.norm(final_position[:, :2], axis=1)
    tangent = np.column_stack((
        -final_position[:, 1] / np.maximum(radius, 1.0e-8),
        final_position[:, 0] / np.maximum(radius, 1.0e-8),
        np.zeros_like(radius),
    ))
    tangential_speed_mean = float(np.mean(np.sum(final_velocity * tangent, axis=1)))
    deterministic_rerun_equal = bool(
        np.array_equal(first["final_position"], second["final_position"])
        and np.array_equal(first["final_velocity"], second["final_velocity"])
    )
    metrics = {
        "tangential_speed_mean": tangential_speed_mean,
        "mean_initial_radius": float(np.mean(np.linalg.norm(positions[:, :2], axis=1))),
        "mean_final_radius": float(np.mean(radius)),
        "mean_vertical_speed": float(np.mean(final_velocity[:, 2])),
        "deterministic_rerun_equal": deterministic_rerun_equal,
        "canaries_intact": bool(first["canaries_intact"]),
        "finite": bool(first["finite"]),
        "effect_signature": first["signature"],
    }
    if not (metrics["finite"] and metrics["canaries_intact"] and deterministic_rerun_equal):
        raise AssertionError(metrics)
    if tangential_speed_mean <= 0.25:
        raise AssertionError(f"vortex did not generate expected tangential motion: {metrics}")
    return first, metrics


def run_stack_demo(*, device: str, trust_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    angles = np.linspace(0.0, math.tau, 36, endpoint=False, dtype=np.float32)
    positions = np.column_stack((1.1 * np.cos(angles), 1.1 * np.sin(angles), np.zeros_like(angles))).astype(np.float32)
    velocities = np.zeros_like(positions)
    pulse = _instance(
        "fluidsim.radial_pulse",
        "demo-pulse",
        0,
        parameters={"controller": "Pulse Controller", "strength": 8.0, "radius": 3.0},
        controllers={"controller": {"object_name": "Pulse Controller", "position": (0.0, 0.0, 0.0)}},
    )
    damping = _instance(
        "fluidsim.velocity_damping",
        "demo-damping",
        1,
        parameters={"damping": 3.0},
    )
    pulse_then_damp = _run_stack((pulse, damping), positions, velocities, steps=8, dt=0.05, device=device, trust_path=trust_path)
    damping_first = EffectInstanceV2.from_dict({**damping.to_dict(), "order": 0})
    pulse_second = EffectInstanceV2.from_dict({**pulse.to_dict(), "order": 1})
    damp_then_pulse = _run_stack((damping_first, pulse_second), positions, velocities, steps=8, dt=0.05, device=device, trust_path=trust_path)

    damping_input_velocity = np.column_stack((
        np.linspace(0.5, 3.0, len(positions), dtype=np.float32),
        np.linspace(-2.0, 1.0, len(positions), dtype=np.float32),
        np.linspace(0.25, 1.25, len(positions), dtype=np.float32),
    ))
    damping_only = _run_stack((damping,), positions, damping_input_velocity, steps=8, dt=0.05, device=device, trust_path=trust_path)
    initial_ke = float(np.sum(damping_input_velocity * damping_input_velocity))
    final_ke = float(np.sum(damping_only["final_velocity"] * damping_only["final_velocity"]))
    kinetic_energy_ratio = final_ke / initial_ke
    expected_ratio = float((1.0 - 3.0 * 0.05) ** (2 * 8))
    stack_order_sensitivity = float(np.linalg.norm(
        pulse_then_damp["final_velocity"] - damp_then_pulse["final_velocity"]
    ))
    metrics = {
        "kinetic_energy_ratio": kinetic_energy_ratio,
        "expected_kinetic_energy_ratio": expected_ratio,
        "kinetic_energy_ratio_error": abs(kinetic_energy_ratio - expected_ratio),
        "stack_order_sensitivity": stack_order_sensitivity,
        "pulse_then_damp_mean_speed": float(np.mean(np.linalg.norm(pulse_then_damp["final_velocity"], axis=1))),
        "damp_then_pulse_mean_speed": float(np.mean(np.linalg.norm(damp_then_pulse["final_velocity"], axis=1))),
        "deterministic_rerun_equal": bool(np.array_equal(
            pulse_then_damp["final_velocity"],
            _run_stack((pulse, damping), positions, velocities, steps=8, dt=0.05, device=device, trust_path=trust_path)["final_velocity"],
        )),
        "canaries_intact": bool(
            pulse_then_damp["canaries_intact"]
            and damp_then_pulse["canaries_intact"]
            and damping_only["canaries_intact"]
        ),
        "finite": bool(
            pulse_then_damp["finite"]
            and damp_then_pulse["finite"]
            and damping_only["finite"]
        ),
        "effect_signature": pulse_then_damp["signature"],
    }
    if metrics["kinetic_energy_ratio_error"] > 2.0e-5:
        raise AssertionError(metrics)
    if stack_order_sensitivity <= 0.05:
        raise AssertionError(f"stack order did not alter the result: {metrics}")
    if not all((metrics["finite"], metrics["canaries_intact"], metrics["deterministic_rerun_equal"])):
        raise AssertionError(metrics)
    pulse_then_damp["comparison_final_position"] = damp_then_pulse["final_position"]
    pulse_then_damp["comparison_final_velocity"] = damp_then_pulse["final_velocity"]
    return pulse_then_damp, metrics


def run_authoring_demo(*, device: str, trust_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    definition = authoring_definition()
    instance = EffectInstanceV2(
        instance_id="demo-lateral-wave",
        definition=definition,
        order=0,
        parameter_values={"amplitude": 2.0, "frequency": 2.5},
        display_name=definition.name,
    )
    y = np.linspace(-1.5, 1.5, 40, dtype=np.float32)
    positions = np.column_stack((np.zeros_like(y), y, np.zeros_like(y))).astype(np.float32)
    velocities = np.zeros_like(positions)
    first = _run_stack(
        (instance,), positions, velocities, steps=10, dt=0.04, device=device,
        trust_path=trust_path, trust_custom=True,
    )
    second = _run_stack(
        (instance,), positions, velocities, steps=10, dt=0.04, device=device,
        trust_path=trust_path, trust_custom=True,
    )
    x_velocity_rms = float(np.sqrt(np.mean(first["final_velocity"][:, 0] ** 2)))
    metrics = {
        "x_velocity_rms": x_velocity_rms,
        "deterministic_rerun_equal": bool(
            np.array_equal(first["final_position"], second["final_position"])
            and np.array_equal(first["final_velocity"], second["final_velocity"])
        ),
        "canaries_intact": bool(first["canaries_intact"]),
        "finite": bool(first["finite"]),
        "content_digest": content_trust_digest(definition),
        "effect_signature": first["signature"],
    }
    if x_velocity_rms <= 0.01 or not all((
        metrics["deterministic_rerun_equal"], metrics["canaries_intact"], metrics["finite"]
    )):
        raise AssertionError(metrics)
    return first, metrics


def _array_payload(result: dict[str, object]) -> dict[str, np.ndarray]:
    keys = (
        "initial_position", "initial_velocity", "final_position", "final_velocity", "trajectory",
        "comparison_final_position", "comparison_final_velocity",
    )
    return {key: np.asarray(result[key]) for key in keys if key in result}


def run_all_demos(work_dir: str | Path, *, device: str = "cpu") -> dict[str, object]:
    """Execute, persist, and analytically validate every custom-effect demo."""

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    trust_path = work_dir / "demo-trust.json"
    records = []
    for name, runner in (
        ("vortex", run_vortex_demo),
        ("stack", run_stack_demo),
        ("authoring", run_authoring_demo),
    ):
        result, metrics = runner(device=device, trust_path=trust_path)
        npz_path = work_dir / f"custom_effects_{name}.npz"
        _atomic_npz(npz_path, _array_payload(result))
        records.append({
            "name": name,
            "npz": str(npz_path),
            "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "metrics": metrics,
            "effect_inventory": [dict(item) for item in result["inventory"]],
            "runtime_stats": result["runtime_stats"],
        })
    payload = {
        "status": "passed",
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "warp_module": str(Path(wp.__file__).resolve()),
        "device": str(device),
        "demo_count": len(records),
        "demos": records,
    }
    _atomic_json(work_dir / "custom_effects_simulation_report.json", payload)
    return payload
