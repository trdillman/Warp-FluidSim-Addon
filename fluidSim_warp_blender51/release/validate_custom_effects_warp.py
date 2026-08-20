#!/usr/bin/env python3
"""Validate Custom Effects v2 against an explicitly supplied Warp package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _fixture_modules():
    from fluidsim_testpkg.sim.effects.enums import (
        EffectCadence,
        EffectDomain,
        EffectStage,
        FieldAccess,
        ParameterKind,
    )
    from fluidsim_testpkg.sim.effects.schema import (
        EffectDefinitionV2,
        EffectInstanceV2,
        FieldBindingV2,
        KernelDefinitionV2,
        ParameterDefinitionV2,
    )

    positive = EffectInstanceV2(
        "warp-proof-positive",
        EffectDefinitionV2(
            effect_id="validation.warp-proof",
            version="1.0.0",
            name="Warp Proof",
            summary="Warp proof",
            description="Warp proof",
            category="Validation",
            author="FluidSim",
            license="GPL-3.0-or-later",
            built_in=True,
            parameters=(
                ParameterDefinitionV2("strength", "Strength", ParameterKind.FLOAT, 0.5),
            ),
            kernels=(
                KernelDefinitionV2(
                    kernel_id="particle_scale",
                    label="Particle Scale",
                    domain=EffectDomain.PARTICLES,
                    stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                    cadence=EffectCadence.EVERY_SUBSTEP,
                    source="particle_velocity[particle_index] = velocity * strength",
                    fields=(
                        FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),
                    ),
                    parameter_keys=("strength",),
                ),
            ),
        ),
    )
    scalar_vec3 = EffectInstanceV2(
        "warp-proof-negative",
        EffectDefinitionV2(
            effect_id="validation.scalar-vec3",
            version="1.0.0",
            name="Scalar Vec3 Negative",
            summary="Negative",
            description="Negative",
            category="Validation",
            author="FluidSim",
            license="GPL-3.0-or-later",
            built_in=True,
            kernels=(
                KernelDefinitionV2(
                    kernel_id="bad_scalar",
                    label="Bad Scalar",
                    domain=EffectDomain.ACTIVE_VOXELS,
                    stage=EffectStage.BEFORE_PRESSURE,
                    cadence=EffectCadence.EVERY_SUBSTEP,
                    source="temperature[flat_voxel_index] = wp.vec3(1.0, 2.0, 3.0)",
                    fields=(FieldBindingV2("temperature", FieldAccess.READ_WRITE_CURRENT),),
                ),
            ),
        ),
    )
    missing_builtin = EffectInstanceV2(
        "warp-proof-missing",
        EffectDefinitionV2(
            effect_id="validation.missing-builtin",
            version="1.0.0",
            name="Missing Builtin Negative",
            summary="Negative",
            description="Negative",
            category="Validation",
            author="FluidSim",
            license="GPL-3.0-or-later",
            built_in=True,
            kernels=(
                KernelDefinitionV2(
                    kernel_id="bad_builtin",
                    label="Bad Builtin",
                    domain=EffectDomain.PARTICLES,
                    stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                    cadence=EffectCadence.EVERY_SUBSTEP,
                    source="particle_velocity[particle_index] = wp.not_a_builtin(velocity)",
                    fields=(FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),),
                ),
            ),
        ),
    )
    return positive, scalar_vec3, missing_builtin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    package_name = "fluidsim_testpkg"
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    if package_name not in sys.modules:
        from types import ModuleType
        package = ModuleType(package_name)
        package.__file__ = str(root / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(root)]
        sys.modules[package_name] = package

    import warp as wp
    from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions
    from fluidsim_testpkg.sim.effects.compiler import compile_effect
    from fluidsim_testpkg.sim.effects.sandbox import sandbox_test
    from fluidsim_testpkg.sim.effects.schema import EffectInstanceV2

    expected = args.expected_warp_root.resolve()
    actual = Path(wp.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(f"Warp imported from {actual}, expected below {expected}")
    wp.init()

    positive, scalar_vec3, missing_builtin = _fixture_modules()
    started = time.perf_counter()
    compiled = compile_effect(positive, device=args.device)
    sandbox = sandbox_test(compiled, device=args.device)
    negative_scalar = compile_effect(scalar_vec3, device=args.device)
    negative_builtin = compile_effect(missing_builtin, device=args.device)
    builtin_results = {}
    for definition in builtin_effect_definitions():
        instance = EffectInstanceV2(
            f"warp-proof-{definition.effect_id.replace('.', '-')}", definition
        )
        builtin_compiled = compile_effect(instance, device=args.device)
        builtin_sandbox = sandbox_test(builtin_compiled, device=args.device)
        builtin_results[definition.effect_id] = {
            "compiled": bool(builtin_compiled.success),
            "sandbox_passed": bool(builtin_sandbox.passed),
            "kernel_count": len(builtin_compiled.kernels),
            "source_digests": [
                kernel.compile_report.source_digest
                for kernel in builtin_compiled.kernels
            ],
            "issues": [issue.to_dict() for issue in (
                builtin_compiled.issues + builtin_sandbox.issues
            )],
        }
    checks = {
        "positive_compile": compiled.success,
        "positive_sandbox": sandbox.passed,
        "positive_canaries": bool(sandbox.kernels and sandbox.kernels[0].canaries_intact),
        "positive_finite": bool(sandbox.kernels and sandbox.kernels[0].finite),
        "scalar_vec3_rejected": not negative_scalar.success,
        "missing_builtin_rejected": not negative_builtin.success,
        "all_builtins_compile_and_sandbox": all(
            result["compiled"] and result["sandbox_passed"]
            for result in builtin_results.values()
        ),
        "builtin_inventory_complete": len(builtin_results) == 8,
    }
    payload = {
        "schema_version": 1,
        "success": all(checks.values()),
        "checks": checks,
        "warp_version": str(wp.__version__),
        "warp_module": str(actual),
        "device": str(args.device),
        "available_devices": [str(device) for device in wp.get_devices()],
        "elapsed_seconds": max(0.0, time.perf_counter() - started),
        "negative_scalar_issues": [issue.to_dict() for issue in negative_scalar.issues],
        "negative_builtin_issues": [issue.to_dict() for issue in negative_builtin.issues],
        "builtins": builtin_results,
    }
    _atomic_json(args.report_json, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
