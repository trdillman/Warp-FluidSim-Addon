"""CUDA acceptance for unified source ownership and atomic direct injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from _test_device import script_device  # noqa: E402

from fluidSim_warp_blender51.sim.enums import (  # noqa: E402
    SourceEmissionMode, SourceEvaluationCadence, SourceLifecycle,
    SourceOperation, SourcePayloadKind, SourceShapeShape,
    SourceShapeSourceMode, SourceShapeType,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings  # noqa: E402
from fluidSim_warp_blender51.sim.source_regions import (  # noqa: E402
    AnalyticRegionPlan, PayloadPlan, PointRegionPlan, SourceKey, SourcePlan,
)
from fluidSim_warp_blender51.sim.sourcing import (  # noqa: E402
    DirectInjectionPacket, SourceShapeDesc,
)


def settings(max_particles=2_000):
    value = SolverSettings()
    value.domain_min = (-0.6, -0.6, 0.0)
    value.domain_max = (0.6, 0.6, 1.2)
    value.voxel_size = 0.1
    value.leaf_capacity = 64
    value.max_particles = max_particles
    value.whitewater_enabled = False
    value.surface_enabled = False
    value.min_substeps = 1
    value.max_substeps = 1
    value.auto_grow_particles = False
    value.device = script_device()
    return value


def direct_plan(key, points):
    return SourcePlan(
        SourceKey(key), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.asarray(points, np.float32).reshape((-1, 3))),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not wp.get_device(script_device()).is_cuda:
        raise RuntimeError("mixed source-batch acceptance requires a CUDA device")
    cases = []

    solver = FluidSolver(settings())
    try:
        legacy = SourceShapeDesc(
            enabled=True, shape=SourceShapeShape.BOX,
            stype=SourceShapeType.SOURCE,
            mode=SourceShapeSourceMode.PARTICLES,
            position=(0.0, 0.0, 0.75), half_extents=(0.2, 0.2, 0.2),
            emit_rate=1.0, emission_mode=SourceEmissionMode.INITIAL_FILL,
            source_id="legacy-mixed")
        direct = direct_plan("native-direct", ((0.45, 0.45, 0.9),))
        volume = SourcePlan(
            SourceKey("native-volume"), SourceOperation.LIQUID,
            SourceLifecycle.ONCE, SourceEvaluationCadence.STATIC,
            AnalyticRegionPlan(
                shape=SourceShapeShape.BOX, position=(-0.45, -0.45, 0.9),
                half_extents=(0.15, 0.15, 0.15)),
            PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                        target_particles_per_voxel=1.0))
        solver.set_source_batch((legacy,), (direct, volume))
        solver.step_frame()
        first = solver.particles.refresh_host_count()
        fired = set(solver.sourcing._initial_fired)
        positions = solver.particles.position.numpy()[:first]
        native_volume_count = int(np.count_nonzero(
            np.linalg.norm(
                positions - np.array((-0.45, -0.45, 0.9), np.float32),
                axis=1) < 0.2))
        native_direct_count = int(np.count_nonzero(
            np.linalg.norm(
                positions - np.array((0.45, 0.45, 0.9), np.float32),
                axis=1) < 0.2))
        legacy_count = int(np.count_nonzero(
            np.linalg.norm(
                positions - np.array((0.0, 0.0, 0.75), np.float32),
                axis=1) < 0.3))
        assert native_volume_count > 0
        assert native_direct_count == 1
        assert legacy_count > 0
        assert fired == {"legacy-mixed", "native-direct", "native-volume"}
        solver.step_frame()
        second = solver.particles.refresh_host_count()
        assert second == first
        cases.append({
            "name": "mixed_once", "count": first,
            "per_region": {
                "legacy": legacy_count,
                "native_volume": native_volume_count,
                "native_direct": native_direct_count,
            },
            "fired": sorted(fired),
        })
    finally:
        solver.release()

    solver = FluidSolver(settings(3))
    try:
        injection = solver.inject_particles(
            np.asarray(((0.0, 0.0, 0.5),), np.float32))
        assert injection.admitted and injection.emitted == 1
        solver.particles.host_count = 0
        packets = (
            DirectInjectionPacket(np.asarray(
                ((-0.2, 0.0, 0.7), (0.0, 0.0, 0.7)), np.float32),
                temperature=2.0),
            DirectInjectionPacket(np.asarray(
                ((0.2, 0.0, 0.7),), np.float32), viscosity=3.0),
        )
        result = solver.inject_particle_packets_exact(packets)
        assert not result.admitted
        assert result.required == 3 and result.available == 2
        assert result.emitted == 0
        assert solver.particles.refresh_host_count() == 1
        cases.append({"name": "stale_host_packet_refusal",
                      "required": result.required,
                      "available": result.available,
                      "count": result.total})
    finally:
        solver.release()

    solver = FluidSolver(settings())
    try:
        zero = SourcePlan(
            SourceKey("zero-liquid"), SourceOperation.LIQUID,
            SourceLifecycle.ONCE, SourceEvaluationCadence.STATIC,
            AnalyticRegionPlan(
                shape=SourceShapeShape.BOX, position=(0.0, 0.0, 0.75),
                half_extents=(0.2, 0.2, 0.2)),
            PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                        target_particles_per_voxel=0.0))
        field = SourcePlan(
            SourceKey("pressure-field"), SourceOperation.PRESSURE,
            SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
            AnalyticRegionPlan(
                shape=SourceShapeShape.BOX, position=(0.0, 0.0, 0.75),
                half_extents=(0.2, 0.2, 0.2)),
            PayloadPlan(kind=SourcePayloadKind.PRESSURE, pressure=4.0))
        solver.set_source_batch(native_plans=(zero, field))
        solver.step_frame()
        assert solver.particles.refresh_host_count() == 0
        assert "zero-liquid" not in solver.sourcing._initial_fired
        assert solver.sourcing.last_pressure_apply_count > 0
        footprint = solver.sourcing._compiled_batch.footprints[0]
        assert footprint.particle_demand == 0 and not footprint.leaf_coords
        cases.append({"name": "zero_liquid_field_active",
                      "pressure_voxels": solver.sourcing.last_pressure_apply_count})
    finally:
        solver.release()

    report = {"status": "pass", "cases": cases}
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf8")
    print("MIXED-SOURCE-BATCH-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
