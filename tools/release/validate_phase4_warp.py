"""Validate Phase-4 whitewater state using the supplied Warp package.

This is a release-only proof script. It executes the persistent type channel,
device classification/transition functions, deterministic admission, aligned
compaction, type counts, and checkpoint restoration on the available Warp
device. CUDA certification remains a separate gate when no NVIDIA device is
exposed by the host.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

VALIDATION_PACKAGE = "fluidsim_validation_source"


def _certified(name: str):
    """Read one certified release constant from the source tree's version.py."""
    path = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51" / "version.py"
    import ast

    values: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in values:
            value = values[node.value.id]  # e.g. CERTIFIED_X = VALIDATION_HOST_X
        else:
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    if name not in values:
        raise RuntimeError(f"{name} is not a literal constant in {path}")
    return values[name]


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


@wp.kernel
def _classify_and_transition(
    sdf: wp.array(dtype=float),
    speed: wp.array(dtype=float),
    churn: wp.array(dtype=float),
    current: wp.array(dtype=wp.int32),
    initial: wp.array(dtype=wp.int32),
    transitioned: wp.array(dtype=wp.int32),
):
    # Imported lazily through globals populated by validate(). Warp resolves
    # these wp.func symbols when the kernel is first compiled.
    initial[wp.tid()] = _CLASSIFY(
        sdf[wp.tid()], speed[wp.tid()], churn[wp.tid()], 0.1, 1.0, 0.5)
    transitioned[wp.tid()] = _TRANSITION(
        current[wp.tid()], sdf[wp.tid()], speed[wp.tid()],
        0.125, 0.075, 0.125, 0.075, 1.0, 0.75)


_CLASSIFY = None
_TRANSITION = None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _copy(target, values, dtype, *, device: str) -> None:
    values = np.ascontiguousarray(values)
    wp.copy(target, wp.array(values, dtype=dtype, device=device), count=len(values))


def validate(
    expected_warp_root: Path,
    module_name: str = VALIDATION_PACKAGE,
    *,
    device: str = "cpu",
) -> dict:
    global _CLASSIFY, _TRANSITION

    wp.init()
    warp_module = Path(wp.__file__).resolve()
    if expected_warp_root.resolve() not in warp_module.parents:
        raise AssertionError(
            f"Warp loaded from {warp_module}, expected under {expected_warp_root}")
    if str(getattr(wp, "__version__", "")) != _certified("CERTIFIED_WARP_VERSION"):
        raise AssertionError(f"unexpected Warp version {getattr(wp, '__version__', '')}")

    _ensure_source_package(module_name)
    whitewater = importlib.import_module(f"{module_name}.sim.whitewater")
    types = importlib.import_module(f"{module_name}.sim.whitewater_types")
    _CLASSIFY = whitewater.classify_whitewater_device
    _TRANSITION = whitewater.transition_whitewater_device

    available_aliases = {str(item.alias) for item in wp.get_devices()}
    if device not in available_aliases:
        raise AssertionError(
            f"requested Warp device {device!r} is unavailable; detected {sorted(available_aliases)}"
        )
    selected = wp.get_device(device)
    if device.startswith("cuda:") and not bool(getattr(selected, "is_cuda", False)):
        raise AssertionError(f"requested device {device!r} is not a CUDA device")

    sdf_host = np.array([-0.5, 0.02, 0.2], dtype=np.float32)
    speed_host = np.array([0.5, 0.4, 4.0], dtype=np.float32)
    churn_host = np.array([0.8, 0.8, 1.0], dtype=np.float32)
    current_host = np.array([1, 0, 2], dtype=np.int32)
    initial = wp.empty(3, dtype=wp.int32, device=device)
    transitioned = wp.empty(3, dtype=wp.int32, device=device)
    wp.launch(
        _classify_and_transition,
        dim=3,
        inputs=[
            wp.array(sdf_host, dtype=float, device=device),
            wp.array(speed_host, dtype=float, device=device),
            wp.array(churn_host, dtype=float, device=device),
            wp.array(current_host, dtype=wp.int32, device=device),
        ],
        outputs=[initial, transitioned],
        device=device,
    )
    wp.synchronize_device(device)
    np.testing.assert_array_equal(initial.numpy(), [1, 0, 2])
    # Bubble stays deep, foam stays near surface, spray stays elevated and fast.
    np.testing.assert_array_equal(transitioned.numpy(), [1, 0, 2])
    for index in range(3):
        expected = types.classify_sample(
            sdf=float(sdf_host[index]), speed=float(speed_host[index]),
            churn=float(churn_host[index]), foam_band=0.1,
            min_spray_speed=1.0, min_churn=0.5)
        if int(expected) != int(initial.numpy()[index]):
            raise AssertionError("host/device classification drifted")

    pool = whitewater.WhitewaterSystem(8, device, candidate_capacity=8)
    if any(hasattr(pool, name) for name in ("foam", "bubbles", "spray")):
        raise AssertionError("typed outputs allocated separate simulation pools")

    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32)
    velocities = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32)
    candidates = wp.array([1, 1, 1], dtype=wp.int32, device=device)
    candidate_types = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    offsets = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    plan = wp.array([3, 3, 0, 0], dtype=wp.int32, device=device)
    wp.launch(
        whitewater.k_commit_whitewater_candidates,
        dim=3,
        inputs=[
            wp.array(positions, dtype=wp.vec3, device=device),
            wp.array(velocities, dtype=wp.vec3, device=device),
            candidates, candidate_types, offsets, 0.0,
            2.0, 2.0, 1.0, 1.5, 0.5, 12345,
            pool.free_slots, plan,
        ],
        outputs=[
            pool.position, pool.velocity, pool.age, pool.life, pool.alive,
            pool.type,
        ],
        device=device,
    )
    _copy(pool.count, np.array([3], dtype=np.int32), wp.int32, device=device)
    pool.update_type_counts(3)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(pool.type.numpy()[:3], [0, 1, 2])
    np.testing.assert_allclose(pool.life.numpy()[:3], [2.0, 3.0, 1.0])
    assert pool.refresh_type_counts() == {
        "foam": 1, "bubble": 1, "spray": 1, "invalid": 0}

    # Kill the middle particle and prove the type channel compacts in lockstep.
    _copy(pool.alive, np.array([1, 0, 1], dtype=np.int32), wp.int32, device=device)
    pool.compact(count_hint=3)
    if pool.refresh_host_count() != 2:
        raise AssertionError("whitewater compaction produced the wrong count")
    np.testing.assert_allclose(pool.position.numpy()[:2], positions[[0, 2]])
    np.testing.assert_array_equal(pool.type.numpy()[:2], [0, 2])
    wp.synchronize_device(device)
    assert pool.refresh_type_counts() == {
        "foam": 1, "bubble": 0, "spray": 1, "invalid": 0}

    checkpoint = {
        "whitewater_position": pool.position.numpy()[:2],
        "whitewater_velocity": pool.velocity.numpy()[:2],
        "whitewater_age": pool.age.numpy()[:2],
        "whitewater_life": pool.life.numpy()[:2],
        "whitewater_alive": pool.alive.numpy()[:2],
        "whitewater_type": pool.type.numpy()[:2],
    }
    restored = whitewater.WhitewaterSystem(8, device, candidate_capacity=8)
    restored.restore_checkpoint_prefix(checkpoint, 2)
    np.testing.assert_array_equal(restored.type.numpy()[:2], [0, 2])
    invalid = dict(checkpoint)
    invalid["whitewater_type"] = np.array([0, 3], dtype=np.int32)
    try:
        restored.restore_checkpoint_prefix(invalid, 2)
    except ValueError as exc:
        if "whitewater type" not in str(exc):
            raise
    else:
        raise AssertionError("reserved whitewater type was restored")

    cuda_devices = [str(item.alias) for item in wp.get_devices() if item.is_cuda]
    return {
        "status": "passed",
        "warp_version": str(wp.__version__),
        "warp_module": str(warp_module),
        "device_executed": device,
        "cuda_devices": cuda_devices,
        "cuda_certified": bool(device.startswith("cuda:")),
        "checks": {
            "host_device_classification_parity": "passed",
            "stable_type_values": "passed",
            "single_simulation_pool": "passed",
            "typed_emission_commit": "passed",
            "type_specific_lifetimes": "passed",
            "type_count_reduction": "passed",
            "aligned_compaction": "passed",
            "checkpoint_restore": "passed",
            "reserved_type_rejection": "passed",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--module", default=VALIDATION_PACKAGE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-json", type=Path, required=True)
    options = parser.parse_args(argv)
    report = validate(options.expected_warp_root, options.module, device=options.device)
    _write_json(options.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
