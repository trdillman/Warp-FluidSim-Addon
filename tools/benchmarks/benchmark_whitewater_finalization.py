"""Benchmark the exact whitewater finalization seam in fresh CUDA processes.

The public ``run`` command alternates baseline-first and candidate-first pairs
(AB, BA, AB, ...).  Every leg is a new Blender 5.1 bundled-Python process.
Reports are provenance-bound raw observations and are never approval-eligible.
The ``validate`` command intentionally needs only system Python.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCHEMA = "fluidsim-whitewater-finalization-campaign-v1"
WORKER_SCHEMA = "fluidsim-whitewater-finalization-worker-v1"
RESULT = "WHITEWATER-FINALIZATION-BENCHMARK-PASS"
WORKER_RESULT = "WHITEWATER-FINALIZATION-WORKER-PASS"
MEASUREMENT_KIND = "targeted_whitewater_finalization"
PACKAGE_NAME = "fluidSim_warp_blender51"
DEFAULT_LIVE_COUNTS = (50_000, 100_000, 200_000)
FIXTURE_CHANNELS = (
    "whitewater_position", "whitewater_velocity", "whitewater_age",
    "whitewater_life", "whitewater_alive", "whitewater_type",
)


def validate_configuration(
    *, pairs: int, iterations: int, warmup: int, live_counts: tuple[int, ...],
) -> None:
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if type(iterations) is not int or iterations < 30:
        raise ValueError("iterations must be at least 30")
    if type(warmup) is not int or warmup < 1:
        raise ValueError("warmup must be a positive integer")
    if not live_counts or any(type(value) is not int or value < 50_000 for value in live_counts):
        raise ValueError("live-counts must contain integers of at least 50000")
    if len(set(live_counts)) != len(live_counts):
        raise ValueError("live-counts must not contain duplicates")


def pair_order(pair: int) -> tuple[str, str]:
    if type(pair) is not int or pair < 1:
        raise ValueError("pair must be a positive integer")
    return ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")


def _parse_counts(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("live-counts must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("live-counts must not be empty")
    return values


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise ValueError(f"could not resolve Git identity for {root}: {detail}")
    return result.stdout.strip()


def _tree_digest(package_root: Path) -> str:
    digest = hashlib.sha256()
    files: list[tuple[str, Path]] = []
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        if path.is_file() and "__pycache__" not in relative.parts and path.suffix not in {".pyc", ".pyo"}:
            files.append((relative.as_posix(), path))
    if not files:
        raise ValueError(f"add-on package contains no files: {package_root}")
    for relative, path in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_code_identity(addon_root: Path) -> dict[str, object]:
    root = addon_root.resolve()
    package_root = root / PACKAGE_NAME
    if not package_root.is_dir() or not (package_root / "__init__.py").is_file():
        raise ValueError(f"addon-root does not contain {PACKAGE_NAME}: {root}")
    git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if git_root != root:
        raise ValueError(f"addon-root {root} does not match Git root {git_root}")
    head = _git(root, "rev-parse", "HEAD")
    if len(head) != 40:
        raise ValueError(f"Git HEAD is not a full commit id: {head!r}")
    return {
        "addon_root": str(root),
        "git_head": head,
        "addon_tree_sha256": _tree_digest(package_root),
        "dirty": bool(_git(root, "status", "--porcelain")),
        "addon_dirty": bool(_git(root, "status", "--porcelain", "--", PACKAGE_NAME)),
    }


def _bootstrap_addon(root: Path) -> Path:
    if PACKAGE_NAME in sys.modules:
        raise RuntimeError("worker requires a fresh process with the add-on not yet imported")
    expected = root.resolve() / PACKAGE_NAME
    sys.path[:] = [str(root.resolve())] + [item for item in sys.path if item != str(root.resolve())]
    package = __import__(PACKAGE_NAME)
    loaded = Path(package.__file__).resolve()
    if expected not in loaded.parents:
        raise RuntimeError(f"loaded {PACKAGE_NAME} from {loaded}, expected {expected}")
    return loaded


def _require_blender_51_python() -> str:
    executable = Path(sys.executable).resolve()
    lowered = tuple(part.lower() for part in executable.parts)
    if "blender foundation" not in lowered or "5.1" not in lowered or "python" not in lowered:
        raise RuntimeError("worker must run under Blender 5.1 bundled Python, got " + str(executable))
    return "5.1"


def configure_writable_warp_cache(wp, *, base: Path | None = None) -> Path:
    parent = Path(base) if base is not None else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="fluidsim-warp-whitewater-cache-", dir=parent))
    wp.config.kernel_cache_dir = str(path)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(path)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _sha256_array(value) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _fixture_arrays(count: int, seed: int) -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(seed + count)
    position = np.ascontiguousarray(rng.uniform(0.05, 0.95, (count, 3)), dtype=np.float32)
    velocity = np.ascontiguousarray(rng.normal(0.0, 0.25, (count, 3)), dtype=np.float32)
    age = np.ascontiguousarray(rng.uniform(0.0, 1.0, count), dtype=np.float32)
    life = np.ascontiguousarray(rng.uniform(1.5, 4.0, count), dtype=np.float32)
    alive = np.ones(count, dtype=np.int32)
    particle_type = np.ascontiguousarray(np.arange(count, dtype=np.int32) % 3)
    return {
        "whitewater_position": position,
        "whitewater_velocity": velocity,
        "whitewater_age": age,
        "whitewater_life": life,
        "whitewater_alive": alive,
        "whitewater_type": particle_type,
    }


def _array_summary(solver, count: int) -> dict[str, object]:
    import numpy as np

    ww = solver.whitewater
    channels = {
        "position_sha256": np.asarray(ww.position.numpy()[:count], dtype=np.float32),
        "velocity_sha256": np.asarray(ww.velocity.numpy()[:count], dtype=np.float32),
        "age_sha256": np.asarray(ww.age.numpy()[:count], dtype=np.float32),
        "life_sha256": np.asarray(ww.life.numpy()[:count], dtype=np.float32),
        "alive_sha256": np.asarray(ww.alive.numpy()[:count], dtype=np.int32),
        "type_sha256": np.asarray(ww.type.numpy()[:count], dtype=np.int32),
        "density_sha256": np.asarray(solver.ww_density.data.numpy()[:solver._nvox], dtype=np.float32),
    }
    return {
        "count_array": [int(value) for value in ww.count.numpy().tolist()],
        "type_count_array": [int(value) for value in ww.type_counts.numpy().tolist()],
        "digests": {name: _sha256_array(value) for name, value in channels.items()},
    }


def _settings(max_whitewater: int, device: str):
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    settings = SolverSettings()
    settings.domain_min = (0.0, 0.0, 0.0)
    settings.domain_max = (1.0, 1.0, 1.0)
    settings.voxel_size = 1.0 / 32.0
    settings.leaf_capacity = 64
    settings.max_particles = 1
    settings.auto_grow_particles = False
    settings.max_whitewater = max_whitewater
    settings.whitewater_enabled = True
    settings.surface_enabled = False
    settings.gravity = (0.0, 0.0, 0.0)
    settings.relax_enabled = False
    settings.device = device
    return settings


def _run_count(args, count: int, solver_mod, whitewater_mod, wp) -> dict[str, object]:
    solver = solver_mod.FluidSolver(_settings(max(args.live_counts), args.device))
    current: dict[str, int] | None = None
    original_launch = solver_mod.wp.launch
    original_compact = whitewater_mod.WhitewaterSystem.compact

    def counted_launch(kernel, *call_args, **call_kwargs):
        if current is not None:
            current["wp_launches"] += 1
            if kernel is whitewater_mod.k_count_whitewater_types:
                current["type_count_launches"] += 1
            elif kernel is whitewater_mod.k_whitewater_set_grid_data:
                current["density_launches"] += 1
        return original_launch(kernel, *call_args, **call_kwargs)

    def counted_compact(instance, *call_args, **call_kwargs):
        if current is not None and instance is solver.whitewater:
            current["compact_calls"] += 1
        return original_compact(instance, *call_args, **call_kwargs)

    solver_mod.wp.launch = counted_launch
    whitewater_mod.WhitewaterSystem.compact = counted_compact
    try:
        leaves = tuple(
            (x, y, z)
            for x in range(int(solver.grid.leaves_dim[0]))
            for y in range(int(solver.grid.leaves_dim[1]))
            for z in range(int(solver.grid.leaves_dim[2]))
        )
        solver.grid.restore_topology(leaves)
        fixture = _fixture_arrays(count, args.seed)
        solver.whitewater.restore_checkpoint_prefix(fixture, count)
        expected_types = [count // 3 + (1 if index < count % 3 else 0) for index in range(3)] + [0]
        input_digests = {name: _sha256_array(value) for name, value in fixture.items()}

        for _ in range(args.warmup):
            solver._finalize_whitewater()
        wp.synchronize_device(args.device)

        samples: list[float] = []
        telemetry = {
            "wp_launches_per_iteration": [],
            "type_count_launches_per_iteration": [],
            "density_launches_per_iteration": [],
            "compact_calls_per_iteration": [],
        }
        observed_counts: list[int] = []
        observed_types: list[list[int]] = []
        for _ in range(args.iterations):
            bucket = {"wp_launches": 0, "type_count_launches": 0, "density_launches": 0, "compact_calls": 0}
            current = bucket
            wp.synchronize_device(args.device)
            started = time.perf_counter()
            observed_count = int(solver._finalize_whitewater())
            wp.synchronize_device(args.device)
            samples.append(float(time.perf_counter() - started))
            current = None
            observed_counts.append(observed_count)
            observed_types.append([int(value) for value in solver.whitewater.type_counts.numpy().tolist()])
            for name in telemetry:
                telemetry[name].append(bucket[name.removesuffix("_per_iteration")])

        output_arrays = _array_summary(solver, count)
        return {
            "live_count": count,
            "calibration": {
                "fixture_seed": args.seed,
                "expected_live_count": count,
                "expected_type_count_array": expected_types,
                "input_array_digests": input_digests,
            },
            "samples_seconds": samples,
            "sample_median_seconds": statistics.median(samples),
            "sample_min_seconds": min(samples),
            "sample_max_seconds": max(samples),
            "device_synchronized_samples": True,
            "observed_live_counts": observed_counts,
            "observed_type_count_arrays": observed_types,
            "telemetry_method": {
                "timed_seam": "FluidSolver._finalize_whitewater",
                "launches": "identity wrapper around solver module wp.launch; array_scan internals excluded",
                "type_counts": "identity match for k_count_whitewater_types",
                "compaction": "WhitewaterSystem.compact wrapper",
            },
            "telemetry": telemetry,
            "output_arrays": output_arrays,
        }
    finally:
        current = None
        solver_mod.wp.launch = original_launch
        whitewater_mod.WhitewaterSystem.compact = original_compact
        release = getattr(solver, "release", None)
        if callable(release):
            release()


def run_worker(args, identity: dict[str, object]) -> dict[str, object]:
    process_instance_token = f"{os.getpid()}-{time.time_ns()}"
    blender_version = _require_blender_51_python()
    package_path = _bootstrap_addon(Path(identity["addon_root"]))
    import inspect
    import warp as wp
    from fluidSim_warp_blender51.sim import solver as solver_mod
    from fluidSim_warp_blender51.sim import whitewater as whitewater_mod

    cache = configure_writable_warp_cache(wp)
    wp.init()
    if not str(args.device).startswith("cuda") or not wp.is_cuda_available():
        raise RuntimeError("whitewater finalization benchmark requires an available CUDA device")
    device = wp.get_device(args.device)
    scenarios = [_run_count(args, count, solver_mod, whitewater_mod, wp) for count in args.live_counts]
    report = {
        "schema": WORKER_SCHEMA,
        "result": WORKER_RESULT,
        "measurement_kind": MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "raw observation only; no standalone speedup or approval claim",
        "role": args.role,
        "pair": args.pair,
        "code_identity": identity,
        "seam_provenance": {
            "callable": "FluidSolver._finalize_whitewater",
            "module_path": str(Path(solver_mod.__file__).resolve()),
            "source_sha256": hashlib.sha256(
                inspect.getsource(solver_mod.FluidSolver._finalize_whitewater).encode("utf-8")
            ).hexdigest(),
            "benchmark_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "blender_version": blender_version,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "warp_version": str(getattr(wp, "__version__", "unknown")),
            "warp_path": str(Path(wp.__file__).resolve()),
            "device": str(getattr(device, "alias", args.device)),
            "device_name": str(getattr(device, "name", args.device)),
            "warp_kernel_cache": str(cache),
            "package_path": str(package_path),
            "process_id": os.getpid(),
            "process_instance_token": process_instance_token,
            "fresh_process_import": True,
        },
        "configuration": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "live_counts": list(args.live_counts),
            "seed": args.seed,
        },
        "scenarios": scenarios,
        "error_count": 0,
        "errors": [],
    }
    errors = validate_worker_report(report)
    if errors:
        raise RuntimeError("invalid worker report: " + "; ".join(errors))
    return report


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _identity_errors(identity: object) -> list[str]:
    if not isinstance(identity, dict):
        return ["code_identity must be an object"]
    errors = []
    if not isinstance(identity.get("addon_root"), str) or not identity["addon_root"]:
        errors.append("code_identity.addon_root is required")
    if not isinstance(identity.get("git_head"), str) or len(identity["git_head"]) != 40:
        errors.append("code_identity.git_head must be a full commit id")
    if not _valid_digest(identity.get("addon_tree_sha256")):
        errors.append("code_identity.addon_tree_sha256 must be sha256")
    for key in ("dirty", "addon_dirty"):
        if type(identity.get(key)) is not bool:
            errors.append(f"code_identity.{key} must be boolean")
    return errors


def validate_worker_report(report: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != WORKER_SCHEMA:
        errors.append(f"schema must be {WORKER_SCHEMA}")
    if report.get("result") != WORKER_RESULT:
        errors.append(f"result must be {WORKER_RESULT}")
    if report.get("measurement_kind") != MEASUREMENT_KIND:
        errors.append(f"measurement_kind must be {MEASUREMENT_KIND}")
    if report.get("approval_eligible") is not False:
        errors.append("raw worker report must not claim approval eligibility")
    if report.get("role") not in {"baseline", "candidate"}:
        errors.append("role must be baseline or candidate")
    if type(report.get("pair")) is not int or report["pair"] < 1:
        errors.append("pair must be a positive integer")
    errors.extend(_identity_errors(report.get("code_identity")))
    seam = report.get("seam_provenance")
    if not isinstance(seam, dict) or seam.get("callable") != "FluidSolver._finalize_whitewater":
        errors.append("seam provenance must identify FluidSolver._finalize_whitewater")
    elif not _valid_digest(seam.get("source_sha256")) or not _valid_digest(seam.get("benchmark_sha256")):
        errors.append("seam provenance digests must be sha256")
    environment = report.get("environment")
    required_environment = (
        "python_executable", "python_version", "warp_version", "warp_path",
        "device", "process_id", "process_instance_token",
    )
    if not isinstance(environment, dict) or any(environment.get(key) in (None, "") for key in required_environment):
        errors.append("environment must contain runtime and process provenance")
    elif environment.get("fresh_process_import") is not True:
        errors.append("environment must attest fresh-process import")
    elif type(environment.get("process_id")) is not int or environment["process_id"] < 1:
        errors.append("environment.process_id must be a positive integer")
    elif not isinstance(environment.get("process_instance_token"), str):
        errors.append("environment.process_instance_token must be a string")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        errors.append("configuration must be an object")
        configuration = {}
    try:
        validate_configuration(
            pairs=1, iterations=configuration.get("iterations"), warmup=configuration.get("warmup"),
            live_counts=tuple(configuration.get("live_counts", ())),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    scenarios = report.get("scenarios")
    raw_counts = configuration.get("live_counts", ())
    counts = list(raw_counts) if isinstance(raw_counts, (list, tuple)) else []
    if not isinstance(scenarios, list) or [item.get("live_count") for item in scenarios if isinstance(item, dict)] != counts:
        errors.append("scenarios must match configured live-count order")
        scenarios = scenarios if isinstance(scenarios, list) else []
    iterations = configuration.get("iterations")
    valid_iterations = iterations if type(iterations) is int and iterations >= 0 else 0
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            errors.append("each scenario must be an object")
            continue
        count = scenario.get("live_count")
        samples = scenario.get("samples_seconds")
        if not isinstance(samples, list) or len(samples) != valid_iterations or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in (samples or ())
        ):
            errors.append(f"live-count {count} samples must be positive, finite, and complete")
        if scenario.get("device_synchronized_samples") is not True:
            errors.append(f"live-count {count} samples must be device-synchronized")
        calibration = scenario.get("calibration")
        expected = calibration.get("expected_type_count_array") if isinstance(calibration, dict) else None
        if not isinstance(calibration, dict) or calibration.get("expected_live_count") != count:
            errors.append(f"live-count {count} calibration is invalid")
        elif not isinstance(expected, list) or len(expected) != 4 or sum(expected) != count:
            errors.append(f"live-count {count} expected type-count array is invalid")
        elif set(calibration.get("input_array_digests", {})) != set(FIXTURE_CHANNELS) or any(
            not _valid_digest(value) for value in calibration.get("input_array_digests", {}).values()
        ):
            errors.append(f"live-count {count} input array digests are invalid")
        observed_counts = scenario.get("observed_live_counts")
        observed_types = scenario.get("observed_type_count_arrays")
        if not isinstance(observed_counts, list) or observed_counts != [count] * valid_iterations:
            errors.append(f"live-count {count} observations must preserve exact live count")
        if not isinstance(observed_types, list) or observed_types != [expected] * valid_iterations:
            errors.append(f"live-count {count} observations must preserve exact type-count arrays")
        telemetry = scenario.get("telemetry")
        keys = ("wp_launches_per_iteration", "type_count_launches_per_iteration", "density_launches_per_iteration", "compact_calls_per_iteration")
        telemetry_valid = isinstance(telemetry, dict) and not any(
            not isinstance(telemetry.get(key), list) or len(telemetry[key]) != valid_iterations
            or any(type(value) is not int or value < 0 for value in telemetry[key]) for key in keys
        )
        if not telemetry_valid:
            errors.append(f"live-count {count} launch telemetry must match measured iterations")
        else:
            expected_type_launches = 2 if report.get("role") == "baseline" else 1
            if telemetry["type_count_launches_per_iteration"] != [expected_type_launches] * valid_iterations:
                errors.append(
                    f"live-count {count} {report.get('role')} type-count launches "
                    f"must equal {expected_type_launches} per iteration"
                )
            if telemetry["density_launches_per_iteration"] != [1] * valid_iterations:
                errors.append(f"live-count {count} density launches must equal 1 per iteration")
            if telemetry["compact_calls_per_iteration"] != [1] * valid_iterations:
                errors.append(f"live-count {count} compact calls must equal 1 per iteration")
        output = scenario.get("output_arrays")
        if not isinstance(output, dict) or output.get("count_array") != [count] or output.get("type_count_array") != expected:
            errors.append(f"live-count {count} exact output arrays are invalid")
        elif set(output.get("digests", {})) != {
            "position_sha256", "velocity_sha256", "age_sha256", "life_sha256",
            "alive_sha256", "type_sha256", "density_sha256",
        } or any(not _valid_digest(value) for value in output["digests"].values()):
            errors.append(f"live-count {count} output digests are invalid")
    if report.get("error_count") != 0 or report.get("errors") != []:
        errors.append("passing worker report must contain no errors")
    return errors


def _correctness_signature(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "live_count": scenario["live_count"],
            "calibration": scenario["calibration"],
            "output_arrays": scenario["output_arrays"],
        }
        for scenario in report["scenarios"]
    ]


def _comparison(baseline: dict[str, object], candidate: dict[str, object]) -> list[dict[str, float | int]]:
    rows = []
    for base, cand in zip(baseline["scenarios"], candidate["scenarios"]):
        base_median = float(base["sample_median_seconds"])
        cand_median = float(cand["sample_median_seconds"])
        rows.append({
            "live_count": int(base["live_count"]),
            "baseline_median_seconds": base_median,
            "candidate_median_seconds": cand_median,
            "candidate_over_baseline_ratio": cand_median / base_median,
        })
    return rows


def _run_worker_process(
    *, blender_python: Path, addon_root: Path, role: str, pair: int, args, output: Path,
) -> dict[str, object]:
    command = [
        str(blender_python), str(Path(__file__).resolve()), "worker",
        "--addon-root", str(addon_root.resolve()), "--role", role, "--pair", str(pair),
        "--device", args.device, "--iterations", str(args.iterations),
        "--warmup", str(args.warmup), "--live-counts", ",".join(map(str, args.live_counts)),
        "--seed", str(args.seed), "--json-out", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker produced no diagnostic"
        raise RuntimeError(f"pair {pair} {role} worker failed: {detail}")
    with output.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    errors = validate_worker_report(report)
    if errors:
        raise RuntimeError(f"pair {pair} {role} worker report invalid: " + "; ".join(errors))
    return report


def run_campaign(args) -> dict[str, object]:
    validate_configuration(
        pairs=args.pairs, iterations=args.iterations, warmup=args.warmup,
        live_counts=args.live_counts,
    )
    blender_python = args.blender_python.resolve()
    if not blender_python.is_file():
        raise ValueError(f"blender-python does not exist: {blender_python}")
    roots = {"baseline": args.baseline_root.resolve(), "candidate": args.candidate_root.resolve()}
    identities = {role: resolve_code_identity(root) for role, root in roots.items()}
    pairs = []
    process_tokens: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="fluidsim-whitewater-campaign-") as temporary:
        temp_root = Path(temporary)
        for pair in range(1, args.pairs + 1):
            order = pair_order(pair)
            legs = []
            reports = {}
            for sequence, role in enumerate(order):
                output = temp_root / f"pair-{pair:02d}-{sequence}-{role}.json"
                report = _run_worker_process(
                    blender_python=blender_python, addon_root=roots[role], role=role,
                    pair=pair, args=args, output=output,
                )
                process_token = str(report["environment"]["process_instance_token"])
                if process_token in process_tokens:
                    raise RuntimeError(f"worker process token was reused: {process_token}")
                process_tokens.add(process_token)
                reports[role] = report
                legs.append(report)
            if _correctness_signature(reports["baseline"]) != _correctness_signature(reports["candidate"]):
                raise RuntimeError(f"pair {pair} baseline/candidate finalization outputs differ")
            pairs.append({
                "pair": pair,
                "order": list(order),
                "legs": legs,
                "comparison": _comparison(reports["baseline"], reports["candidate"]),
            })
    report = {
        "schema": SCHEMA,
        "result": RESULT,
        "measurement_kind": MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "raw paired observation only; requires independent evidence review",
        "campaign_provenance": {
            "orchestrator_python_executable": str(Path(sys.executable).resolve()),
            "benchmark_path": str(Path(__file__).resolve()),
            "benchmark_sha256": _sha256_file(Path(__file__).resolve()),
            "fresh_processes": True,
            "unique_worker_processes": len(process_tokens),
        },
        "code_identities": identities,
        "configuration": {
            "pairs": args.pairs, "iterations": args.iterations, "warmup": args.warmup,
            "live_counts": list(args.live_counts), "seed": args.seed, "device": args.device,
        },
        "pairs": pairs,
        "error_count": 0,
        "errors": [],
    }
    errors = validate_campaign_report(report)
    if errors:
        raise RuntimeError("invalid campaign report: " + "; ".join(errors))
    return report


def validate_campaign_report(report: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if report.get("result") != RESULT:
        errors.append(f"result must be {RESULT}")
    if report.get("measurement_kind") != MEASUREMENT_KIND:
        errors.append(f"measurement_kind must be {MEASUREMENT_KIND}")
    if report.get("approval_eligible") is not False:
        errors.append("raw campaign must not claim approval eligibility")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        errors.append("configuration must be an object")
        configuration = {}
    try:
        validate_configuration(
            pairs=configuration.get("pairs"), iterations=configuration.get("iterations"),
            warmup=configuration.get("warmup"), live_counts=tuple(configuration.get("live_counts", ())),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    identities = report.get("code_identities")
    if not isinstance(identities, dict) or set(identities) != {"baseline", "candidate"}:
        errors.append("code_identities must contain baseline and candidate")
    else:
        for role in ("baseline", "candidate"):
            errors.extend(f"{role}: {error}" for error in _identity_errors(identities[role]))
        baseline_identity = identities["baseline"]
        candidate_identity = identities["candidate"]
        if isinstance(baseline_identity, dict) and isinstance(candidate_identity, dict):
            if baseline_identity == candidate_identity:
                errors.append("baseline and candidate code identities must be distinct")
            if baseline_identity.get("addon_root") == candidate_identity.get("addon_root"):
                errors.append("baseline and candidate must use distinct add-on roots")
            if baseline_identity.get("addon_tree_sha256") == candidate_identity.get("addon_tree_sha256"):
                errors.append("baseline and candidate must use distinct add-on tree digests")
    records = report.get("pairs")
    configured_pairs = configuration.get("pairs")
    expected_pairs = configured_pairs if type(configured_pairs) is int and configured_pairs >= 0 else 0
    if not isinstance(records, list) or len(records) != expected_pairs:
        errors.append("pair records must match configured pair count")
        records = records if isinstance(records, list) else []
    process_tokens = []
    for index, record in enumerate(records, 1):
        expected_order = list(pair_order(index))
        if not isinstance(record, dict) or record.get("pair") != index or record.get("order") != expected_order:
            errors.append(f"pair {index} must preserve alternating order")
            continue
        legs = record.get("legs")
        if not isinstance(legs, list) or [leg.get("role") for leg in legs if isinstance(leg, dict)] != expected_order:
            errors.append(f"pair {index} legs must match recorded order")
            continue
        reports = {leg["role"]: leg for leg in legs}
        for role, leg in reports.items():
            errors.extend(f"pair {index} {role}: {error}" for error in validate_worker_report(leg))
            if leg.get("pair") != index:
                errors.append(f"pair {index} {role} leg has wrong pair provenance")
            if isinstance(identities, dict) and role in identities and leg.get("code_identity") != identities[role]:
                errors.append(f"pair {index} {role} code identity differs from campaign provenance")
            worker_configuration = leg.get("configuration", {})
            expected_worker_configuration = {
                key: configuration.get(key)
                for key in ("iterations", "warmup", "live_counts", "seed")
            }
            if not isinstance(worker_configuration, dict) or any(
                worker_configuration.get(key) != value
                for key, value in expected_worker_configuration.items()
            ):
                errors.append(f"pair {index} {role} configuration differs from campaign")
            environment = leg.get("environment", {})
            if isinstance(environment, dict) and isinstance(environment.get("process_instance_token"), str):
                process_tokens.append(environment["process_instance_token"])
        comparable_environment = ("python_executable", "python_version", "warp_version", "device", "device_name")
        if any(
            reports["baseline"].get("environment", {}).get(key)
            != reports["candidate"].get("environment", {}).get(key)
            for key in comparable_environment
        ):
            errors.append(f"pair {index} baseline/candidate runtime environments differ")
        baseline_source = reports["baseline"].get("seam_provenance", {}).get("source_sha256")
        candidate_source = reports["candidate"].get("seam_provenance", {}).get("source_sha256")
        if baseline_source == candidate_source:
            errors.append(f"pair {index} baseline/candidate must use distinct finalizer source digests")
        try:
            if _correctness_signature(reports["baseline"]) != _correctness_signature(reports["candidate"]):
                errors.append(f"pair {index} baseline/candidate outputs differ")
            if record.get("comparison") != _comparison(reports["baseline"], reports["candidate"]):
                errors.append(f"pair {index} comparison does not match measured samples")
        except (KeyError, TypeError):
            errors.append(f"pair {index} correctness evidence is incomplete")
    provenance = report.get("campaign_provenance")
    if not isinstance(provenance, dict) or provenance.get("fresh_processes") is not True:
        errors.append("campaign provenance must attest fresh processes")
    elif len(process_tokens) != 2 * expected_pairs or len(set(process_tokens)) != len(process_tokens):
        errors.append("every campaign leg must have a unique worker process token")
    elif provenance.get("unique_worker_processes") != len(process_tokens):
        errors.append("campaign unique-worker count must match leg provenance")
    if isinstance(provenance, dict):
        benchmark_digest = provenance.get("benchmark_sha256")
        if not _valid_digest(benchmark_digest):
            errors.append("campaign benchmark provenance must contain a sha256 digest")
        elif any(
            leg.get("seam_provenance", {}).get("benchmark_sha256") != benchmark_digest
            for record in records if isinstance(record, dict)
            for leg in record.get("legs", ()) if isinstance(leg, dict)
        ):
            errors.append("worker benchmark digests must match campaign provenance")
    if report.get("error_count") != 0 or report.get("errors") != []:
        errors.append("passing campaign report must contain no errors")
    return errors


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _add_measurement_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--live-counts", type=_parse_counts, default=DEFAULT_LIVE_COUNTS)
    parser.add_argument("--seed", type=int, default=1729)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run an alternating fresh-process paired campaign")
    run.add_argument("--baseline-root", type=Path, required=True)
    run.add_argument("--candidate-root", type=Path, required=True)
    run.add_argument("--blender-python", type=Path, required=True)
    run.add_argument("--pairs", type=int, default=7)
    _add_measurement_options(run)
    run.add_argument("--json-out", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate a report using system Python")
    validate.add_argument("report", type=Path)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--addon-root", type=Path, required=True)
    worker.add_argument("--role", choices=("baseline", "candidate"), required=True)
    worker.add_argument("--pair", type=int, required=True)
    _add_measurement_options(worker)
    worker.add_argument("--json-out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            with args.report.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            errors = validate_campaign_report(report)
            if errors:
                raise ValueError("; ".join(errors))
            print(f"WHITEWATER-FINALIZATION-REPORT-PASS: {args.report.resolve()}")
            return 0
        if args.command == "worker":
            validate_configuration(
                pairs=1, iterations=args.iterations, warmup=args.warmup,
                live_counts=args.live_counts,
            )
            identity = resolve_code_identity(args.addon_root)
            report = run_worker(args, identity)
            _atomic_json(args.json_out, report)
            print(f"{WORKER_RESULT}: {args.json_out.resolve()}")
            return 0
        report = run_campaign(args)
        _atomic_json(args.json_out, report)
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"WHITEWATER-FINALIZATION-BENCHMARK-FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{RESULT}: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
