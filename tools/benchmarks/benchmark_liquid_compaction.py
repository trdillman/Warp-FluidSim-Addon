"""Benchmark ``ParticleSystem.compact`` in fresh Blender CUDA processes.

``run`` executes an AB/BA alternating (ABBA-balanced) paired campaign.  Each
leg is a new Blender 5.1 bundled-Python process.  Worker and campaign reports
are provenance-bound raw observations and are never approval-eligible.
``validate`` intentionally runs under ordinary system Python.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import inspect
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


SCHEMA = "fluidsim-liquid-compaction-campaign-v1"
WORKER_SCHEMA = "fluidsim-liquid-compaction-worker-v1"
RESULT = "LIQUID-COMPACTION-BENCHMARK-PASS"
WORKER_RESULT = "LIQUID-COMPACTION-WORKER-PASS"
MEASUREMENT_KIND = "targeted_liquid_particle_compaction"
PACKAGE_NAME = "fluidSim_warp_blender51"
DEFAULT_PARTICLE_COUNTS = (50_000, 250_000, 1_000_000)
DEATH_PATTERNS = ("zero-death", "one-death", "half-death", "last-dead")
OPTIONAL_CHANNELS = ("color", "source_id", "uid")
CHANNEL_NAMES = (
    "position", "velocity", "age", "life", "temperature", "viscosity",
    "density", "time_offset", "color", "source_id", "uid", "alive",
)
TELEMETRY_KEYS = (
    "array_scans_per_iteration",
    "kernel_launches_per_iteration",
    "k_last_offset_count_per_iteration",
    "k_compact_alive_scatter_per_iteration",
    "k_compact_vec4_scatter_per_iteration",
    "k_compact_int_scatter_per_iteration",
)


def validate_configuration(
    *, pairs: int, iterations: int, warmup: int, particle_counts: tuple[int, ...],
) -> None:
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if type(iterations) is not int or iterations < 30:
        raise ValueError("iterations must be at least 30")
    if type(warmup) is not int or warmup < 1:
        raise ValueError("warmup must be a positive integer")
    if not particle_counts or any(
        type(value) is not int or value < 50_000 for value in particle_counts
    ):
        raise ValueError("particle-counts must contain integers of at least 50000")
    if len(set(particle_counts)) != len(particle_counts):
        raise ValueError("particle-counts must not contain duplicates")


def pair_order(pair: int) -> tuple[str, str]:
    if type(pair) is not int or pair < 1:
        raise ValueError("pair must be a positive integer")
    return ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")


def _parse_counts(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "particle-counts must be comma-separated integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("particle-counts must not be empty")
    return values


def death_mask(pattern: str, count: int):
    """Return a deterministic keep mask covering production edge distributions."""
    import numpy as np

    if pattern not in DEATH_PATTERNS:
        raise ValueError(f"unknown death pattern: {pattern}")
    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer of at least 2")
    alive = np.ones(count, dtype=np.int32)
    if pattern == "one-death":
        alive[count // 3] = 0
    elif pattern == "half-death":
        alive[::2] = 0
    elif pattern == "last-dead":
        alive[-1] = 0
    return alive


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
    root_text = str(root.resolve())
    sys.path[:] = [root_text] + [item for item in sys.path if item != root_text]
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
    path = Path(tempfile.mkdtemp(prefix="fluidsim-warp-compaction-cache-", dir=parent))
    wp.config.kernel_cache_dir = str(path)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(path)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _sha256_array(value) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _fixture_arrays(count: int, seed: int, pattern: str) -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(seed + count + DEATH_PATTERNS.index(pattern) * 1_000_003)
    values = {
        "position": np.ascontiguousarray(rng.uniform(-2.0, 2.0, (count, 3)), dtype=np.float32),
        "velocity": np.ascontiguousarray(rng.normal(0.0, 1.0, (count, 3)), dtype=np.float32),
        "age": np.ascontiguousarray(rng.uniform(0.0, 3.0, count), dtype=np.float32),
        "life": np.ascontiguousarray(rng.uniform(3.0, 8.0, count), dtype=np.float32),
        "temperature": np.ascontiguousarray(rng.uniform(270.0, 350.0, count), dtype=np.float32),
        "viscosity": np.ascontiguousarray(rng.uniform(0.0, 2.0, count), dtype=np.float32),
        "density": np.ascontiguousarray(rng.uniform(800.0, 1200.0, count), dtype=np.float32),
        "time_offset": np.ascontiguousarray(rng.uniform(0.0, 1.0 / 24.0, count), dtype=np.float32),
        "color": np.ascontiguousarray(rng.uniform(0.0, 1.0, (count, 4)), dtype=np.float32),
        "source_id": np.ascontiguousarray(np.arange(count, dtype=np.int32) % 97),
        "uid": np.ascontiguousarray(np.arange(count, dtype=np.int32) + 10_000),
        "alive": death_mask(pattern, count),
    }
    return values


def _expected_output(fixture: dict[str, object]) -> tuple[dict[str, object], object]:
    import numpy as np

    survivor_indices = np.flatnonzero(fixture["alive"]).astype(np.int32)
    output = {}
    for name in CHANNEL_NAMES:
        if name == "alive":
            output[name] = np.ones(len(survivor_indices), dtype=np.int32)
        else:
            output[name] = np.ascontiguousarray(fixture[name][survivor_indices])
    return output, survivor_indices


def _device_fixture(wp, device, fixture: dict[str, object]) -> dict[str, object]:
    dtypes = {
        "position": wp.vec3, "velocity": wp.vec3, "age": float, "life": float,
        "temperature": float, "viscosity": float, "density": float,
        "time_offset": float, "color": wp.vec4, "source_id": wp.int32,
        "uid": wp.int32, "alive": wp.int32,
    }
    return {name: wp.array(value, dtype=dtypes[name], device=device) for name, value in fixture.items()}


def _restore_pool(wp, pool, device_fixture: dict[str, object], count_array, count: int) -> None:
    wp.copy(pool.position, device_fixture["position"], count=count)
    wp.copy(pool.velocity, device_fixture["velocity"], count=count)
    for name in pool.ATTR_NAMES:
        wp.copy(pool.attrs[name], device_fixture[name], count=count)
    wp.copy(pool.time_offset, device_fixture["time_offset"], count=count)
    wp.copy(pool.vec4_attrs["color"], device_fixture["color"], count=count)
    wp.copy(pool.int_attrs["source_id"], device_fixture["source_id"], count=count)
    wp.copy(pool.int_attrs["uid"], device_fixture["uid"], count=count)
    wp.copy(pool.alive, device_fixture["alive"], count=count)
    wp.copy(pool.count, count_array, count=1)
    pool.host_count = count


def _pool_output(pool, count: int) -> dict[str, object]:
    import numpy as np

    return {
        "position": np.ascontiguousarray(pool.position.numpy()[:count], dtype=np.float32),
        "velocity": np.ascontiguousarray(pool.velocity.numpy()[:count], dtype=np.float32),
        **{
            name: np.ascontiguousarray(pool.attrs[name].numpy()[:count], dtype=np.float32)
            for name in pool.ATTR_NAMES
        },
        "time_offset": np.ascontiguousarray(pool.time_offset.numpy()[:count], dtype=np.float32),
        "color": np.ascontiguousarray(pool.vec4_attrs["color"].numpy()[:count], dtype=np.float32),
        "source_id": np.ascontiguousarray(pool.int_attrs["source_id"].numpy()[:count], dtype=np.int32),
        "uid": np.ascontiguousarray(pool.int_attrs["uid"].numpy()[:count], dtype=np.int32),
        "alive": np.ascontiguousarray(pool.alive.numpy()[:count], dtype=np.int32),
    }


def _run_scenario(args, particles_mod, wp, count: int, pattern: str) -> dict[str, object]:
    import numpy as np

    fixture = _fixture_arrays(count, args.seed, pattern)
    expected, survivor_indices = _expected_output(fixture)
    expected_count = len(survivor_indices)
    expected_digests = {name: _sha256_array(value) for name, value in expected.items()}
    pool = particles_mod.ParticleSystem(count, args.device, optional_channels=OPTIONAL_CHANNELS)
    device_fixture = _device_fixture(wp, args.device, fixture)
    count_array = wp.array([count], dtype=wp.int32, device=args.device)

    original_launch = particles_mod.wp.launch
    original_scan = particles_mod.wp.utils.array_scan
    current: dict[str, int] | None = None
    known_kernels = {
        name: getattr(particles_mod, name, None) for name in (
            "k_last_offset_count", "k_compact_alive_scatter",
            "k_compact_vec4_scatter", "k_compact_int_scatter",
        )
    }

    def counted_launch(kernel, *call_args, **call_kwargs):
        if current is not None:
            current["kernel_launches"] += 1
            for name, known in known_kernels.items():
                if known is not None and kernel is known:
                    current[name] += 1
        return original_launch(kernel, *call_args, **call_kwargs)

    def counted_scan(*call_args, **call_kwargs):
        if current is not None:
            current["array_scans"] += 1
        return original_scan(*call_args, **call_kwargs)

    samples: list[float] = []
    observed_counts: list[int] = []
    telemetry = {key: [] for key in TELEMETRY_KEYS}
    particles_mod.wp.launch = counted_launch
    particles_mod.wp.utils.array_scan = counted_scan
    try:
        for _ in range(args.warmup):
            _restore_pool(wp, pool, device_fixture, count_array, count)
            wp.synchronize_device(args.device)
            pool.compact(count_hint=count)
            wp.synchronize_device(args.device)
        for _ in range(args.iterations):
            _restore_pool(wp, pool, device_fixture, count_array, count)
            wp.synchronize_device(args.device)
            bucket = {"array_scans": 0, "kernel_launches": 0, **{name: 0 for name in known_kernels}}
            current = bucket
            started = time.perf_counter()
            pool.compact(count_hint=count)
            wp.synchronize_device(args.device)
            samples.append(float(time.perf_counter() - started))
            current = None
            observed_counts.append(int(pool.count.numpy()[0]))
            telemetry["array_scans_per_iteration"].append(bucket["array_scans"])
            telemetry["kernel_launches_per_iteration"].append(bucket["kernel_launches"])
            for name in known_kernels:
                telemetry[f"{name}_per_iteration"].append(bucket[name])
        actual_count = int(pool.count.numpy()[0])
        actual = _pool_output(pool, actual_count)
        actual_survivor_indices = np.ascontiguousarray(
            actual["uid"] - np.int32(10_000), dtype=np.int32
        )
        exact = actual_count == expected_count and all(
            np.array_equal(actual[name], expected[name]) for name in CHANNEL_NAMES
        )
        actual_digests = {name: _sha256_array(value) for name, value in actual.items()}
        return {
            "particle_count": count,
            "death_pattern": pattern,
            "optional_channels": list(OPTIONAL_CHANNELS),
            "calibration": {
                "fixture_seed": args.seed,
                "death_count": count - expected_count,
                "expected_survivor_count": expected_count,
                "survivor_indices_sha256": _sha256_array(survivor_indices),
                "expected_channel_digests": expected_digests,
            },
            "samples_seconds": samples,
            "sample_median_seconds": statistics.median(samples),
            "sample_min_seconds": min(samples),
            "sample_max_seconds": max(samples),
            "device_synchronized_warmup": True,
            "device_synchronized_samples": True,
            "observed_survivor_counts": observed_counts,
            "telemetry_method": "process-local wrappers around wp.launch and wp.utils.array_scan",
            "telemetry": telemetry,
            "output": {
                "count_array": [actual_count],
                "survivor_indices_sha256": _sha256_array(actual_survivor_indices),
                "channel_digests": actual_digests,
                "exact_stable_survivor_arrays": exact,
            },
        }
    finally:
        current = None
        particles_mod.wp.launch = original_launch
        particles_mod.wp.utils.array_scan = original_scan


def run_worker(args, identity: dict[str, object]) -> dict[str, object]:
    blender_version = _require_blender_51_python()
    package_path = _bootstrap_addon(Path(identity["addon_root"]))
    import warp as wp
    from fluidSim_warp_blender51.sim import particles as particles_mod

    cache = configure_writable_warp_cache(wp)
    wp.init()
    if not str(args.device).startswith("cuda") or not wp.is_cuda_available():
        raise RuntimeError("liquid-compaction benchmark requires an available CUDA device")
    device = wp.get_device(args.device)
    scenarios = [
        _run_scenario(args, particles_mod, wp, count, pattern)
        for count in args.particle_counts for pattern in DEATH_PATTERNS
    ]
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
            "callable": "ParticleSystem.compact",
            "module_path": str(Path(particles_mod.__file__).resolve()),
            "source_sha256": hashlib.sha256(
                inspect.getsource(particles_mod.ParticleSystem.compact).encode("utf-8")
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
            "fresh_process_import": True,
        },
        "configuration": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "particle_counts": list(args.particle_counts),
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
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


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


def _expected_scenario_keys(configuration: dict[str, object]) -> list[tuple[int, str]]:
    return [
        (count, pattern)
        for count in configuration.get("particle_counts", ())
        for pattern in DEATH_PATTERNS
    ]


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
    if not isinstance(seam, dict) or seam.get("callable") != "ParticleSystem.compact":
        errors.append("seam provenance must identify ParticleSystem.compact")
    elif not _valid_digest(seam.get("source_sha256")) or not _valid_digest(seam.get("benchmark_sha256")):
        errors.append("seam provenance digests must be sha256")
    environment = report.get("environment")
    required_environment = (
        "python_executable", "python_version", "warp_version", "warp_path", "device", "process_id",
    )
    if not isinstance(environment, dict) or any(environment.get(key) in (None, "") for key in required_environment):
        errors.append("environment must contain runtime and process provenance")
    elif environment.get("fresh_process_import") is not True:
        errors.append("environment must attest fresh-process import")
    elif environment.get("blender_version") != "5.1" or not str(environment.get("device", "")).startswith("cuda"):
        errors.append("environment must identify Blender 5.1 bundled Python and CUDA")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        errors.append("configuration must be an object")
        configuration = {}
    try:
        validate_configuration(
            pairs=1,
            iterations=configuration.get("iterations"),
            warmup=configuration.get("warmup"),
            particle_counts=tuple(configuration.get("particle_counts", ())),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    scenarios = report.get("scenarios")
    expected_keys = _expected_scenario_keys(configuration)
    observed_keys = [
        (item.get("particle_count"), item.get("death_pattern"))
        for item in scenarios if isinstance(item, dict)
    ] if isinstance(scenarios, list) else []
    if not isinstance(scenarios, list) or observed_keys != expected_keys:
        errors.append("scenarios must match every particle count and death pattern in order")
        scenarios = scenarios if isinstance(scenarios, list) else []
    iterations = configuration.get("iterations")
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            errors.append("each scenario must be an object")
            continue
        label = f"{scenario.get('particle_count')} {scenario.get('death_pattern')}"
        if scenario.get("optional_channels") != list(OPTIONAL_CHANNELS):
            errors.append(f"{label} must enable color/source_id/uid optional channels")
        calibration = scenario.get("calibration")
        output = scenario.get("output")
        if not isinstance(calibration, dict) or not isinstance(output, dict):
            errors.append(f"{label} calibration and output must be objects")
            continue
        expected_count = calibration.get("expected_survivor_count")
        particle_count = scenario.get("particle_count")
        pattern = scenario.get("death_pattern")
        expected_deaths = None
        if type(particle_count) is int:
            expected_deaths = {
                "zero-death": 0,
                "one-death": 1,
                "half-death": (particle_count + 1) // 2,
                "last-dead": 1,
            }.get(pattern)
        if (
            expected_deaths is None
            or calibration.get("death_count") != expected_deaths
            or expected_count != particle_count - expected_deaths
            or not _valid_digest(calibration.get("survivor_indices_sha256"))
        ):
            errors.append(f"{label} survivor calibration does not match its death distribution")
        expected_digests = calibration.get("expected_channel_digests")
        actual_digests = output.get("channel_digests")
        if set(expected_digests or {}) != set(CHANNEL_NAMES) or any(
            not _valid_digest(value) for value in (expected_digests or {}).values()
        ):
            errors.append(f"{label} expected channel digests are invalid")
        if set(actual_digests or {}) != set(CHANNEL_NAMES) or any(
            not _valid_digest(value) for value in (actual_digests or {}).values()
        ):
            errors.append(f"{label} output channel digests are invalid")
        if (
            output.get("count_array") != [expected_count]
            or output.get("survivor_indices_sha256") != calibration.get("survivor_indices_sha256")
            or actual_digests != expected_digests
            or output.get("exact_stable_survivor_arrays") is not True
        ):
            errors.append(f"{label} exact stable survivor arrays do not match calibration")
        samples = scenario.get("samples_seconds")
        if not isinstance(samples, list) or len(samples) != iterations or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in (samples or ())
        ):
            errors.append(f"{label} samples must be positive, finite, and complete")
        elif not math.isclose(float(scenario.get("sample_median_seconds", -1)), statistics.median(samples)):
            errors.append(f"{label} sample median must match samples")
        if scenario.get("device_synchronized_warmup") is not True or scenario.get("device_synchronized_samples") is not True:
            errors.append(f"{label} warmup and measured samples must be device-synchronized")
        if scenario.get("observed_survivor_counts") != [expected_count] * iterations:
            errors.append(f"{label} observed survivor counts must be exact")
        telemetry = scenario.get("telemetry")
        if not isinstance(telemetry, dict) or any(
            not isinstance(telemetry.get(key), list)
            or len(telemetry[key]) != iterations
            or any(type(value) is not int or value < 0 for value in telemetry[key])
            for key in TELEMETRY_KEYS
        ):
            errors.append(f"{label} launch telemetry must match measured iterations")
    if report.get("error_count") != 0 or report.get("errors") != []:
        errors.append("passing worker report must contain no errors")
    return errors


def _correctness_signature(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "particle_count": scenario["particle_count"],
            "death_pattern": scenario["death_pattern"],
            "optional_channels": scenario["optional_channels"],
            "calibration": scenario["calibration"],
            "output": scenario["output"],
        }
        for scenario in report["scenarios"]
    ]


def _comparison(baseline: dict[str, object], candidate: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for base, cand in zip(baseline["scenarios"], candidate["scenarios"]):
        base_median = float(base["sample_median_seconds"])
        candidate_median = float(cand["sample_median_seconds"])
        rows.append({
            "particle_count": base["particle_count"],
            "death_pattern": base["death_pattern"],
            "baseline_median_seconds": base_median,
            "candidate_median_seconds": candidate_median,
            "candidate_over_baseline_ratio": candidate_median / base_median,
            "baseline_k_last_offset_count_per_iteration": base["telemetry"]["k_last_offset_count_per_iteration"],
            "candidate_k_last_offset_count_per_iteration": cand["telemetry"]["k_last_offset_count_per_iteration"],
        })
    return rows


def _last_offset_launches(records: list[dict[str, object]], role: str) -> int:
    total = 0
    for record in records:
        leg = next(item for item in record["legs"] if item["role"] == role)
        for scenario in leg["scenarios"]:
            total += sum(scenario["telemetry"]["k_last_offset_count_per_iteration"])
    return total


def _run_worker_process(
    *, blender_python: Path, addon_root: Path, role: str, pair: int, args, output: Path,
) -> dict[str, object]:
    command = [
        str(blender_python), str(Path(__file__).resolve()), "worker",
        "--addon-root", str(addon_root.resolve()), "--role", role, "--pair", str(pair),
        "--device", args.device, "--iterations", str(args.iterations),
        "--warmup", str(args.warmup),
        "--particle-counts", ",".join(map(str, args.particle_counts)),
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
        particle_counts=args.particle_counts,
    )
    blender_python = args.blender_python.resolve()
    if not blender_python.is_file():
        raise ValueError(f"blender-python does not exist: {blender_python}")
    roots = {"baseline": args.baseline_root.resolve(), "candidate": args.candidate_root.resolve()}
    identities = {role: resolve_code_identity(root) for role, root in roots.items()}
    records = []
    process_ids: set[int] = set()
    with tempfile.TemporaryDirectory(prefix="fluidsim-compaction-campaign-") as temporary:
        temporary_root = Path(temporary)
        for pair in range(1, args.pairs + 1):
            order = pair_order(pair)
            reports = {}
            legs = []
            for sequence, role in enumerate(order):
                output = temporary_root / f"pair-{pair:02d}-{sequence}-{role}.json"
                report = _run_worker_process(
                    blender_python=blender_python, addon_root=roots[role], role=role,
                    pair=pair, args=args, output=output,
                )
                process_id = int(report["environment"]["process_id"])
                if process_id in process_ids:
                    raise RuntimeError(f"worker process id was reused: {process_id}")
                process_ids.add(process_id)
                reports[role] = report
                legs.append(report)
            if _correctness_signature(reports["baseline"]) != _correctness_signature(reports["candidate"]):
                raise RuntimeError(f"pair {pair} baseline/candidate compaction outputs differ")
            records.append({
                "pair": pair, "order": list(order), "legs": legs,
                "comparison": _comparison(reports["baseline"], reports["candidate"]),
            })
    baseline_launches = _last_offset_launches(records, "baseline")
    candidate_launches = _last_offset_launches(records, "candidate")
    report = {
        "schema": SCHEMA,
        "result": RESULT,
        "measurement_kind": MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "raw paired observation only; no standalone approval claim",
        "campaign_provenance": {
            "orchestrator_python_executable": str(Path(sys.executable).resolve()),
            "benchmark_path": str(Path(__file__).resolve()),
            "benchmark_sha256": _sha256_file(Path(__file__).resolve()),
            "fresh_processes": True,
            "unique_worker_processes": len(process_ids),
            "order_policy": "AB/BA alternating (ABBA balanced)",
        },
        "code_identities": identities,
        "configuration": {
            "pairs": args.pairs, "iterations": args.iterations, "warmup": args.warmup,
            "particle_counts": list(args.particle_counts), "seed": args.seed, "device": args.device,
        },
        "pairs": records,
        "launch_change": {
            "kernel": "k_last_offset_count",
            "baseline_launches": baseline_launches,
            "candidate_launches": candidate_launches,
            "removed_in_candidate": baseline_launches > 0 and candidate_launches == 0,
        },
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
            warmup=configuration.get("warmup"),
            particle_counts=tuple(configuration.get("particle_counts", ())),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    identities = report.get("code_identities")
    if not isinstance(identities, dict) or set(identities) != {"baseline", "candidate"}:
        errors.append("code_identities must contain baseline and candidate")
    else:
        for role in ("baseline", "candidate"):
            errors.extend(f"{role}: {error}" for error in _identity_errors(identities[role]))
    records = report.get("pairs")
    expected_pairs = configuration.get("pairs")
    if not isinstance(records, list) or len(records) != expected_pairs:
        errors.append("pair records must match configured pair count")
        records = records if isinstance(records, list) else []
    process_ids = []
    for index, record in enumerate(records, 1):
        expected_order = list(pair_order(index))
        if not isinstance(record, dict) or record.get("pair") != index or record.get("order") != expected_order:
            errors.append(f"pair {index} must preserve AB/BA alternating order")
            continue
        legs = record.get("legs")
        if not isinstance(legs, list) or [leg.get("role") for leg in legs if isinstance(leg, dict)] != expected_order:
            errors.append(f"pair {index} legs must match recorded order")
            continue
        reports = {leg["role"]: leg for leg in legs}
        for role, leg in reports.items():
            errors.extend(f"pair {index} {role}: {error}" for error in validate_worker_report(leg))
            if leg.get("pair") != index:
                errors.append(f"pair {index} {role} worker pair provenance is inconsistent")
            if isinstance(identities, dict) and leg.get("code_identity") != identities.get(role):
                errors.append(f"pair {index} {role} code identity differs from campaign provenance")
            environment = leg.get("environment", {})
            if isinstance(environment, dict) and type(environment.get("process_id")) is int:
                process_ids.append(environment["process_id"])
        if _correctness_signature(reports["baseline"]) != _correctness_signature(reports["candidate"]):
            errors.append(f"pair {index} baseline/candidate outputs differ")
        if record.get("comparison") != _comparison(reports["baseline"], reports["candidate"]):
            errors.append(f"pair {index} comparison does not match raw worker observations")
    provenance = report.get("campaign_provenance")
    if not isinstance(provenance, dict) or provenance.get("fresh_processes") is not True:
        errors.append("campaign provenance must attest fresh processes")
    elif len(process_ids) != 2 * expected_pairs or len(set(process_ids)) != len(process_ids):
        errors.append("every campaign leg must have a unique worker process id")
    elif provenance.get("unique_worker_processes") != len(process_ids):
        errors.append("campaign unique-worker count must match leg provenance")
    change = report.get("launch_change")
    if records:
        baseline_launches = _last_offset_launches(records, "baseline")
        candidate_launches = _last_offset_launches(records, "candidate")
    else:
        baseline_launches = candidate_launches = 0
    if (
        not isinstance(change, dict)
        or change.get("kernel") != "k_last_offset_count"
        or change.get("baseline_launches") != baseline_launches
        or change.get("candidate_launches") != candidate_launches
        or change.get("removed_in_candidate") is not True
        or baseline_launches <= 0
        or candidate_launches != 0
    ):
        errors.append("campaign must truthfully show k_last_offset_count removed in candidate")
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
    parser.add_argument("--particle-counts", type=_parse_counts, default=DEFAULT_PARTICLE_COUNTS)
    parser.add_argument("--seed", type=int, default=1729)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a seven-pair ABBA-balanced fresh-process campaign")
    run.add_argument("--baseline-root", type=Path, required=True)
    run.add_argument("--candidate-root", type=Path, required=True)
    run.add_argument("--blender-python", type=Path, required=True)
    run.add_argument("--pairs", type=int, default=7)
    _add_measurement_options(run)
    run.add_argument("--json-out", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate a campaign report using system Python")
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
            print(f"LIQUID-COMPACTION-REPORT-PASS: {args.report.resolve()}")
            return 0
        if args.command == "worker":
            validate_configuration(
                pairs=1, iterations=args.iterations, warmup=args.warmup,
                particle_counts=args.particle_counts,
            )
            identity = resolve_code_identity(args.addon_root)
            report = run_worker(args, identity)
            _atomic_json(args.json_out, report)
            print(f"{WORKER_RESULT}: {args.json_out.resolve()}")
            return 0
        report = run_campaign(args)
        _atomic_json(args.json_out, report)
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"LIQUID-COMPACTION-BENCHMARK-FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{RESULT}: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
