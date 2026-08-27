"""Fresh-Blender worker for immutable cache-v2 playback measurements.

Run::

    blender --background --factory-startup --python tests/blender_cache_playback_benchmark.py -- \
        --addon-root C:/clean/checkout --cache-dir C:/external/cache-leg \
        --json-out C:/external/raw-leg.json

Cache construction and a ten-frame warm-up happen before instrumentation is
recorded.  The measured region calls the production ``play_cache_frame`` seam
for sixty distinct frames and records its read, manifest/metadata, signature,
publication, exact payload, and exact status behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:  # Helpers remain importable under system Python unit tests.
    import bpy
except ModuleNotFoundError:  # pragma: no cover - Blender execution owns this branch
    bpy = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import benchmark_cache_playback_validation as validation

MESH_NAME = "CachePlaybackStaticMeshSource"
GRID_SIDE = 65
PARTICLE_COUNT = 4096
FIXED_TIMESTAMP = "2026-01-01T00:00:00Z"
NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,gpu_uuid",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_TIMEOUT_SECONDS = 5.0


def payload_sha256(payload: dict[str, object]) -> str:
    """Hash exact named array payloads including dtype and shape."""
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = np.ascontiguousarray(payload[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def classify_nvidia_smi_process_rows(
    stdout: str, *, current_pid: int,
) -> list[dict[str, object]]:
    """Classify strict nvidia-smi process rows as this worker or contention."""
    result: list[dict[str, object]] = []
    raw_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    parsed_rows = list(csv.reader(raw_lines, skipinitialspace=True))
    for raw, fields in zip(raw_lines, parsed_rows, strict=True):
        normalized = [field.strip() for field in fields]
        if len(normalized) != 3:
            raise ValueError(f"ambiguous nvidia-smi process row: {raw!r}")
        raw_pid, process_name, gpu_uuid = normalized
        unsupported = {"n/a", "not supported", "[not supported]"}
        if (
            not raw_pid.isdecimal()
            or int(raw_pid) < 1
            or not process_name
            or not gpu_uuid
            or process_name.casefold() in unsupported
            or gpu_uuid.casefold() in unsupported
        ):
            raise ValueError(f"ambiguous nvidia-smi process row: {raw!r}")
        pid = int(raw_pid)
        result.append({
            "raw": raw,
            "pid": pid,
            "process_name": process_name,
            "gpu_uuid": gpu_uuid,
            "classification": "self" if pid == current_pid else "other",
        })
    return result


def _capture_contention_sample(phase: str) -> dict[str, object]:
    executable = shutil.which(NVIDIA_SMI_QUERY[0])
    if not executable:
        raise RuntimeError("nvidia-smi process query is unavailable")
    try:
        completed = subprocess.run(
            [executable, *NVIDIA_SMI_QUERY[1:]],
            capture_output=True,
            text=True,
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi process query is unavailable: {exc}") from exc
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    stderr_lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    if completed.returncode != 0:
        detail = "; ".join(stderr_lines or stdout_lines) or "no diagnostic output"
        raise RuntimeError(
            f"nvidia-smi process query is unavailable (exit {completed.returncode}): {detail}"
        )
    classifications = classify_nvidia_smi_process_rows(
        completed.stdout, current_pid=os.getpid(),
    )
    return {
        "phase": phase,
        "query_available": True,
        "returncode": completed.returncode,
        "raw_stdout_lines": stdout_lines,
        "raw_stderr_lines": stderr_lines,
        "classifications": classifications,
        "query_executable": str(Path(executable).resolve()),
    }


def _contention_record(observations: list[dict[str, object]]) -> dict[str, object]:
    other_pids = sorted({
        int(classified["pid"])
        for observation in observations
        for classified in observation["classifications"]
        if classified["classification"] == "other"
    })
    executables = {observation["query_executable"] for observation in observations}
    if len(executables) != 1:
        raise RuntimeError("nvidia-smi process-query executable changed during measurement")
    return {
        "other_gpu_processes": len(other_pids),
        "query_available": True,
        "classification": (
            "zero-other-gpu-processes" if not other_pids else "other-gpu-processes-observed"
        ),
        "method": (
            "bounded nvidia-smi compute-process query immediately before and after "
            "measured segment"
        ),
        "query_executable": executables.pop(),
        "query_command": list(NVIDIA_SMI_QUERY),
        "samples": [
            {key: value for key, value in observation.items() if key != "query_executable"}
            for observation in observations
        ],
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_cache_destination(addon_root: Path, cache_dir: Path) -> None:
    """Require a new or empty cache directory outside the measured checkout."""
    root = Path(addon_root).resolve()
    destination = Path(cache_dir).resolve()
    if _is_relative_to(destination, root):
        raise ValueError("cache directory must be outside the add-on checkout")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("cache directory must be absent or empty")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"cache tree contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _addon_tree_sha256(package_root: Path) -> str:
    digest = hashlib.sha256()
    files = []
    for path in package_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append((path.relative_to(package_root).as_posix(), path))
    if not files:
        raise ValueError(f"add-on package contains no files: {package_root}")
    for relative, path in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise ValueError(f"could not resolve Git identity for {root}: {detail}")
    return completed.stdout.strip()


def resolve_code_identity(addon_root: Path) -> dict[str, object]:
    root = Path(addon_root).resolve()
    package_root = root / validation.PACKAGE_NAME
    if not (package_root / "__init__.py").is_file():
        raise ValueError(f"addon-root does not contain {validation.PACKAGE_NAME}: {root}")
    git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    status = _git(git_root, "status", "--porcelain=v1", "--untracked-files=all")
    addon_status = _git(
        git_root, "status", "--porcelain=v1", "--untracked-files=all", "--",
        package_root.relative_to(git_root).as_posix(),
    )
    identity = {
        "addon_root": str(root),
        "git_root": str(git_root),
        "git_head": _git(git_root, "rev-parse", "HEAD"),
        "checkout_dirty": bool(status),
        "addon_dirty": bool(addon_status),
        "addon_tree_sha256": _addon_tree_sha256(package_root),
        "worker_sha256": _sha256_file(Path(__file__).resolve()),
    }
    errors = validation._identity_errors(identity)
    if errors:
        raise ValueError("; ".join(errors))
    return identity


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def _load_repo_addon(root: Path):
    import importlib.util

    loader_path = root / "tools" / "repo_addon_loader.py"
    spec = importlib.util.spec_from_file_location("_cache_playback_repo_addon_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repo add-on loader from {loader_path}")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    return loader.enable_repo_addon(str(root), default_set=False, persistent=False)


def _create_heavy_static_mesh_source(scene):
    vertices = [
        (float(x) / (GRID_SIDE - 1) * 4.0 - 2.0,
         float(y) / (GRID_SIDE - 1) * 4.0 - 2.0,
         0.5 + 0.05 * math.sin(x * 0.25) * math.cos(y * 0.25))
        for y in range(GRID_SIDE) for x in range(GRID_SIDE)
    ]
    triangles = []
    for y in range(GRID_SIDE - 1):
        for x in range(GRID_SIDE - 1):
            lower = y * GRID_SIDE + x
            triangles.append((lower, lower + 1, lower + GRID_SIDE + 1))
            triangles.append((lower, lower + GRID_SIDE + 1, lower + GRID_SIDE))
    mesh = bpy.data.meshes.new(MESH_NAME + "Mesh")
    mesh.from_pydata(vertices, [], triangles)
    mesh.update()
    source = bpy.data.objects.new(MESH_NAME, mesh)
    scene.collection.objects.link(source)
    source.fluidsim.role = "SOURCE"
    source.fluidsim.enabled = True
    if hasattr(source.fluidsim, "source_shape"):
        source.fluidsim.source_shape = "MESH"
    source.matrix_world.identity()
    return source, len(vertices), len(triangles)


def _configure_scene(scene, cache_dir: Path) -> None:
    fs = scene.fluidsim
    fs.cache_path = str(cache_dir)
    fs.cache_playback_enabled = True
    fs.cache_use_custom_range = True
    fs.cache_frame_start = 1
    fs.cache_frame_end = validation.MIN_CACHE_FRAMES
    fs.show_particles = True
    fs.show_surface = False
    fs.surface_enabled = False
    fs.show_whitewater = False
    fs.whitewater_enabled = False
    fs.particle_display_fraction = 1.0
    fs.particle_attribute_mode = "NONE"
    fs.surface_attribute_mode = "NONE"
    fs.whitewater_attribute_mode = "NONE"
    scene.frame_start = 1
    scene.frame_end = validation.MIN_CACHE_FRAMES
    for name in dir(fs):
        if name.startswith(("surface_attribute_", "particles_attribute_", "whitewater_attribute_")):
            try:
                if isinstance(getattr(fs, name), bool):
                    setattr(fs, name, False)
            except (AttributeError, TypeError):
                pass


def _base_particle_arrays() -> tuple[np.ndarray, np.ndarray]:
    side = 16
    indices = np.arange(PARTICLE_COUNT, dtype=np.int32)
    x = (indices % side).astype(np.float32)
    y = ((indices // side) % side).astype(np.float32)
    z = (indices // (side * side)).astype(np.float32)
    positions = np.column_stack((x, y, z)).astype(np.float32) * np.float32(0.035)
    velocities = np.empty_like(positions)
    velocities[:] = np.array((0.048, -0.024, 0.012), dtype=np.float32)
    return positions, velocities


class _FixtureHostFrame:
    def __init__(self, frame: int):
        base, velocities = _base_particle_arrays()
        offset = np.array(
            (frame * 0.002, math.sin(frame * 0.17) * 0.01, frame * 0.0005),
            dtype=np.float32,
        )
        self.positions = np.ascontiguousarray(base + offset)
        self.velocities = velocities
        self.whitewater = np.zeros((0, 3), dtype=np.float32)
        self.surface_mesh = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        )
        self._frame = frame

    def particle_attr(self, name: str) -> np.ndarray:
        if name in {"age", "time_offset"}:
            return np.full(PARTICLE_COUNT, self._frame / 24.0, dtype=np.float32)
        if name == "life":
            return np.full(PARTICLE_COUNT, 10.0, dtype=np.float32)
        if name == "density":
            return np.full(PARTICLE_COUNT, 1000.0, dtype=np.float32)
        return np.zeros(PARTICLE_COUNT, dtype=np.float32)

    def whitewater_attr(self, name: str) -> np.ndarray:
        if name == "velocity":
            return np.zeros((0, 3), dtype=np.float32)
        if name == "type":
            return np.zeros(0, dtype=np.int32)
        return np.zeros(0, dtype=np.float32)

    def output_attributes(self, settings, output_kind: str, *, positions=None) -> dict:
        return {}


class _FixtureSolver:
    def __init__(self, settings):
        self.settings = settings
        self.frame = 1
        self.sim_time = 1.0 / 24.0
        self.stats = SimpleNamespace(
            whitewater_required=0, whitewater_available=0,
            whitewater_admitted=0, whitewater_dropped=0,
            whitewater_policy="FIXED_CAPACITY_DETERMINISTIC_DROPS",
            whitewater_status="OK",
        )

    def host_frame(self):
        return _FixtureHostFrame(self.frame)


def _deterministic_npz_atomic(path: Path, payload: dict, *, compressed=False):
    """Create byte-reproducible NPZ files by fixing member timestamps/order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(temporary_path, "w", compression=compression) as archive:
            for name in sorted(payload):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asanyarray(payload[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = compression
                info.external_attr = 0o100444 << 16
                archive.writestr(info, buffer.getvalue())
        os.replace(temporary_path, path)
        return _sha256_file(path), path.stat().st_size
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _build_immutable_cache(cache_dir: Path, settings, signatures: dict[str, str]):
    from fluidSim_warp_blender51.io import cache, cache_schema

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_schema.CacheManifestV2.new(
        solver_version="cache-playback-benchmark-fixture-v1",
        scene_key=signatures["scene_key"],
        settings_signature=signatures["settings_signature"],
        source_signature=signatures["source_signature"],
        frame_start=1,
        frame_end=validation.MIN_CACHE_FRAMES,
        checkpoint_interval=1,
        timestamp=FIXED_TIMESTAMP,
    )
    solver = _FixtureSolver(settings)
    original_writer = cache._write_npz_atomic
    cache._write_npz_atomic = _deterministic_npz_atomic
    try:
        for frame in range(1, validation.MIN_CACHE_FRAMES + 1):
            solver.frame = frame
            solver.sim_time = frame / 24.0
            frame_path, frame_digest, frame_size = cache.write_frame_v2_atomic(
                cache_dir,
                solver,
                frame,
                scene_key_value=signatures["scene_key"],
                settings_signature=signatures["settings_signature"],
                source_signature=signatures["source_signature"],
                compressed=False,
            )
            checkpoint_path = cache_dir / "checkpoints" / f"frame_{frame:06d}.fluidchk"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(
                b"cache-playback-benchmark-checkpoint-v1\0" + frame.to_bytes(4, "little")
            )
            entry = cache_schema.FrameEntry(
                frame=frame,
                frame_file=frame_path.relative_to(cache_dir).as_posix(),
                frame_sha256=frame_digest,
                frame_size=frame_size,
                checkpoint_file=checkpoint_path.relative_to(cache_dir).as_posix(),
                checkpoint_sha256=_sha256_file(checkpoint_path),
                checkpoint_size=checkpoint_path.stat().st_size,
                completed_at=FIXED_TIMESTAMP,
                solver_frame=frame,
                simulation_time=frame / 24.0,
                settings_signature=signatures["settings_signature"],
                source_signature=signatures["source_signature"],
            )
            manifest = manifest.with_committed_frame(entry, timestamp=FIXED_TIMESTAMP)
        cache_schema.write_manifest_atomic(cache_schema.manifest_path(cache_dir), manifest)
    finally:
        cache._write_npz_atomic = original_writer
    for path in cache_dir.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    return manifest


def _matrix_sha256(matrix) -> str:
    values = np.asarray([[float(component) for component in row] for row in matrix], dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@dataclass
class _Instrumentation:
    current: dict[str, object] | None = None
    counts: dict[str, int] | None = None
    warmup_counts: dict[str, int] | None = None
    inside_playback_read: bool = False
    phase: str = "idle"

    def __post_init__(self):
        self.counts = {key: 0 for key in validation.COUNT_FIELDS}
        self.warmup_counts = {key: 0 for key in validation.COUNT_FIELDS}

    def record_timing(self, field: str, elapsed: float) -> None:
        if self.current is not None:
            self.current[field] = float(self.current.get(field, 0.0)) + elapsed

    def count(self, field: str) -> None:
        if self.current is not None:
            self.counts[field] += 1
        elif self.phase == "warmup":
            self.warmup_counts[field] += 1


def _run_benchmark(args, identity: dict[str, object]) -> dict[str, object]:
    if bpy is None:
        raise RuntimeError("benchmark must run inside Blender")
    if bpy.app.background is not True:
        raise RuntimeError("cache-playback benchmark requires Blender --background")
    if tuple(bpy.app.version[:2]) != (5, 1):
        raise RuntimeError(f"cache-playback benchmark requires Blender 5.1, got {bpy.app.version_string}")
    if validation.PACKAGE_NAME in sys.modules:
        raise RuntimeError("benchmark requires a fresh process with the add-on not yet imported")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    root = Path(identity["addon_root"])
    addon = _load_repo_addon(root)
    expected_addon = (root / validation.PACKAGE_NAME / "__init__.py").resolve()
    if Path(addon.__file__).resolve() != expected_addon:
        raise RuntimeError(f"loaded add-on from {addon.__file__}, expected {expected_addon}")

    from fluidSim_warp_blender51.blender import (
        cache_identity,
        output_publication,
        playback,
        scene_bridge,
        solver_settings,
    )
    from fluidSim_warp_blender51.io import cache, cache_schema

    scene = bpy.context.scene
    _configure_scene(scene, args.cache_dir)
    source, vertex_count, triangle_count = _create_heavy_static_mesh_source(scene)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    signatures_before = cache_identity.signature_bundle(scene, depsgraph)
    settings = solver_settings.build_settings(scene)

    # Cache construction is complete before any warm-up or timed invocation.
    manifest = _build_immutable_cache(args.cache_dir, settings, signatures_before)
    manifest_path = cache_schema.manifest_path(args.cache_dir)
    manifest_digest_before = _sha256_file(manifest_path)
    tree_digest_before = _tree_sha256(args.cache_dir)
    metadata_identity = hashlib.sha256(json.dumps(
        {
            "format_version": manifest.format_version,
            "scene_key": manifest.scene_key,
            "settings_signature": manifest.settings_signature,
            "source_signature": manifest.source_signature,
            "frame_start": manifest.frame_start,
            "frame_end": manifest.frame_end,
            "committed_frames": manifest.committed_frames,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()

    sess = scene_bridge.Session()
    instrumentation = _Instrumentation()
    original_read_frame = cache.read_frame
    original_read_for_playback = getattr(cache, "read_frame_for_playback", None)
    original_metadata = cache.read_sequence_metadata
    original_manifest_loader = cache_schema.load_manifest
    original_signature = cache_identity.signature_bundle
    original_publication = output_publication.update_outputs_from_cache_frame
    playback_read_api = (
        "read_frame_for_playback"
        if callable(original_read_for_playback)
        else "read_frame"
    )
    signature_validation_mode = (
        "static_generation"
        if playback_read_api == "read_frame_for_playback"
        and hasattr(playback, "CacheValidationCursor")
        else "per_frame"
    )

    def timed_read_frame(*call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            return original_read_frame(*call_args, **call_kwargs)
        finally:
            instrumentation.record_timing("read_frame_seconds", time.perf_counter() - started)
            instrumentation.count("read_frame")

    def timed_metadata(*call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            return original_metadata(*call_args, **call_kwargs)
        finally:
            instrumentation.record_timing("metadata_load_seconds", time.perf_counter() - started)
            instrumentation.count("manifest_metadata_load")

    def timed_manifest_loader(*call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            return original_manifest_loader(*call_args, **call_kwargs)
        finally:
            if instrumentation.inside_playback_read:
                instrumentation.record_timing(
                    "metadata_load_seconds", time.perf_counter() - started
                )
                instrumentation.count("manifest_metadata_load")

    def timed_read_for_playback(*call_args, **call_kwargs):
        started = time.perf_counter()
        instrumentation.inside_playback_read = True
        try:
            return original_read_for_playback(*call_args, **call_kwargs)
        finally:
            instrumentation.inside_playback_read = False
            instrumentation.record_timing(
                "read_frame_seconds", time.perf_counter() - started
            )
            instrumentation.count("read_frame")

    def timed_signature(*call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            return original_signature(*call_args, **call_kwargs)
        finally:
            instrumentation.record_timing("signature_bundle_seconds", time.perf_counter() - started)
            instrumentation.count("signature_bundle")

    def timed_publication(publication_scene, frame_data):
        if instrumentation.current is not None:
            instrumentation.current.update({
                "published_payload_sha256": payload_sha256(frame_data),
                "published_particle_count": len(frame_data.get("positions", ())),
                "published_whitewater_count": len(frame_data.get("whitewater", ())),
                "published_surface_vertex_count": len(frame_data.get("surface_verts", ())),
                "published_surface_triangle_count": len(frame_data.get("surface_tris", ())),
            })
        started = time.perf_counter()
        try:
            return original_publication(publication_scene, frame_data)
        finally:
            instrumentation.record_timing("publication_seconds", time.perf_counter() - started)
            instrumentation.count("output_publication")

    if playback_read_api == "read_frame_for_playback":
        cache.read_frame_for_playback = timed_read_for_playback
        cache_schema.load_manifest = timed_manifest_loader
    else:
        cache.read_frame = timed_read_frame
        cache.read_sequence_metadata = timed_metadata
    cache_identity.signature_bundle = timed_signature
    output_publication.update_outputs_from_cache_frame = timed_publication
    samples: list[dict[str, object]] = []
    contention_observations: list[dict[str, object]] = []
    try:
        instrumentation.phase = "warmup"
        for frame in range(1, validation.WARMUP_FRAMES + 1):
            playback.play_cache_frame(sess, scene, frame)
            instrumentation.count("play_cache_frame")
        instrumentation.phase = "idle"
        contention_observations.append(_capture_contention_sample("before"))
        for frame in range(validation.WARMUP_FRAMES + 1, validation.MIN_CACHE_FRAMES + 1):
            sample: dict[str, object] = {
                "frame": frame,
                "read_frame_seconds": 0.0,
                "metadata_load_seconds": 0.0,
                "signature_bundle_seconds": 0.0,
                "publication_seconds": 0.0,
            }
            instrumentation.current = sample
            started = time.perf_counter()
            playback.play_cache_frame(sess, scene, frame)
            sample["total_seconds"] = time.perf_counter() - started
            instrumentation.count("play_cache_frame")
            sample["status"] = str(scene.fluidsim.status)
            instrumentation.current = None
            samples.append(sample)
        contention_observations.append(_capture_contention_sample("after"))
    finally:
        instrumentation.current = None
        instrumentation.phase = "idle"
        if playback_read_api == "read_frame_for_playback":
            cache.read_frame_for_playback = original_read_for_playback
            cache_schema.load_manifest = original_manifest_loader
        else:
            cache.read_frame = original_read_frame
            cache.read_sequence_metadata = original_metadata
        cache_identity.signature_bundle = original_signature
        output_publication.update_outputs_from_cache_frame = original_publication

    signatures_after = original_signature(scene, bpy.context.evaluated_depsgraph_get())
    manifest_digest_after = _sha256_file(manifest_path)
    tree_digest_after = _tree_sha256(args.cache_dir)
    totals = [float(sample["total_seconds"]) for sample in samples]
    report = {
        "schema": validation.WORKER_SCHEMA,
        "result": validation.WORKER_RESULT,
        "measurement_kind": validation.MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "raw observation only; no standalone optimization approval",
        "code_identity": identity,
        "environment": {
            "blender_version": bpy.app.version_string,
            "blender_executable": str(Path(bpy.app.binary_path).resolve()),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "addon_module_path": str(Path(addon.__file__).resolve()),
            "playback_module_path": str(Path(playback.__file__).resolve()),
            "cache_module_path": str(Path(cache.__file__).resolve()),
            "process_id": os.getpid(),
            "fresh_process_import": True,
            "background": True,
        },
        "configuration": {
            "cache_frames": validation.MIN_CACHE_FRAMES,
            "warmup_frames": validation.WARMUP_FRAMES,
            "measured_frames": validation.MEASURED_FRAMES,
            "warmup_sequence": list(range(1, validation.WARMUP_FRAMES + 1)),
            "measured_sequence": list(range(validation.WARMUP_FRAMES + 1, validation.MIN_CACHE_FRAMES + 1)),
            "distinct_measured_frames": True,
            "cache_construction_outside_timed_region": True,
            "timing_scope": "play_cache_frame",
            "playback_read_api": playback_read_api,
            "signature_validation_mode": signature_validation_mode,
        },
        "fixture": {
            "mesh_name": source.name,
            "role": source.fluidsim.role,
            "static": True,
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
            "matrix_world_sha256": _matrix_sha256(source.matrix_world),
            "signature_before": signatures_before,
            "signature_after": signatures_after,
        },
        "cache_provenance": {
            "cache_path": str(args.cache_dir.resolve()),
            "destination_initially_empty": True,
            "outside_addon_root": not _is_relative_to(args.cache_dir.resolve(), root),
            "format_version": manifest.format_version,
            "frame_start": manifest.frame_start,
            "frame_end": manifest.frame_end,
            "committed_frame_count": len(manifest.committed_frames),
            "manifest_sha256_before": manifest_digest_before,
            "manifest_sha256_after": manifest_digest_after,
            "cache_tree_sha256_before": tree_digest_before,
            "cache_tree_sha256_after": tree_digest_after,
            "immutable_after_build": all(
                not bool(path.stat().st_mode & stat.S_IWRITE)
                for path in args.cache_dir.rglob("*") if path.is_file()
            ),
            "metadata_identity": metadata_identity,
        },
        "publication_counts": instrumentation.counts,
        "warmup_counts": instrumentation.warmup_counts,
        "contention": _contention_record(contention_observations),
        "samples": samples,
        "sample_median_seconds": statistics.median(totals),
        "sample_p95_seconds": validation._nearest_rank_p95(totals),
        "error_count": 0,
        "errors": [],
    }
    errors = validation.validate_worker_report(report)
    if errors:
        raise RuntimeError("invalid cache-playback evidence: " + "; ".join(errors))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser


def _script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(_script_args() if argv is None else argv)
    try:
        args.addon_root = args.addon_root.resolve()
        args.cache_dir = args.cache_dir.resolve()
        validate_cache_destination(args.addon_root, args.cache_dir)
        identity = resolve_code_identity(args.addon_root)
        report = _run_benchmark(args, identity)
        _atomic_json(args.json_out, report)
    except Exception as exc:  # noqa: BLE001 - fail-closed benchmark boundary
        print(f"CACHE-PLAYBACK-BENCHMARK-FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"{validation.WORKER_RESULT} "
        f"median_ms={report['sample_median_seconds'] * 1000.0:.6f} "
        f"json={args.json_out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
