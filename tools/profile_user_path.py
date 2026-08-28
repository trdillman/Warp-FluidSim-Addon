"""Profile and validate the interactive Blender user frame path.

This protects one narrow path: repo add-on load, Quick Setup, Initialize
Solver, then the same per-frame scene-sync, solver-step, viewport-output update,
and optional cache-write calls used by live viewport playback. It is intended to
catch regressions from solver, scene-bridge, viewport, and cache changes that
still pass bpy-free solver profiling.

Times each segment separately so display and cache overhead can be compared
against pure solver cost:

- ``sync``: ``sync_scene_to_solver`` (sources + collider gather)
- ``solve``: ``FluidSolver.step_frame``
- ``view``: ``output_publication.update_outputs`` (readback + datablock update)
- ``cache``: ``io.cache.write_frame`` (only when ``--cache``)

Run:
    blender --background --python tools/profile_user_path.py -- \
        [--frames N] [--voxel-size V] [--particles] [--cache] \
        [--collider-subdiv K] [--json-out PATH] [--no-validate]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np

USER_PATH_CONTRACT = {
    "name": "interactive_viewport_user_path",
    "protected_path": (
        "repo add-on load -> fluidsim.setup_scene -> fluidsim.initialize -> "
        "sync_scene_to_solver -> FluidSolver.step_frame -> "
        "output_publication.update_outputs -> optional cache.write_frame"
    ),
    "parameter_surface": [
        "--frames",
        "--voxel-size",
        "--max-particles",
        "--particles",
        "--cache",
        "--collider-subdiv",
        "--surface-attributes",
        "--surface-attribution",
        "--max-mean-view-ms",
        "--max-mean-total-ms",
        "--min-particles",
        "--min-voxels",
    ],
    "inspection_surface": [
        "result",
        "checks",
        "means",
        "frames",
        "viewport_counts",
        "scene_stats_probe",
        "particles_mean",
        "voxels_mean",
        "surface_verts_last",
        "surface_dense_dim_last",
        "reset_probe",
        "phase2_costs",
        "host_output_operations",
        "output_entrypoint_counts",
        "surface_attribution",
        "evidence",
    ],
    "regression_focus": [
        "CUDA session remains active",
        "solver frame and scene stats advance across playback frames",
        "viewport datablocks receive non-empty changing output",
        "collider gather, viewport readback, and optional cache writes stay bounded",
    ],
}


DEFAULT_SURFACE_ATTRIBUTE_KEYS = ("velocity", "motion")
SURFACE_ATTRIBUTE_READBACKS_PER_PUBLICATION = {
    "off": 0,
    "default": 1,
    "all": 9,
}
SURFACE_ATTRIBUTE_NUMPY_CALLS_PER_PUBLICATION = {
    "off": 0,
    "default": 1,
    "all": 10,
}


class _SurfaceTimingProbe:
    """Host-observed surface attribution without adding GPU synchronization.

    Timed stages may enqueue asynchronous GPU work.  Device-array ``numpy``
    calls are therefore measured separately as the existing synchronization /
    readback boundary.  The report deliberately calls all remaining time a
    handler remainder instead of assigning asynchronous spillover to a phase.
    """

    _STAGES = (
        "surface_build",
        "surface_extract",
        "surface_attribute_adaptation",
        "output_publication",
    )

    def __init__(self, *, clock=time.perf_counter):
        self._clock = clock
        self._frame = None
        self._stack = []
        self.rows = []

    @property
    def active(self):
        return self._frame is not None

    def begin_frame(self, *, frame):
        if self.active:
            raise RuntimeError("surface attribution frame is already active")
        self._frame = {
            "frame": int(frame),
            "stage_host_ms": {stage: 0.0 for stage in self._STAGES},
            "stage_calls": {stage: 0 for stage in self._STAGES},
            "readback_ms": {stage: 0.0 for stage in self._STAGES},
            "readback_calls": {stage: 0 for stage in self._STAGES},
            "unclassified_readback_ms": 0.0,
            "unclassified_readback_calls": 0,
        }
        self._stack = []

    @contextmanager
    def stage(self, name):
        if not self.active or name not in self._STAGES:
            yield
            return
        node = {"name": name, "start": self._clock(), "child_s": 0.0}
        self._stack.append(node)
        try:
            yield
        finally:
            elapsed = max(0.0, self._clock() - node["start"])
            current = self._stack.pop()
            if current is not node:
                raise RuntimeError("surface attribution stage stack is corrupt")
            exclusive = max(0.0, elapsed - node["child_s"])
            self._frame["stage_host_ms"][name] += exclusive * 1000.0
            self._frame["stage_calls"][name] += 1
            if self._stack:
                self._stack[-1]["child_s"] += elapsed

    @contextmanager
    def device_readback(self):
        if not self.active:
            yield
            return
        parent = self._stack[-1]["name"] if self._stack else None
        start = self._clock()
        try:
            yield
        finally:
            elapsed = max(0.0, self._clock() - start)
            if parent in self._STAGES:
                self._frame["readback_ms"][parent] += elapsed * 1000.0
                self._frame["readback_calls"][parent] += 1
            else:
                self._frame["unclassified_readback_ms"] += elapsed * 1000.0
                self._frame["unclassified_readback_calls"] += 1
            if self._stack:
                self._stack[-1]["child_s"] += elapsed

    def end_frame(self, *, handler_total_ms):
        if not self.active:
            raise RuntimeError("surface attribution frame is not active")
        if self._stack:
            raise RuntimeError("surface attribution frame ended inside a stage")
        frame = self._frame
        row = {"frame": frame["frame"]}
        attributed = 0.0
        for stage in self._STAGES:
            host_ms = frame["stage_host_ms"][stage]
            readback_ms = frame["readback_ms"][stage]
            row[f"{stage}_host_ms"] = host_ms
            row[f"{stage}_calls"] = frame["stage_calls"][stage]
            row[f"{stage}_readback_ms"] = readback_ms
            row[f"{stage}_readback_calls"] = frame["readback_calls"][stage]
            attributed += host_ms + readback_ms
        row["unclassified_readback_ms"] = frame["unclassified_readback_ms"]
        row["unclassified_readback_calls"] = frame[
            "unclassified_readback_calls"]
        total = max(0.0, float(handler_total_ms))
        row["handler_total_ms"] = total
        row["surface_attributed_ms"] = attributed
        row["handler_remainder_ms"] = max(0.0, total - attributed)
        row["attribution_overflow_ms"] = max(0.0, attributed - total)
        self.rows.append(row)
        self._frame = None
        return row


def _surface_evidence_binding_errors(evidence):
    """Return fail-closed errors for the raw surface evidence envelope."""
    errors = []
    claim = evidence.get("claim", {})
    if claim.get("label") != "RAW-PROFILER-EVIDENCE-NOT-APPROVAL":
        errors.append("claim.label is not the raw non-approval label")
    if claim.get("production_approved") is not False:
        errors.append("claim.production_approved must remain false")
    required = {
        "source": (
            "git_head", "addon_git_tree", "addon_source_sha256",
            "addon_module_file", "profiler_sha256",
        ),
        "runtime": (
            "blender_version", "python_version", "warp_version", "device",
        ),
        "process": ("pid",),
        "contention": ("classification",),
        "workload": ("frames_requested", "surface_enabled"),
        "correctness": ("fingerprint_sha256",),
    }
    for section, names in required.items():
        values = evidence.get(section, {})
        for name in names:
            if values.get(name) in (None, ""):
                errors.append(f"{section}.{name} is missing")
    return errors


def _install_surface_attribution(probe, solver, output_publication,
                                 host_frame_type):
    """Wrap existing host boundaries and return an exact restoration hook."""
    patches = []

    def patch(owner, name, stage, predicate=None):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            if predicate is not None and not predicate(args, kwargs):
                return original(*args, **kwargs)
            with probe.stage(stage):
                return original(*args, **kwargs)

        setattr(owner, name, measured)
        patches.append((owner, name, original))

    surface = solver.surface
    patch(type(surface), "build", "surface_build")
    patch(type(surface.render_extractor), "extract", "surface_extract")

    def is_surface_attributes(args, kwargs):
        output_kind = kwargs.get("output_kind")
        if output_kind is None and len(args) >= 3:
            output_kind = args[2]
        return str(output_kind).lower() == "surface"

    patch(
        host_frame_type, "output_attributes",
        "surface_attribute_adaptation", is_surface_attributes,
    )
    # Selected viewport coloring can request a channel outside the persistent
    # output inventory, so it is a second real surface-adaptation boundary.
    patch(
        output_publication, "_surface_values",
        "surface_attribute_adaptation",
    )
    patch(output_publication, "publish_payload", "output_publication")

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)

    return restore


def _source_tree_digest(root):
    """Hash actual source-tree bytes, excluding interpreter cache products."""
    root = Path(root).resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        total_bytes += len(data)
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "byte_count": total_bytes,
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(repo_root, *args):
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True,
            timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_identity(repo_root, addon_module_file):
    addon_root = Path(repo_root, "fluidSim_warp_blender51")
    digest = _source_tree_digest(addon_root)
    profiler_path = Path(__file__).resolve()
    return {
        "repo_root": str(Path(repo_root).resolve()),
        "git_head": _git_text(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_text(
            repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": _git_text(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all"),
        "addon_git_tree": _git_text(
            repo_root, "rev-parse", "HEAD:fluidSim_warp_blender51"),
        "addon_source_sha256": digest["sha256"],
        "addon_source_file_count": digest["file_count"],
        "addon_source_byte_count": digest["byte_count"],
        "addon_module_file": str(Path(addon_module_file).resolve()),
        "profiler_file": str(profiler_path),
        "profiler_sha256": _file_sha256(profiler_path),
    }


def _runtime_identity(bpy, wp, device):
    try:
        runtime_device = wp.get_device(device)
    except Exception:  # noqa: BLE001 - evidence must retain partial identity
        runtime_device = None

    def device_value(name, default=""):
        value = getattr(runtime_device, name, default)
        return value() if callable(value) else value

    return {
        "blender_version": str(getattr(bpy.app, "version_string", "")),
        "blender_binary": str(getattr(bpy.app, "binary_path", "")),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "warp_version": str(getattr(wp, "__version__", "")),
        "device": str(device),
        "device_alias": str(device_value("alias")),
        "device_name": str(device_value("name")),
        "device_arch": str(device_value("arch")),
        "device_is_cuda": bool(device_value("is_cuda", False)),
    }


def _nvidia_smi_snapshot():
    """Capture compute-process contention outside the measured frame loop."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        gpu = subprocess.run(
            [
                executable,
                (
                    "--query-gpu=index,name,uuid,driver_version,memory.total,"
                    "memory.used,utilization.gpu"
                ),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
        processes = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    if gpu.returncode != 0 or processes.returncode != 0:
        error = (gpu.stderr.strip() or processes.stderr.strip()
                 or "nvidia-smi query failed")
        return {"available": False, "error": error}

    compute = []
    for line in processes.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            used_memory = int(fields[2])
        except ValueError:
            continue
        compute.append({
            "pid": pid,
            "process_name": os.path.basename(fields[1]),
            "used_memory_mb": used_memory,
        })
    return {
        "available": True,
        "gpu_rows": [line.strip() for line in gpu.stdout.splitlines()
                     if line.strip()],
        "compute_processes": compute,
    }


def _contention_evidence(before, after, own_pid):
    if not before.get("available") or not after.get("available"):
        classification = "unavailable"
        other = []
    else:
        by_pid = {}
        for snapshot in (before, after):
            for process in snapshot.get("compute_processes", ()):
                if int(process["pid"]) != int(own_pid):
                    by_pid[int(process["pid"])] = process
        other = [by_pid[pid] for pid in sorted(by_pid)]
        classification = (
            "other_compute_processes_observed" if other
            else "no_other_compute_processes_observed"
        )
    return {
        "classification": classification,
        "scope": "nvidia-smi compute-process snapshots before and after timing",
        "own_pid": int(own_pid),
        "other_compute_processes": other,
        "before": before,
        "after": after,
    }


def _update_array_digest(digest, name, values):
    values = np.ascontiguousarray(values)
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(values.tobytes())


def _output_correctness_evidence(bpy, rows, output_counts):
    """Fingerprint observed outputs after timing, outside all timed regions."""
    digest = hashlib.sha256()
    facts = {
        "frame_facts": [
            {
                key: row[key]
                for key in (
                    "frame", "solver_frame", "particles", "voxels",
                    "whitewater", "surface_verts", "particle_points",
                    "surface_dense_dim",
                )
            }
            for row in rows
        ],
        "viewport_counts": output_counts,
    }
    digest.update(json.dumps(
        facts, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    surface = bpy.data.objects.get("FluidSim Surface")
    geometry = {
        "available": bool(surface is not None and surface.type == "MESH"),
        "vertices": 0,
        "loops": 0,
        "polygons": 0,
    }
    if geometry["available"]:
        mesh = surface.data
        geometry.update({
            "vertices": len(mesh.vertices),
            "loops": len(mesh.loops),
            "polygons": len(mesh.polygons),
        })
        coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        vertex_indices = np.empty(len(mesh.loops), dtype=np.int32)
        loop_starts = np.empty(len(mesh.polygons), dtype=np.int32)
        loop_totals = np.empty(len(mesh.polygons), dtype=np.int32)
        if len(coordinates):
            mesh.vertices.foreach_get("co", coordinates)
        if len(vertex_indices):
            mesh.loops.foreach_get("vertex_index", vertex_indices)
        if len(loop_starts):
            mesh.polygons.foreach_get("loop_start", loop_starts)
            mesh.polygons.foreach_get("loop_total", loop_totals)
        _update_array_digest(digest, "surface_coordinates", coordinates)
        _update_array_digest(digest, "surface_vertex_indices", vertex_indices)
        _update_array_digest(digest, "surface_loop_starts", loop_starts)
        _update_array_digest(digest, "surface_loop_totals", loop_totals)
    return {
        "fingerprint_sha256": digest.hexdigest(),
        "semantics": (
            "diagnostic workload/output fingerprint; not bitwise simulation "
            "approval or cross-run determinism"
        ),
        "surface_geometry": geometry,
        **facts,
    }


def _surface_attribution_summary(rows, synchronization_calls):
    if not rows:
        return {
            "enabled": False,
            "measurement_semantics": "not_requested",
            "frames": [],
        }
    warm = _warm_rows(rows)
    numeric = (
        "surface_build_host_ms",
        "surface_build_readback_ms",
        "surface_extract_host_ms",
        "surface_extract_readback_ms",
        "surface_attribute_adaptation_host_ms",
        "surface_attribute_adaptation_readback_ms",
        "output_publication_host_ms",
        "output_publication_readback_ms",
        "unclassified_readback_ms",
        "surface_attributed_ms",
        "handler_remainder_ms",
        "attribution_overflow_ms",
    )
    return {
        "enabled": True,
        "measurement_semantics": "HOST-OBSERVED-NO-ADDED-GPU-SYNCHRONIZATION",
        "raw_non_approval": True,
        "adds_gpu_synchronization": False,
        "observed_synchronization_calls": int(synchronization_calls),
        "definitions": {
            "host_ms": (
                "exclusive host wall time at the existing call boundary; "
                "asynchronous GPU work may complete at a later readback"
            ),
            "readback_ms": (
                "wall time of existing Warp array.numpy calls charged to the "
                "enclosing observed stage"
            ),
            "handler_remainder_ms": (
                "complete sync+solve+view frame time not safely attributable "
                "to the observed surface boundaries"
            ),
            "output_publication": (
                "complete persistent output publication call for the "
                "surface-mode frame, including any hidden-output maintenance"
            ),
        },
        "warm_means": {
            name: statistics.fmean(row[name] for row in warm)
            for name in numeric
        },
        "frames": rows,
    }


def configure_surface_attribute_profile(settings, mode):
    from fluidSim_warp_blender51.sim.attributes import (
        OUTPUT_ATTRIBUTE_KEYS,
        OUTPUT_TOGGLE_NAMES,
        enabled_attribute_keys,
    )

    mode = str(mode).lower()
    if mode == "off":
        enabled = frozenset()
    elif mode == "default":
        enabled = frozenset(DEFAULT_SURFACE_ATTRIBUTE_KEYS)
    elif mode == "all":
        enabled = frozenset(OUTPUT_ATTRIBUTE_KEYS["surface"])
    else:
        raise ValueError(f"unknown surface attribute profile {mode!r}")
    for key, property_name in OUTPUT_TOGGLE_NAMES["surface"].items():
        setattr(settings, property_name, key in enabled)
    return enabled_attribute_keys(settings, "surface")


def expected_surface_attribute_readbacks(mode):
    try:
        return SURFACE_ATTRIBUTE_READBACKS_PER_PUBLICATION[str(mode).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown surface attribute profile {mode!r}") from exc


def expected_surface_attribute_numpy_calls(mode):
    try:
        return SURFACE_ATTRIBUTE_NUMPY_CALLS_PER_PUBLICATION[
            str(mode).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown surface attribute profile {mode!r}") from exc


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--max-particles", type=int, default=2_000_000)
    ap.add_argument("--cfl", type=float, default=0.0,
                    help="override the scene CFL value for comparison runs")
    ap.add_argument("--particles", action="store_true",
                    help="show particle point cloud instead of surface mesh")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--collider-subdiv", type=int, default=5,
                    help="icosphere subdivision for a static high-poly collider")
    ap.add_argument(
        "--surface-attributes", choices=("off", "default", "all"),
        default="default",
        help="surface attribute inventory used by the viewport profile",
    )
    ap.add_argument(
        "--surface-attribution", action="store_true",
        help=(
            "attribute host-observed surface build/extract/readback/attribute/"
            "publication time without inserting GPU synchronization"
        ),
    )
    ap.add_argument("--disable-whitewater", action="store_true",
                    help="keep whitewater disabled to match the default Quick Setup path")
    ap.add_argument("--profile-phases", action="store_true",
                    help="record solver phase timers in per-frame output")
    ap.add_argument("--profile-sync", action="store_true",
                    help="synchronize CUDA around each phase timer")
    ap.add_argument("--domain-scale-x", type=float, default=0.0,
                    help="override Quick Setup domain X scale after scene setup")
    ap.add_argument("--domain-location-x", type=float, default=0.0,
                    help="override Quick Setup domain X location after scene setup")
    ap.add_argument("--max-mean-view-ms", type=float, default=80.0,
                    help="validation ceiling for warm-frame viewport update mean")
    ap.add_argument("--max-mean-total-ms", type=float, default=250.0,
                    help="validation ceiling for warm-frame sync+solve+view(+cache) mean")
    ap.add_argument("--min-particles", type=int, default=10_000,
                    help="minimum mean active particles required by validation")
    ap.add_argument("--min-voxels", type=int, default=1_000,
                    help="minimum mean active voxels required by validation")
    ap.add_argument("--no-validate", action="store_true",
                    help="write timing data without failing on regression checks")
    ap.add_argument("--frame-csv-out", default="",
                    help="stream per-frame timing rows while profiling")
    ap.add_argument("--stop-after-total-ms", type=float, default=0.0,
                    help="stop once a frame exceeds this total time")
    ap.add_argument("--json-out", default="")
    return ap.parse_args(argv)


FRAME_CSV_FIELDS = [
    "frame",
    "sync_ms",
    "solve_ms",
    "view_ms",
    "cache_ms",
    "total_ms",
    "particles",
    "voxels",
    "whitewater",
    "substeps",
    "pressure_iterations",
    "pressure_residual",
    "surface_verts",
    "particle_points",
    "surface_dense_dim",
    "timer_solver",
    "timer_compute_0",
    "timer_compute_1",
    "timer_compute_2",
    "timer_compute_3",
    "timer_compute_4",
    "timer_compute_5",
    "timer_compute_6",
    "timer_compute_7",
    "timer_compute_8",
    "timer_compute_9",
    "timer_compute_10",
    "timer_compute_11",
    "timer_compute_12",
    "timer_compute_13",
    "timer_compute_14",
    "timer_compute_15",
]


def _open_frame_csv(path):
    if not path:
        return None, None
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fh = open(path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FRAME_CSV_FIELDS)
    writer.writeheader()
    fh.flush()
    return fh, writer


def _viewport_counts(bpy, particles_mode):
    surf = bpy.data.objects.get("FluidSim Surface")
    parts = bpy.data.objects.get("FluidSim Particles")
    surface_verts = len(surf.data.vertices) if surf and surf.type == "MESH" else 0
    # Output clouds carry spare capacity, so the attribute length is not the
    # count -- profiling numbers would read as inflated by the padding.
    from fluidSim_warp_blender51.blender import output_publication

    particle_points = output_publication.live_point_count(parts)
    primary_count = particle_points if particles_mode else surface_verts
    return {
        "surface_verts": int(surface_verts),
        "particle_points": int(particle_points),
        "primary_count": int(primary_count),
    }


def _check(checks, name, condition, detail=""):
    ok = bool(condition)
    checks.append({"name": name, "ok": ok, "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def _surface_dense_dim(sess):
    extractor = getattr(sess.solver, "surface_extractor", None)
    dims = getattr(extractor, "_dense_dim", None)
    if dims is None:
        return [0, 0, 0]
    return [int(d) for d in dims]


def _expected_surface_publications(frames, cadence):
    """Count frames where the live cadence contract requires a surface."""
    cadence = max(1, int(cadence))
    return sum(
        1 for row in frames
        if int(row["solver_frame"]) <= 1
        or cadence <= 1
        or int(row["solver_frame"]) % cadence == 0
    )


def _warm_rows(rows):
    """Exclude the first lazy-compilation frames when enough samples exist."""
    return rows[3:] if len(rows) > 3 else rows


def _validate_contract(args, summary, sess, device, output_counts):
    checks = []
    frames = summary["frames"]
    means = summary["means"]
    primary_counts = [c["primary_count"] for c in output_counts]
    particle_points = [c["particle_points"] for c in output_counts]
    scene_probe = summary["scene_stats_probe"]

    _check(checks, "contract path", summary["contract"]["name"] == USER_PATH_CONTRACT["name"])
    _check(checks, "cuda session", sess.active and str(device).startswith("cuda"), device)
    _check(checks, "frames measured", len(frames) == args.frames, len(frames))
    _check(checks, "solver advanced", scene_probe["solver_frame"] >= args.frames,
           scene_probe["solver_frame"])
    _check(checks, "particles active", summary["particles_mean"] >= args.min_particles,
           f"{summary['particles_mean']:.0f} >= {args.min_particles}")
    _check(checks, "voxels active", summary["voxels_mean"] >= args.min_voxels,
           f"{summary['voxels_mean']:.0f} >= {args.min_voxels}")
    _check(checks, "viewport primary output nonempty", max(primary_counts or [0]) > 0,
           max(primary_counts or [0]))
    _check(checks, "viewport output changes", len(set(primary_counts)) > 1,
           f"{primary_counts[0] if primary_counts else 0} -> {primary_counts[-1] if primary_counts else 0}")
    if args.particles:
        _check(checks, "particle output available", max(particle_points or [0]) > 0,
               max(particle_points or [0]))
    _check(checks, "scene stats updated",
           scene_probe["particles"] == scene_probe["step_particles"] and
           scene_probe["voxels"] == scene_probe["step_voxels"] and
           scene_probe["particles"] > 0 and scene_probe["voxels"] > 0,
           f"{scene_probe['particles']} particles, {scene_probe['voxels']} voxels")
    _check(checks, "scene stats history recorded",
           scene_probe["frame_stats_len"] > 0 and scene_probe["status"].startswith("Frame "),
           f"{scene_probe['frame_stats_len']} frames, status={scene_probe['status']!r}")
    _check(checks, "viewport mean bounded", means["view_ms"] <= args.max_mean_view_ms,
           f"{means['view_ms']:.3f} <= {args.max_mean_view_ms:.3f} ms")
    _check(checks, "total mean bounded", means["total_ms"] <= args.max_mean_total_ms,
           f"{means['total_ms']:.3f} <= {args.max_mean_total_ms:.3f} ms")
    if args.cache:
        _check(checks, "cache writes timed", means["cache_ms"] > 0.0,
               f"{means['cache_ms']:.3f} ms")

    measured = len(frames)
    entrypoints = summary["output_entrypoint_counts"]
    _check(checks, "no output-path Warp synchronization",
           entrypoints["warp_synchronizations"] == 0,
           entrypoints["warp_synchronizations"])
    binding_errors = _surface_evidence_binding_errors(summary["evidence"])
    _check(checks, "surface evidence identity complete",
           not binding_errors, "; ".join(binding_errors))
    expected_surface = _expected_surface_publications(
        frames, summary["config"]["live_surface_publish_cadence"])
    if not args.particles:
        attribute_readbacks = expected_surface_attribute_readbacks(
            args.surface_attributes)
        attribute_numpy_calls = expected_surface_attribute_numpy_calls(
            args.surface_attributes)
        expected_numpy = expected_surface * (5 + attribute_numpy_calls)
        _check(checks, "surface readback entrypoints cadence-bounded",
               entrypoints["warp_array_numpy"] == expected_numpy,
               f"{entrypoints['warp_array_numpy']} == {expected_numpy}")
    else:
        attribute_readbacks = 0

    costs = summary["phase2_costs"]
    live = costs["live_adaptation"]
    publication = costs["output_publication"]
    _check(checks, "one host-frame acquisition per live publication",
           live["host_frame_acquisitions"] == measured,
           f"{live['host_frame_acquisitions']} == {measured}")
    _check(checks, "one live adaptation per measured frame",
           live["invocations"] == measured,
           f"{live['invocations']} == {measured}")
    _check(checks, "one output publication per measured frame",
           publication["invocations"] == measured,
           f"{publication['invocations']} == {measured}")
    _check(checks, "one surface extraction per scheduled surface frame",
           live["surface_extractions"] == expected_surface,
           f"{live['surface_extractions']} == {expected_surface}")
    expected_device_readbacks = expected_surface * attribute_readbacks
    host_operations = summary["host_output_operations"]
    _check(checks, "surface attribute device readbacks exactly bounded",
           live["device_readbacks"] == expected_device_readbacks,
           f"{live['device_readbacks']} == {expected_device_readbacks}")
    _check(checks, "HostFrame reports the same bounded device readbacks",
           host_operations["device_readbacks"] == expected_device_readbacks,
           f"{host_operations['device_readbacks']} == "
           f"{expected_device_readbacks}")
    _check(checks, "HostFrame surface extractions match publication cadence",
           host_operations["surface_extractions"] == expected_surface,
           f"{host_operations['surface_extractions']} == {expected_surface}")
    for event in (
            "synchronizations",
            "blender_id_creations", "blender_id_removals",
            "depsgraph_updates", "relation_changes", "structural_replacements",
            "datablock_replacements"):
        added = live[event] + publication[event]
        _check(checks, f"no added output-path {event.replace('_', ' ')}",
               added == 0, added)

    if args.surface_attribution:
        attribution = summary["surface_attribution"]
        attribution_rows = attribution["frames"]
        _check(checks, "surface attribution measured every handler frame",
               len(attribution_rows) == measured,
               f"{len(attribution_rows)} == {measured}")
        _check(checks, "surface attribution adds no GPU synchronization",
               attribution["adds_gpu_synchronization"] is False,
               attribution["measurement_semantics"])
        _check(checks, "surface extraction attribution matches cadence",
               sum(row["surface_extract_calls"]
                   for row in attribution_rows) == expected_surface,
               f"{sum(row['surface_extract_calls'] for row in attribution_rows)} "
               f"== {expected_surface}")
        _check(checks, "publication attribution covers every handler frame",
               sum(row["output_publication_calls"]
                   for row in attribution_rows) == measured,
               f"{sum(row['output_publication_calls'] for row in attribution_rows)} "
               f"== {measured}")
        _check(checks, "surface attribution accounting does not overflow",
               max((row["attribution_overflow_ms"]
                    for row in attribution_rows), default=0.0) < 0.01,
               max((row["attribution_overflow_ms"]
                    for row in attribution_rows), default=0.0))

    return checks


def main():
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.surface_attribution and args.particles:
        raise SystemExit(
            "--surface-attribution requires surface viewport mode; "
            "omit --particles")
    import addon_utils
    import bmesh
    import bpy

    # Measure the repo checkout, not a possibly stale installed addon copy:
    # repo root precedes Blender's addon dirs on sys.path so the package
    # import inside addon_utils.enable resolves here.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    sys.modules.pop("fluidSim_warp_blender51", None)

    addon_utils.enable("fluidSim_warp_blender51", default_set=True, persistent=True)
    import fluidSim_warp_blender51 as _fs_mod
    assert os.path.abspath(_fs_mod.__file__).startswith(repo_root), (
        f"addon loaded from {_fs_mod.__file__}, expected repo {repo_root}")
    source_identity = _source_identity(repo_root, _fs_mod.__file__)
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.voxel_size = args.voxel_size
    fs.max_particles = args.max_particles
    if args.cfl > 0.0:
        fs.cfl = args.cfl
    fs.max_whitewater = 500_000
    fs.whitewater_enabled = not args.disable_whitewater
    fs.surface_enabled = True
    fs.profiling_enabled = bool(args.profile_phases)
    fs.profiling_sync = bool(args.profile_sync)
    if args.particles:
        fs.liquid_view_mode = "PARTICLES"
    else:
        fs.liquid_view_mode = "SURFACE"
    fs.particle_display_fraction = 1.0
    fs.cache_enabled = False  # timed manually below

    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    enabled_surface_attributes = configure_surface_attribute_profile(
        fs, args.surface_attributes)
    if args.domain_scale_x > 0.0:
        domain = fs.domain_object
        domain.scale.x = args.domain_scale_x
        domain.location.x = args.domain_location_x

    # Static high-poly collider to exercise the per-frame collider gather.
    if args.collider_subdiv > 0:
        mesh = bpy.data.meshes.new("BenchCollider")
        obj = bpy.data.objects.new("BenchCollider", mesh)
        scene.collection.objects.link(obj)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(
            bm, subdivisions=args.collider_subdiv, radius=0.8)
        bm.to_mesh(mesh)
        bm.free()
        obj.location = (0.0, 0.0, 0.8)
        obj.fluidsim.role = "COLLIDER"
        obj.fluidsim.enabled = True

    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status

    import warp as wp

    from fluidSim_warp_blender51.blender import output_publication, scene_bridge
    from fluidSim_warp_blender51.io import cache as cache_io

    sess = scene_bridge.session()
    assert sess.active
    runtime_identity = _runtime_identity(bpy, wp, sess.solver.device)
    process_identity = {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "executable": sys.executable,
        "argv": list(sys.argv),
    }
    domain = fs.domain_object
    domain_bounds = None
    if domain is not None:
        x0 = float(domain.location.x - domain.scale.x * 0.5)
        x1 = float(domain.location.x + domain.scale.x * 0.5)
        domain_bounds = {
            "location": [float(v) for v in domain.location],
            "scale": [float(v) for v in domain.scale],
            "x_bounds": [x0, x1],
        }
    if args.cache:
        fs.cache_enabled = True
        fs.cache_path = "//_bench_user_cache/"

    # Measure only the repeated frame path. Initialization/preparation has
    # already completed, matching the Phase 1 baseline profile boundary.
    output_publication.output_cost_counts(reset=True)
    from fluidSim_warp_blender51.sim.frame_output import (
        host_output_operation_counts,
    )
    host_output_operation_counts(reset=True)

    contention_before = _nvidia_smi_snapshot()
    surface_probe = (
        _SurfaceTimingProbe() if args.surface_attribution else None)
    restore_surface_attribution = None
    if surface_probe is not None:
        from fluidSim_warp_blender51.sim.frame_output import HostFrame
        restore_surface_attribution = _install_surface_attribution(
            surface_probe, sess.solver, output_publication, HostFrame)

    rows = []
    output_counts = []
    output_entrypoint_counts = {
        "warp_array_numpy": 0,
        "warp_synchronizations": 0,
        "surface_attribution_synchronizations": 0,
    }
    in_output_path = False
    original_numpy = wp.array.numpy
    original_synchronize = wp.synchronize
    original_synchronize_device = wp.synchronize_device

    def counted_numpy(array, *args, **kwargs):
        if in_output_path:
            output_entrypoint_counts["warp_array_numpy"] += 1
        if surface_probe is not None and surface_probe.active:
            with surface_probe.device_readback():
                return original_numpy(array, *args, **kwargs)
        return original_numpy(array, *args, **kwargs)

    def counted_synchronize(*args, **kwargs):
        if in_output_path:
            output_entrypoint_counts["warp_synchronizations"] += 1
        if surface_probe is not None and surface_probe.active:
            output_entrypoint_counts[
                "surface_attribution_synchronizations"] += 1
        return original_synchronize(*args, **kwargs)

    def counted_synchronize_device(*args, **kwargs):
        if in_output_path:
            output_entrypoint_counts["warp_synchronizations"] += 1
        if surface_probe is not None and surface_probe.active:
            output_entrypoint_counts[
                "surface_attribution_synchronizations"] += 1
        return original_synchronize_device(*args, **kwargs)

    wp.array.numpy = counted_numpy
    wp.synchronize = counted_synchronize
    wp.synchronize_device = counted_synchronize_device
    solve_profiler = None
    if os.environ.get("FLUIDSIM_SOLVE_CPROFILE"):
        import cProfile
        solve_profiler = cProfile.Profile()
    frame_csv_fh, frame_csv_writer = _open_frame_csv(args.frame_csv_out)
    try:
        for f in range(args.frames):
            scene.frame_set(f + 1)
            if surface_probe is not None:
                surface_probe.begin_frame(frame=f)
            t0 = time.perf_counter()
            dg = bpy.context.evaluated_depsgraph_get()
            scene_bridge.sync_scene_to_solver(scene, sess, dg)
            t1 = time.perf_counter()
            if solve_profiler is not None:
                solve_profiler.enable()
            stats = sess.solver.step_frame()
            if solve_profiler is not None:
                solve_profiler.disable()
            t2 = time.perf_counter()
            in_output_path = True
            try:
                output_publication.update_outputs(scene, sess)
            finally:
                in_output_path = False
            t3 = time.perf_counter()
            if surface_probe is not None:
                surface_probe.end_frame(
                    handler_total_ms=(t3 - t0) * 1000.0)
            counts = _viewport_counts(bpy, args.particles)
            output_counts.append(counts)
            cache_ms = 0.0
            if args.cache:
                tc = time.perf_counter()
                cache_io.write_frame(scene, sess)
                cache_ms = (time.perf_counter() - tc) * 1000.0
            row = {
                "frame": f,
                "solver_frame": int(sess.solver.frame),
                "sync_ms": (t1 - t0) * 1000.0,
                "solve_ms": (t2 - t1) * 1000.0,
                "view_ms": (t3 - t2) * 1000.0,
                "cache_ms": cache_ms,
                "total_ms": (t3 - t0) * 1000.0 + cache_ms,
                "particles": int(stats.active_particles),
                "voxels": int(stats.active_voxels),
                "whitewater": int(stats.whitewater_particles),
                "substeps": int(stats.substeps),
                "pressure_iterations": int(stats.pressure_iterations),
                "pressure_residual": float(stats.pressure_residual),
                "surface_verts": counts["surface_verts"],
                "particle_points": counts["particle_points"],
                "surface_dense_dim": _surface_dense_dim(sess),
            }
            timers = getattr(stats, "timers", {}) or {}
            row["timer_solver"] = float(timers.get("solver", 0.0))
            for i in range(16):
                row[f"timer_compute_{i}"] = float(timers.get(f"compute_{i}", 0.0))
            rows.append(row)
            if frame_csv_writer is not None:
                frame_csv_writer.writerow(row)
                frame_csv_fh.flush()
            if args.stop_after_total_ms > 0.0 and row["total_ms"] >= args.stop_after_total_ms:
                break
    finally:
        wp.array.numpy = original_numpy
        wp.synchronize = original_synchronize
        wp.synchronize_device = original_synchronize_device
        if restore_surface_attribution is not None:
            restore_surface_attribution()
        if frame_csv_fh is not None:
            frame_csv_fh.close()
    contention_after = _nvidia_smi_snapshot()
    if solve_profiler is not None:
        solve_profiler.dump_stats("artifacts/perf_goal_solver.pstats")

    warm = _warm_rows(rows)

    def mean(key):
        return statistics.fmean(r[key] for r in warm)

    phase2_costs = output_publication.output_cost_counts()
    host_output_operations = host_output_operation_counts()
    correctness_evidence = _output_correctness_evidence(
        bpy, rows, output_counts)
    surface_attribution = _surface_attribution_summary(
        surface_probe.rows if surface_probe is not None else [],
        output_entrypoint_counts["surface_attribution_synchronizations"],
    )

    # Untimed wrapper-path probe: the segmented timings above isolate sync,
    # solve, and viewport cost; this call verifies the live scene-bridge surface
    # still publishes UI/debug stats from the same playback path.
    probe_stats = scene_bridge.step_frame(scene, sess)
    if probe_stats is None:
        raise RuntimeError("scene_bridge.step_frame returned no stats")

    surf = bpy.data.objects.get("FluidSim Surface")
    n_surf_verts = len(surf.data.vertices) if surf and surf.type == "MESH" else 0
    scene_stats_probe = {
        "particles": int(fs.stat_particles),
        "voxels": int(fs.stat_voxels),
        "whitewater": int(fs.stat_whitewater),
        "step_ms": float(fs.stat_step_ms),
        "substeps": int(fs.stat_substeps),
        "status": str(fs.status),
        "frame_stats_len": len(sess.frame_stats),
        "solver_frame": int(sess.solver.frame),
        "step_particles": int(probe_stats.active_particles),
        "step_voxels": int(probe_stats.active_voxels),
        "step_whitewater": int(probe_stats.whitewater_particles),
    }
    pre_reset_dense_dim = _surface_dense_dim(sess)
    scene_bridge.reset_simulation(scene)
    reset_sess = scene_bridge.session()
    reset_probe = {
        "pre_reset_surface_dense_dim": pre_reset_dense_dim,
        "post_reset_surface_dense_dim": _surface_dense_dim(reset_sess),
        "solver_frame_after_reset": int(reset_sess.solver.frame),
    }
    object_manifest = []
    for obj in sorted(scene.objects, key=lambda item: item.name):
        object_fs = getattr(obj, "fluidsim", None)
        object_manifest.append({
            "name": str(obj.name),
            "type": str(obj.type),
            "role": str(getattr(object_fs, "role", "")),
            "enabled": bool(getattr(object_fs, "enabled", False)),
        })
    evidence = {
        "claim": {
            "label": "RAW-PROFILER-EVIDENCE-NOT-APPROVAL",
            "production_approved": False,
            "scope": (
                "single-workload host-observed attribution; comparison and "
                "optimization approval require a separate paired campaign"
            ),
        },
        "source": source_identity,
        "runtime": runtime_identity,
        "process": process_identity,
        "contention": _contention_evidence(
            contention_before, contention_after, process_identity["pid"]),
        "workload": {
            "scene_name": str(scene.name),
            "frames_requested": int(args.frames),
            "frames_measured": len(rows),
            "surface_enabled": bool(fs.surface_enabled),
            "surface_visible": bool(fs.show_surface),
            "surface_attributes": str(args.surface_attributes),
            "enabled_surface_attributes": list(enabled_surface_attributes),
            "live_surface_publish_cadence": int(
                fs.live_surface_publish_cadence),
            "voxel_size": float(args.voxel_size),
            "max_particles": int(args.max_particles),
            "collider_subdiv": int(args.collider_subdiv),
            "cache_enabled": bool(args.cache),
            "domain": domain_bounds,
            "objects": object_manifest,
        },
        "correctness": correctness_evidence,
    }
    evidence["binding_errors"] = _surface_evidence_binding_errors(evidence)
    summary = {
        "result": "USER-PATH-PROFILE-PASS",
        "contract": USER_PATH_CONTRACT,
        "config": {
            **vars(args),
            "live_surface_publish_cadence": int(
                fs.live_surface_publish_cadence),
            "enabled_surface_attributes": list(enabled_surface_attributes),
        },
        "domain": domain_bounds,
        "means": {k: mean(k) for k in
                  ("sync_ms", "solve_ms", "view_ms", "cache_ms", "total_ms")},
        "particles_mean": mean("particles"),
        "voxels_mean": mean("voxels"),
        "whitewater_mean": mean("whitewater"),
        "surface_verts_last": n_surf_verts,
        "substeps_mean": mean("substeps"),
        "pressure_iterations_mean": mean("pressure_iterations"),
        "pressure_residual_last": rows[-1]["pressure_residual"] if rows else 0.0,
        "surface_dense_dim_last": rows[-1]["surface_dense_dim"] if rows else [0, 0, 0],
        "reset_probe": reset_probe,
        "viewport_counts": output_counts,
        "scene_stats_probe": scene_stats_probe,
        "phase2_costs": phase2_costs,
        "host_output_operations": host_output_operations,
        "output_entrypoint_counts": output_entrypoint_counts,
        "surface_attribution": surface_attribution,
        "evidence": evidence,
        "frames": rows,
    }
    if not args.no_validate:
        checks = _validate_contract(args, summary, sess, sess.solver.device, output_counts)
        summary["checks"] = checks
        failed = [check for check in checks if not check["ok"]]
        if failed:
            summary["result"] = "USER-PATH-REGRESSION-FAIL"
    else:
        summary["checks"] = []
    evidence["correctness"]["contract_checks_passed"] = bool(
        summary["checks"] and all(check["ok"] for check in summary["checks"]))
    if evidence["binding_errors"]:
        summary["result"] = "USER-PATH-EVIDENCE-BINDING-FAIL"
    print("USER-PATH-PROFILE " + json.dumps(summary["means"]))
    print(f"scene: voxels={summary['voxels_mean']:.0f} "
          f"particles={summary['particles_mean']:.0f} "
          f"whitewater={summary['whitewater_mean']:.0f} "
          f"surface_verts={n_surf_verts}")
    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=1)
    if summary["result"] != "USER-PATH-PROFILE-PASS":
        print(summary["result"])
        sys.exit(1)
    print(summary["result"])
    sys.exit(0)


if __name__ == "__main__":
    main()
