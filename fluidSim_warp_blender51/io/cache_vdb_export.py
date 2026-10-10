"""Export verified cache-v3 liquid level sets without importing Blender or Warp.

New cache-v3 frames carry their checksum-bound grid corner origin. Older frames
require the exact baked settings signature before deriving an origin from the
saved scene. Never guess it from changed settings or silently use world zero.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cache, cache_schema
from .vdb_writer import write_level_set_vdb


@dataclass(frozen=True)
class VDBExportPlan:
    base_dir: Path
    manifest_sha256: str
    settings_signature: str
    frames: tuple[int, ...]
    origin: tuple[float, float, float] | None
    voxel_size: float
    background: float


class LegacyOriginRequired(ValueError):
    """Old cache-v3 frames need their matching saved scene to locate the grid."""


def _manifest_snapshot(base):
    path = cache_schema.manifest_path(base)
    if not path.is_file():
        raise ValueError("There is no baked cache to export. Bake first.")
    data = path.read_bytes()
    manifest = cache_schema.CacheManifestV3.from_dict(json.loads(data))
    return manifest, hashlib.sha256(data).hexdigest()


def _manifest(plan):
    manifest, digest = _manifest_snapshot(plan.base_dir)
    if digest != plan.manifest_sha256:
        raise ValueError("The baked cache changed after VDB export was prepared. Retry export.")
    return manifest


def _sample_origin(plan, payload):
    corner = payload.get("liquid_sdf_origin")
    if corner is not None:
        return tuple(float(v) for v in np.asarray(corner,dtype=np.float64) + 0.5 * plan.voxel_size)
    if plan.origin is None:
        raise LegacyOriginRequired(
            "This older v3 cache has no baked origin. Open its unchanged saved scene or re-bake with origin metadata before VDB export.")
    return plan.origin


def _frame(plan, number):
    _manifest(plan)
    payload = cache.read_frame(str(plan.base_dir), number)
    if payload is None:
        raise ValueError(f"Committed cache frame {number} is missing.")
    cache.validate_liquid_sdf(payload, enabled=True)
    h = float(payload["liquid_sdf_voxel_size"])
    if not math.isclose(h, plan.voxel_size, rel_tol=2e-7, abs_tol=0.0):
        raise ValueError(f"Frame {number} liquid voxel size disagrees with its baked grid.")
    coords = payload["liquid_sdf_leaf_coords"]
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"Frame {number} has duplicate liquid leaf coordinates.")
    _sample_origin(plan, payload)
    _manifest(plan)
    return payload


def plan_export(base_dir, settings=None, *, frame_start=0, frame_end=0) -> VDBExportPlan:
    """Freeze a validated contiguous export; zero bounds select the full cache.

    ``settings`` is an optional authenticated fallback for older frames without
    origin metadata. When supplied it must match the canonical baked settings.
    New frames export without a scene, Blender, Warp, or a solver.
    """
    base = Path(base_dir).resolve()
    manifest, digest = _manifest_snapshot(base)
    if not manifest.narrow_band["enabled"]:
        raise ValueError("This cache has no liquid level set. Enable Narrow-Band Particles and re-bake before VDB export.")
    frames = cache_schema.verified_contiguous_frames(base, manifest)
    if not frames:
        raise ValueError("The cache has no committed frames. Bake first.")
    first = int(frame_start) or frames[0]
    last = int(frame_end) or frames[-1]
    if first > last or first < frames[0] or last > frames[-1]:
        raise ValueError(f"Export range {first}-{last} is outside the cached range {frames[0]}-{frames[-1]}.")
    selected = tuple(f for f in frames if first <= f <= last)
    # SparseGrid stores its corner origin and voxel size as Warp float32.
    # SDF samples are cell-centred: the VDB sample origin is corner + h/2.
    h = float(np.float32(manifest.grid_spec["voxel_size"]))
    fallback = None
    if settings is not None:
        # Only a byte-identical canonical settings record can authenticate the
        # unstored origin of old frames. New frames do not consult live state.
        if cache.settings_record(settings)["signature"] != manifest.settings_signature:
            raise ValueError("Restore the baked domain and simulation settings before exporting an older cache without origin metadata.")
        corner = np.asarray(settings.domain_min, dtype=np.float32).astype(np.float64)
        if corner.shape != (3,) or not np.all(np.isfinite(corner)):
            raise ValueError("The baked domain origin is not a finite three-vector.")
        fallback = tuple(float(v) for v in corner + 0.5 * h)
    plan = VDBExportPlan(base, digest, manifest.settings_signature, selected,
                         fallback, h, float(manifest.narrow_band["band_voxels"]) * h)
    for number in selected:
        _frame(plan, number)
    return plan


def export_sequence(plan: VDBExportPlan, destination) -> tuple[Path, ...]:
    """Publish all frames into a NEW destination/frames directory, or publish none.

    Staging is on the same filesystem. Exclusive creation of the destination
    prevents replacing an existing file, directory, or competing export.
    Never recursively remove the user-selected destination on an error.
    """
    dest = Path(destination).absolute()
    if os.path.lexists(dest):
        raise FileExistsError(f"Choose a new destination folder; {dest} already exists and will not be replaced.")
    if not dest.parent.is_dir():
        raise ValueError("The destination parent folder does not exist.")
    manifest = _manifest(plan)
    cache_schema.verified_contiguous_frames(plan.base_dir, manifest)
    stage = Path(tempfile.mkdtemp(prefix=".fluidsim-vdb-", dir=dest.parent))
    reserved = False
    try:
        records = []
        for number in plan.frames:
            payload = _frame(plan, number)
            filename = f"surface_{number:06d}.vdb"
            path = stage / filename
            write_level_set_vdb(path, payload["liquid_sdf_sparse"],
                payload["liquid_sdf_leaf_coords"], plan.voxel_size,
                origin=_sample_origin(plan, payload), background=plan.background, grid_name="surface")
            records.append({"frame":number,"file":filename,
                            "sample_origin":list(_sample_origin(plan,payload)),
                            "origin_source":"cache" if "liquid_sdf_origin" in payload else "matching_scene",
                            "sha256":cache_schema.sha256_file(path)})
        receipt = {"schema":"fluidsim-vdb-export-v1", "grid":"surface",
                   "source_cache":str(plan.base_dir),
                   "manifest_sha256":plan.manifest_sha256,
                   "settings_signature":plan.settings_signature,
                   "frames":list(plan.frames),
                   "voxel_size":plan.voxel_size,"files":records}
        (stage / "export.json").write_text(json.dumps(receipt, indent=2),encoding="utf-8")
        _manifest(plan)
        dest.mkdir()  # atomic exclusive reservation; a racing export must fail
        reserved = True
        stage.rename(dest / "frames")
        return tuple(dest / "frames" / record["file"] for record in records)
    except BaseException:
        if reserved:
            try:
                dest.rmdir()  # only if still empty; never erase another writer
            except OSError:
                pass
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # only the private directory created above
