"""Synthetic, manifest-committed SDF cache; no solver, no Warp, no resume claim."""
from pathlib import Path
import hashlib

import numpy as np

from fluidSim_warp_blender51.io import cache, cache_schema, checkpoint
from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings
from tests.cache_io.fixtures.tiny_cache import _payload
from tests.support.vdb_sphere import sphere_leaves


def fixture_settings():
    return SolverSettings(domain_min=(-1.25, -0.5, -0.75),
                          domain_max=(2.75, 3.5, 3.25), voxel_size=0.05,
                          narrow_band_enabled=True, narrow_band_voxels=3.0)


def build_sdf_cache(base, settings=None, *, enabled=True, count=2, include_origin=True):
    settings = settings or fixture_settings()
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    signature = cache.settings_record(settings)["signature"]
    h = float(np.float32(settings.voxel_size))
    origin = np.asarray(settings.domain_min, dtype=np.float32).astype(float) + 0.5 * h
    manifest = cache_schema.CacheManifestV3.new(
        solver_version="3.0.0", scene_key="vdb-fixture", settings_signature=signature,
        source_signature="synthetic-sdf", frame_start=1, frame_end=count,
        source_schema_version=1, transfer_mode="FLIP",
        grid_spec={"resolution_divisions":80,"voxel_size":float(settings.voxel_size),
                   "domain_size":[4.0,4.0,4.0],"leaf_dim":8},
        narrow_band={"enabled":enabled,"band_voxels":3.0 if enabled else 0.0},
        platform={"os":"fixture","gpu_name":"none","driver":"none","warp":"none"})
    for frame in range(1, count + 1):
        payload = _payload(frame, signature)
        payload.voxel_size = h
        values, coords, _ = sphere_leaves(radius=0.3, voxel=h,
            center=(0.1 * frame, 0.3, 0.1), origin=origin)
        sdf = {"liquid_sdf_sparse":values,"liquid_sdf_leaf_coords":coords,
               "liquid_sdf_voxel_size":np.array(h,dtype=np.float32)}
        if include_origin:
            sdf["liquid_sdf_origin"] = np.asarray(settings.domain_min, dtype=np.float32)
        stamp = hashlib.sha256(f"frame-{frame}".encode()).hexdigest()
        path, digest, size = cache.write_frame_v3_atomic(base,payload,frame,
            scene_key_value=manifest.scene_key,settings_signature=signature,
            source_signature=manifest.source_signature,source_frame_stamp=stamp,
            simulation_time=frame/24.0,
            liquid_sdf=sdf if enabled else None,narrow_band_enabled=enabled)
        cp = base / "checkpoints" / f"checkpoint_{frame:06d}.npz"
        cdigest = checkpoint.write_checkpoint_atomic(cp, checkpoint.CheckpointPayload(
            frame=frame,sim_time=frame/24.0,settings_signature=signature,
            source_signature=manifest.source_signature))
        manifest = manifest.with_committed_frame(cache_schema.FrameEntry(
            frame=frame,frame_file=path.relative_to(base).as_posix(),
            frame_sha256=digest,frame_size=size,checkpoint_file=cp.relative_to(base).as_posix(),
            checkpoint_sha256=cdigest,checkpoint_size=cp.stat().st_size,
            completed_at="2026-10-10T00:00:00Z",solver_frame=frame,simulation_time=frame/24.0,
            settings_signature=signature,source_signature=manifest.source_signature,
            source_frame_stamp=stamp,whitewater_count=0,particle_count=len(payload.positions)))
    cache_schema.write_manifest_atomic(cache_schema.manifest_path(base),manifest)
    return base, settings
