"""Synthetic sphere SDF in the cache's sparse-leaf layout (numpy only)."""
import numpy as np

LEAF = 8


def sphere_leaves(radius=0.5, voxel=0.05, center=(0.0, 0.0, 0.0), band=3.0,
                  origin=(0.0, 0.0, 0.0)):
    """Return (values (N,512) f32, coords (N,3) i32, voxel) for a sphere.

    Voxel (i,j,k) is sampled at ``origin + (i,j,k) * voxel``. Leaves are kept
    when any voxel is inside the narrow band (|sdf| < band * voxel). Values
    are clamped to +-band*voxel; flat order is cache order (x fastest).
    """
    bg = band * voxel
    c_idx = (np.asarray(center, float) - np.asarray(origin, float)) / voxel
    ext = (radius + bg) / voxel + 1.0
    lo = np.floor((c_idx - ext) / LEAF).astype(int)
    hi = np.ceil((c_idx + ext) / LEAF).astype(int)
    local = np.arange(LEAF)
    lz, ly, lx = np.meshgrid(local, local, local, indexing="ij")  # z slowest
    values, coords = [], []
    for a in range(lo[0], hi[0] + 1):
        for b in range(lo[1], hi[1] + 1):
            for c in range(lo[2], hi[2] + 1):
                x = origin[0] + (a * LEAF + lx) * voxel - center[0]
                y = origin[1] + (b * LEAF + ly) * voxel - center[1]
                z = origin[2] + (c * LEAF + lz) * voxel - center[2]
                sdf = np.sqrt(x * x + y * y + z * z) - radius
                if np.any(np.abs(sdf) < bg):
                    values.append(np.clip(sdf, -bg, bg).astype(np.float32).reshape(-1))
                    coords.append((a, b, c))
    return (np.stack(values), np.asarray(coords, dtype=np.int32), voxel)
