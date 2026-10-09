"""Minimal OpenVDB (.vdb) writer for one float level-set grid. numpy + stdlib.

Spike module (3.0 lane I). Blender ships no ``pyopenvdb`` and reads only
``.vdb``, so this emits the OpenVDB file format directly.

Input is the cache triple ``liquid_sdf_sparse`` / ``liquid_sdf_leaf_coords`` /
``liquid_sdf_voxel_size`` (``io/cache.py``).

Conventions
-----------
* ``leaf_coords`` are **leaf indices** (units of 8 voxels), the same
  convention as ``sim.capacity_planner.leaf_coords_for_world_bounds``: voxel
  ``(i, j, k)`` of leaf ``(a, b, c)`` is the index-space voxel
  ``(8a + i, 8b + j, 8c + k)``.
* Cache leaves are flat ``(lz * 8 + ly) * 8 + lx`` (x fastest). OpenVDB leaves
  are x slowest, so the writer transposes.
* Index-space voxel ``(0, 0, 0)`` has its sample at world ``origin`` (default
  the world origin); the transform is a uniform scale of ``voxel_size`` plus
  that translation.

File layout: format version 224, tree ``Tree_float_5_4_3``, no compression
(grid compression flags 0, "no mask, all values" per node buffer), one grid.
A voxel is active when ``abs(value) < background``.
"""
from __future__ import annotations

import struct
import uuid as _uuid

import numpy as np

LEAF_DIM = 8
LEAF_VOLUME = LEAF_DIM ** 3

FILE_VERSION = 224
LIBRARY_VERSION = (12, 0)
_MAGIC = 0x56444220
_TREE_TYPE = "Tree_float_5_4_3"
_NO_MASK_AND_ALL_VALS = 6

# Node geometry: leaf 2^3, internal 2^4, internal 2^5.
_L1_DIM = 16
_L2_DIM = 32
_L1_SPAN = 128      # voxels spanned by one level-1 node per axis
_L2_SPAN = 4096


def _s(text: str) -> bytes:
    raw = text.encode("ascii")
    return struct.pack("<I", len(raw)) + raw


def _meta_string(name: str, value: str) -> bytes:
    raw = value.encode("ascii")
    return _s(name) + _s("string") + struct.pack("<I", len(raw)) + raw


def _meta_float(name: str, value: float) -> bytes:
    return _s(name) + _s("float") + struct.pack("<I", 4) + struct.pack("<f", value)


def _mask_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()


def _transform(voxel_size: float, origin) -> bytes:
    v = float(voxel_size)
    out = [_s("UniformScaleTranslateMap")]
    for triple in (tuple(float(x) for x in origin), (v,) * 3, (v,) * 3,
                   (1.0 / v,) * 3, (1.0 / (v * v),) * 3, (0.5 / v,) * 3):
        out.append(struct.pack("<3d", *triple))
    return b"".join(out)


def _build_tree(leaf_values, leaf_coords):
    """Group leaves as {root_origin: {l2_index: {l1_index: {leaf_index: row}}}}."""
    tree: dict = {}
    for row, (a, b, c) in enumerate(leaf_coords.tolist()):
        x, y, z = a * LEAF_DIM, b * LEAF_DIM, c * LEAF_DIM
        root = (x & ~(_L2_SPAN - 1), y & ~(_L2_SPAN - 1), z & ~(_L2_SPAN - 1))
        l2 = ((((x & (_L2_SPAN - 1)) // _L1_SPAN) << 10)
              | (((y & (_L2_SPAN - 1)) // _L1_SPAN) << 5)
              | ((z & (_L2_SPAN - 1)) // _L1_SPAN))
        l1 = ((((x & (_L1_SPAN - 1)) // LEAF_DIM) << 8)
              | (((y & (_L1_SPAN - 1)) // LEAF_DIM) << 4)
              | ((z & (_L1_SPAN - 1)) // LEAF_DIM))
        tree.setdefault(root, {}).setdefault(l2, {}).setdefault(l1, {})[0] = None
        tree[root][l2][l1] = row
    return tree


def _node_header(child_indices, node_dim, background) -> bytes:
    count = node_dim ** 3
    child_bits = np.zeros(count, dtype=bool)
    child_bits[list(child_indices)] = True
    return b"".join((
        _mask_bytes(child_bits),
        _mask_bytes(np.zeros(count, dtype=bool)),
        struct.pack("<b", _NO_MASK_AND_ALL_VALS),
        np.full(count, background, dtype="<f4").tobytes(),
    ))


def _leaf_row_xslow(values_row: np.ndarray) -> np.ndarray:
    # cache flat index is (z*8+y)*8+x; OpenVDB offset is (x*8+y)*8+z.
    return values_row.reshape(LEAF_DIM, LEAF_DIM, LEAF_DIM).transpose(2, 1, 0).reshape(-1)


def _write_grid_body(values, tree, voxel_size, origin, background, grid_name):
    """Return (grid_head, topology, buffers): compression..topology, then buffers."""
    meta = [
        _meta_string("class", "level set"),
        _meta_float("file_voxel_size", float(voxel_size)),
        _meta_string("name", grid_name),
    ]
    head = (struct.pack("<I", 0)  # compression flags: none
            + struct.pack("<I", len(meta)) + b"".join(meta)
            + _transform(voxel_size, origin))

    topo = [struct.pack("<i", 1),                      # tree buffer count
            struct.pack("<f", background),             # root background
            struct.pack("<II", 0, len(tree))]          # tiles, children
    bufs = []
    for root in sorted(tree):
        topo.append(struct.pack("<3i", *root))
        l2_nodes = tree[root]
        topo.append(_node_header(l2_nodes.keys(), _L2_DIM, background))
        for l2 in sorted(l2_nodes):
            l1_nodes = l2_nodes[l2]
            topo.append(_node_header(l1_nodes.keys(), _L1_DIM, background))
            for l1 in sorted(l1_nodes):
                data = _leaf_row_xslow(values[l1_nodes[l1]]).astype("<f4")
                mask = _mask_bytes(np.abs(data) < background)
                topo.append(mask)
                bufs.append(mask + struct.pack("<b", _NO_MASK_AND_ALL_VALS)
                            + data.tobytes())
    return head, b"".join(topo), b"".join(bufs)


def write_level_set_vdb(path, leaf_values, leaf_coords, voxel_size, *,
                        background=None, grid_name="surface",
                        origin=(0.0, 0.0, 0.0)) -> None:
    """Write ``leaf_values`` (N, 512) float32 at ``leaf_coords`` (N, 3) int32.

    ``background`` defaults to ``3 * voxel_size``. Missing leaves read back as
    ``+background`` (exterior); no interior tile fill is written.
    """
    values = np.asarray(leaf_values)
    coords = np.asarray(leaf_coords)
    voxel_size = float(voxel_size)
    if values.ndim != 2 or values.shape[1] != LEAF_VOLUME or values.dtype != np.float32:
        raise ValueError(f"leaf_values must be float32 (N, {LEAF_VOLUME}); "
                         f"got {values.dtype} {values.shape}")
    if coords.ndim != 2 or coords.shape != (values.shape[0], 3) or coords.dtype.kind not in "iu":
        raise ValueError("leaf_coords must be an integer array of shape (N, 3)")
    if not (np.isfinite(voxel_size) and voxel_size > 0.0):
        raise ValueError("voxel_size must be a positive finite number")
    if values.shape[0] == 0:
        raise ValueError("at least one leaf is required")
    if len({tuple(c) for c in coords.tolist()}) != coords.shape[0]:
        raise ValueError("leaf_coords must be unique")
    if not np.all(np.isfinite(values)):
        raise ValueError("leaf_values must be finite")
    bg = np.float32(3.0 * voxel_size if background is None else background)
    if not bg > 0:
        raise ValueError("background must be positive")
    bg = float(bg)

    tree = _build_tree(values, coords)
    head, topo, bufs = _write_grid_body(
        values, tree, voxel_size, origin, bg, str(grid_name))

    header = b"".join((
        struct.pack("<i", _MAGIC),
        struct.pack("<I", FILE_VERSION),
        struct.pack("<II", *LIBRARY_VERSION),
        struct.pack("<b", 1),                               # has grid offsets
        str(_uuid.uuid4()).encode("ascii"),                 # 36 chars
        struct.pack("<I", 0),                               # file metadata count
        struct.pack("<i", 1),                               # grid count
    ))
    desc = _s(grid_name) + _s(_TREE_TYPE) + _s("")
    grid_pos = len(header) + len(desc) + 24
    block_pos = grid_pos + len(head) + len(topo)
    end_pos = block_pos + len(bufs)
    with open(path, "wb") as fh:
        fh.write(header + desc + struct.pack("<3q", grid_pos, block_pos, end_pos))
        fh.write(head + topo + bufs)
