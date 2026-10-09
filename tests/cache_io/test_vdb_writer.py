"""Byte-layout and round-trip tests for io/vdb_writer.py.

The reader below parses the writer's own layout; it proves self-consistency,
not OpenVDB compatibility (that is tests/acceptance/blender_vdb_writer_acceptance.py).
"""
import struct

import numpy as np
import pytest

from fluidSim_warp_blender51.io.vdb_writer import write_level_set_vdb
from tests.support.vdb_sphere import sphere_leaves


class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        out = self.d[self.p:self.p + n]
        assert len(out) == n
        self.p += n
        return out

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def s(self):
        return self.take(self.u("I")[0]).decode()


def _mask(r, nbits):
    return np.unpackbits(np.frombuffer(r.take(nbits // 8), np.uint8), bitorder="little")


def _read(path):
    r = _R(open(path, "rb").read())
    out = {"magic": r.u("q")[0], "version": r.u("I")[0], "lib": r.u("II"),
           "offsets": r.u("b")[0], "uuid": r.take(36).decode()}
    assert r.u("I")[0] == 0 and r.u("i")[0] == 1
    out["name"], out["type"], parent = r.s(), r.s(), r.s()
    assert parent == ""
    grid_pos, block_pos, end_pos = r.u("3q")
    assert r.p == grid_pos
    assert r.u("I")[0] == 0
    meta = {}
    for _ in range(r.u("I")[0]):
        name, kind = r.s(), r.s()
        size = r.u("I")[0]
        raw = r.take(size)
        meta[name] = raw.decode() if kind == "string" else struct.unpack("<f", raw)[0]
    out["meta"] = meta
    out["map"] = r.s()
    out["xform"] = [r.u("3d") for _ in range(6)]
    assert r.u("i")[0] == 1
    out["background"] = r.u("f")[0]
    tiles, nroot = r.u("II")
    assert tiles == 0
    leaves = []  # (voxel origin, mask)

    def node(dim):
        child = _mask(r, dim ** 3)
        assert not _mask(r, dim ** 3).any()
        assert r.u("b")[0] == 6
        vals = np.frombuffer(r.take(4 * dim ** 3), "<f4")
        assert np.all(vals == out["background"])
        return np.flatnonzero(child)

    for _ in range(nroot):
        ox, oy, oz = r.u("3i")
        for i2 in node(32):
            b2 = (ox + (i2 >> 10) * 128, oy + ((i2 >> 5) & 31) * 128, oz + (i2 & 31) * 128)
            for i1 in node(16):
                o = (b2[0] + (i1 >> 8) * 8, b2[1] + ((i1 >> 4) & 15) * 8, b2[2] + (i1 & 15) * 8)
                leaves.append([o, _mask(r, 512)])
    assert r.p == block_pos
    for leaf in leaves:
        assert np.array_equal(_mask(r, 512), leaf[1])
        assert r.u("b")[0] == 6
        leaf.append(np.frombuffer(r.take(2048), "<f4").reshape(8, 8, 8))  # [x,y,z]
    assert r.p == end_pos == len(r.d)
    out["leaves"] = leaves
    return out


def _sphere(tmp_path, **kw):
    values, coords, voxel = sphere_leaves(**kw)
    path = tmp_path / "s.vdb"
    write_level_set_vdb(path, values, coords, voxel, origin=kw.get("origin", (0, 0, 0)))
    return values, coords, voxel, _read(path)


def test_header_metadata_and_transform(tmp_path):
    _, _, voxel, f = _sphere(tmp_path)
    assert f["magic"] == 0x56444220 and f["version"] == 224 and f["offsets"] == 1
    assert len(f["uuid"]) == 36
    assert f["name"] == "surface" and f["type"] == "Tree_float_5_4_3"
    assert f["meta"]["class"] == "level set" and f["meta"]["name"] == "surface"
    assert f["meta"]["file_voxel_size"] == pytest.approx(voxel)
    assert f["map"] == "UniformScaleTranslateMap"
    assert f["xform"][1] == pytest.approx((voxel,) * 3)
    assert f["background"] == pytest.approx(3 * voxel)


def test_leaves_round_trip_with_x_slowest_layout(tmp_path):
    values, coords, _, f = _sphere(
        tmp_path, center=(0.3, -0.2, 0.1), origin=(-1.0, -1.5, -0.25))
    assert f["xform"][0] == pytest.approx((-1.0, -1.5, -0.25))
    got = {tuple(o): (m, v) for o, m, v in f["leaves"]}
    assert len(got) == len(coords)
    for row, c in enumerate(coords.tolist()):
        mask, vox = got[tuple(8 * np.array(c))]
        cache = values[row].reshape(8, 8, 8)  # [z, y, x]
        assert np.array_equal(vox, cache.transpose(2, 1, 0))
        active = np.abs(vox) < f["background"]
        assert np.array_equal(mask.reshape(8, 8, 8), active.astype(np.uint8))


def test_leaves_spanning_negative_coords_and_node_boundaries(tmp_path):
    coords = np.array([[-1, -1, -1], [0, 0, 0], [15, 15, 15], [16, 0, 0], [512, 0, -513]],
                      dtype=np.int32)
    values = np.random.default_rng(1).uniform(-0.1, 0.1, (5, 512)).astype(np.float32)
    path = tmp_path / "b.vdb"
    write_level_set_vdb(path, values, coords, 0.05)
    f = _read(path)
    assert sorted(tuple(int(v) for v in o) for o, _, _ in f["leaves"]) == sorted(
        tuple(8 * v for v in c) for c in coords.tolist())


@pytest.mark.parametrize("kwargs, message", [
    (dict(voxel_size=0.0), "voxel_size"),
    (dict(leaf_values=np.zeros((1, 512), np.float64)), "float32"),
    (dict(leaf_values=np.zeros((0, 512), np.float32), leaf_coords=np.zeros((0, 3), np.int32)),
     "at least one"),
    (dict(leaf_coords=np.zeros((2, 3), np.int32), leaf_values=np.zeros((2, 512), np.float32)),
     "unique"),
])
def test_rejects_bad_input(tmp_path, kwargs, message):
    args = dict(leaf_values=np.zeros((1, 512), np.float32),
                leaf_coords=np.zeros((1, 3), np.int32), voxel_size=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        write_level_set_vdb(tmp_path / "x.vdb", **args)
