"""Regression tests from the cache-layer review (io/*, writers and loaders)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender import production_bake
from fluidSim_warp_blender51.io import cache, cache_schema
from fluidSim_warp_blender51.io import export as export_mod
from fluidSim_warp_blender51.io import fsutil
from tests.cache_io.fixtures.tiny_cache import build_tiny_cache
from tests.cache_io.fixtures.vdb_cache import build_sdf_cache


def _edit_manifest(base, edit):
    path = Path(cache_schema.manifest_path(base))
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


# --- manifest loading ------------------------------------------------------

def test_sparse_manifest_is_refused_without_building_a_huge_list(tmp_path):
    base = build_tiny_cache(tmp_path / "c")

    def edit(m):
        entry = m["frames"]["1"]
        m["frame_start"] = 0
        m["frame_end"] = 10**12
        m["frames"] = {str(10**12): dict(
            entry, frame=10**12, solver_frame=10**12)}

    _edit_manifest(base, edit)
    with pytest.raises(ValueError, match="contiguous"):
        cache_schema.load_manifest(cache_schema.manifest_path(base))


def test_huge_json_integer_is_a_value_error_not_overflow(tmp_path):
    base = build_tiny_cache(tmp_path / "c")
    _edit_manifest(
        base, lambda m: m["grid_spec"].update(voxel_size=10**400))
    with pytest.raises(ValueError):
        cache_schema.load_manifest(cache_schema.manifest_path(base))


@pytest.mark.parametrize("raw", [b"\xff\xfe\x00bad", b"[" * 200000])
def test_unreadable_manifest_bytes_are_a_clean_value_error(tmp_path, raw):
    base = build_tiny_cache(tmp_path / "c")
    Path(cache_schema.manifest_path(base)).write_bytes(raw)
    with pytest.raises(ValueError, match="invalid cache manifest"):
        cache_schema.load_manifest(cache_schema.manifest_path(base))
    # Format detection must not crash on the same file either.
    assert cache.detect_cache_format(base) == cache_schema.FORMAT_VERSION


@pytest.mark.parametrize("bad_key", ["01", "-0", "\u0663", "\u00b2", "+1", " 1"])
def test_noncanonical_frame_keys_are_refused(tmp_path, bad_key):
    base = build_tiny_cache(tmp_path / "c")

    def edit(m):
        m["frames"][bad_key] = dict(m["frames"]["1"])

    _edit_manifest(base, edit)
    with pytest.raises(ValueError, match="invalid frame key"):
        cache_schema.load_manifest(cache_schema.manifest_path(base))


# --- writers -----------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_frame_files_follow_the_umask_like_the_manifest(tmp_path):
    old = os.umask(0o022)
    try:
        base = build_tiny_cache(tmp_path / "c")
    finally:
        os.umask(old)
    frame = next((Path(base) / "frames").glob("frame_*.npz"))
    assert stat.S_IMODE(frame.stat().st_mode) == 0o644


def test_frame_digest_matches_the_published_file(tmp_path):
    base = build_tiny_cache(tmp_path / "c")
    manifest = cache_schema.load_manifest(cache_schema.manifest_path(base))
    for entry in manifest.frames.values():
        path = Path(base) / entry.frame_file
        assert cache_schema.sha256_file(path) == entry.frame_sha256
        assert path.stat().st_size == entry.frame_size


def test_directory_fsync_refusal_does_not_fail_a_write(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("directory fsync is skipped on Windows")
    real_fsync = os.fsync

    def picky(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", picky)
    fsutil.fsync_directory(tmp_path)  # must not raise
    base = build_tiny_cache(tmp_path / "c")
    assert (Path(base) / "frames" / "frame_000001.npz").is_file()


# --- exports -------------------------------------------------------------------

def _solver(*, surface=True, particles=1, whitewater=0, tris=None):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    host = SimpleNamespace(
        surface_mesh=(verts, np.zeros((0, 3), np.int32) if tris is None else tris),
        positions=np.ones((particles, 3), np.float32),
        velocities=np.zeros((particles, 3), np.float32),
        whitewater=np.ones((whitewater, 3), np.float32),
    )
    return SimpleNamespace(
        host_frame=lambda: host,
        settings=SimpleNamespace(
            surface_enabled=surface, surface_resolution_mode=0,
            surface_resolution_multiplier=1.0, surface_smooth_iterations=0),
        surface=SimpleNamespace(metadata=lambda: {"voxel_size": 0.1}),
    )


def test_export_frame_with_nothing_to_write_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Nothing to export"):
        export_mod.export_frame(
            _solver(surface=False, particles=0), str(tmp_path / "f"))
    assert list(tmp_path.iterdir()) == []


def test_surface_triangle_indices_must_be_in_range(tmp_path):
    bad = np.array([[0, 1, 7]], dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        export_mod.export_surface_obj(
            _solver(tris=bad), str(tmp_path / "s.obj"))
    assert not (tmp_path / "s.obj").exists()
    good = np.array([[0, 1, 2]], dtype=np.int32)
    export_mod.export_surface_obj(_solver(tris=good), str(tmp_path / "s.obj"))
    assert "f 1 2 3" in (tmp_path / "s.obj").read_text()


def test_interrupt_during_publish_restores_the_previous_bundle(
        tmp_path, monkeypatch):
    base = str(tmp_path / "f")
    export_mod.export_frame(_solver(), base)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert len(before) == 2
    real_replace = os.replace
    calls = {"n": 0}

    def interrupting(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:  # the second new file being published
            raise KeyboardInterrupt
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", interrupting)
    with pytest.raises(KeyboardInterrupt):
        export_mod.export_frame(_solver(particles=2), base)
    monkeypatch.undo()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_single_file_export_keeps_old_file_when_the_write_fails(
        tmp_path, monkeypatch):
    path = tmp_path / "p.ply"
    export_mod.export_points_ply(np.zeros((1, 3), np.float32), str(path))
    good = path.read_bytes()

    def broken_writer(stage, payload):
        Path(stage).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(export_mod, "_write_points_ply", broken_writer)
    with pytest.raises(OSError, match="disk full"):
        export_mod.export_points_ply(np.ones((2, 3), np.float32), str(path))
    assert path.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir()] == ["p.ply"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_exports_follow_the_umask(tmp_path):
    old = os.umask(0o022)
    try:
        export_mod.export_frame(_solver(), str(tmp_path / "f"))
    finally:
        os.umask(old)
    modes = {stat.S_IMODE(p.stat().st_mode) for p in tmp_path.iterdir()}
    assert modes == {0o644}


# --- VDB range planning ----------------------------------------------------------

def test_vdb_plan_frame_zero_is_a_real_frame_not_a_full_range(tmp_path):
    from fluidSim_warp_blender51.io import cache_vdb_export as vdb

    base, settings = build_sdf_cache(tmp_path / "cache")
    full = vdb.plan_export(base, settings)
    assert full.frames == vdb.plan_export(
        base, settings, frame_start=None, frame_end=None).frames
    # The cache holds frames 1..2, so frame 0 is outside it, not "everything".
    with pytest.raises(ValueError, match="outside the cached range"):
        vdb.plan_export(base, settings, frame_start=0)
    with pytest.raises(ValueError, match="outside the cached range"):
        vdb.plan_export(base, settings, frame_end=0)


# --- rebake cleanup ----------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("frame_000010.npz", 10),
    ("checkpoint_000003.npz", 3),
    ("frame_-00002.npz", -2),
    ("frame_000010.npz.ab_cd1.tmp", 10),
])
def test_rebake_cleanup_pattern_matches_frame_checkpoint_and_temp_files(
        name, expected):
    match = production_bake._FRAME_FILE_RE.match(name)
    assert match is not None and int(match.group(1)) == expected


@pytest.mark.parametrize("name", [
    "frame_000010.npz.bak", "notes.npz", "frame_x.npz", "frame_000010.npz.old.txt",
])
def test_rebake_cleanup_pattern_leaves_other_files_alone(name):
    assert production_bake._FRAME_FILE_RE.match(name) is None


# --- Windows path helper ---------------------------------------------------------------

class _FakePath:
    def __init__(self, text):
        self.text = text

    def resolve(self):
        return self.text


@pytest.mark.parametrize("text,expected", [
    ("C:\\proj\\cache", "\\\\?\\C:\\proj\\cache"),
    ("\\\\server\\share\\c", "\\\\?\\UNC\\server\\share\\c"),
    ("\\\\?\\C:\\proj", "\\\\?\\C:\\proj"),
    ("\\\\.\\PIPE\\x", "\\\\.\\PIPE\\x"),
])
def test_filesystem_path_prefixes_on_windows(monkeypatch, text, expected):
    from fluidSim_warp_blender51 import filesystem

    monkeypatch.setattr(filesystem.os, "name", "nt")
    monkeypatch.setattr(filesystem, "Path", lambda value: str(value))
    assert filesystem.filesystem_path(_FakePath(text)) == expected


# --- frame output cache ------------------------------------------------------------------

def test_surface_velocity_is_not_reused_for_different_points():
    from fluidSim_warp_blender51.sim.frame_output import HostFrame

    solver = SimpleNamespace(
        settings=SimpleNamespace(voxel_size=1.0),
        particle_positions=lambda: np.array([[0, 0, 0], [10, 0, 0]], np.float32),
        particle_velocities=lambda: np.array([[1, 0, 0], [2, 0, 0]], np.float32),
    )
    frame = HostFrame(solver)
    first = frame.surface_velocity([[0, 0, 0]])
    second = frame.surface_velocity([[10, 0, 0]])
    assert first.tolist() == [[1.0, 0.0, 0.0]]
    assert second.tolist() == [[2.0, 0.0, 0.0]]
    again = frame.surface_velocity([[10, 0, 0]])
    assert again is second


def test_directory_destination_is_refused_and_left_untouched(tmp_path):
    target = tmp_path / "out.ply"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(IsADirectoryError):
        export_mod.export_points_ply(np.zeros((1, 3), np.float32), str(target))
    with pytest.raises(IsADirectoryError):
        export_mod.export_surface_obj(_solver(), str(target))
    assert (target / "keep.txt").read_text() == "keep"
    assert [p.name for p in tmp_path.iterdir()] == ["out.ply"]
