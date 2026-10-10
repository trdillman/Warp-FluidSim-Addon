"""Cache-to-VDB export contract; Blender read-back is a separate acceptance lane."""
from dataclasses import replace
import json

import numpy as np
import pytest

from fluidSim_warp_blender51.io import cache_schema
from tests.cache_io.fixtures.vdb_cache import build_sdf_cache, fixture_settings
from test_vdb_writer import _read


def api():
    from fluidSim_warp_blender51.io import cache_vdb_export
    return cache_vdb_export


def test_export_two_frames_preserves_cached_cell_centres(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    plan = module.plan_export(base, settings)
    output = tmp_path / "export"
    paths = module.export_sequence(plan, output)
    assert [p.name for p in paths] == ["surface_000001.vdb", "surface_000002.vdb"]
    expected_origin = np.asarray(settings.domain_min,dtype=np.float32).astype(float) + 0.5 * np.float32(settings.voxel_size)
    for path in paths:
        parsed = _read(path)
        np.testing.assert_allclose(parsed["xform"][0],expected_origin,rtol=0,atol=1e-12)
        assert parsed["name"] == "surface"
    receipt = json.loads((output / "frames" / "export.json").read_text())
    assert receipt["frames"] == [1,2]
    assert receipt["manifest_sha256"] == cache_schema.sha256_file(cache_schema.manifest_path(base))


def test_refuses_no_cache_and_full_band_cache(tmp_path):
    module = api()
    with pytest.raises(ValueError,match="Bake"):
        module.plan_export(tmp_path,fixture_settings())
    base, settings = build_sdf_cache(tmp_path / "cache",enabled=False)
    with pytest.raises(ValueError,match="Narrow-Band"):
        module.plan_export(base,settings)


def test_refuses_unproven_world_origin_after_scene_change(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache", include_origin=False)
    with pytest.raises(ValueError,match="baked.*settings|settings.*baked"):
        module.plan_export(base,replace(settings,domain_min=(1.0,2.0,3.0)))


def test_corrupt_later_frame_never_creates_destination(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    path = base / "frames" / "frame_000002.npz"
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ValueError):
        module.plan_export(base,settings)
    assert not (tmp_path / "export").exists()


def test_plan_pins_manifest_and_refuses_changed_cache(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    plan = module.plan_export(base,settings)
    manifest = cache_schema.manifest_path(base)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError,match="changed"):
        module.export_sequence(plan,tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_refuses_existing_destination_without_changing_it(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    dest = tmp_path / "export"
    dest.mkdir()
    keeper = dest / "owned-by-user.txt"
    keeper.write_text("keep")
    with pytest.raises(FileExistsError):
        module.export_sequence(module.plan_export(base,settings),dest)
    assert keeper.read_text() == "keep"
    assert list(dest.iterdir()) == [keeper]


def test_writer_failure_leaves_no_published_sequence(tmp_path,monkeypatch):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    plan = module.plan_export(base,settings)
    calls = []
    writer = module.write_level_set_vdb
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return writer(*args,**kwargs)
    monkeypatch.setattr(module,"write_level_set_vdb",fail_second)
    with pytest.raises(OSError,match="disk full"):
        module.export_sequence(plan,tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not list(tmp_path.glob(".fluidsim-vdb-*"))


def test_ranges_are_contiguous_and_bounded(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    plan = module.plan_export(base,settings,frame_start=2,frame_end=2)
    assert plan.frames == (2,)
    for start,end in [(2,1),(1,3)]:
        with pytest.raises(ValueError,match="range"):
            module.plan_export(base,settings,frame_start=start,frame_end=end)


def test_self_contained_cache_exports_without_scene_or_device(tmp_path):
    from fluidSim_warp_blender51.io import cache
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache")
    payload = cache.read_frame(str(base), 1)
    np.testing.assert_array_equal(payload["liquid_sdf_origin"],
                                  np.asarray(settings.domain_min,dtype=np.float32))
    plan = module.plan_export(base)
    paths = module.export_sequence(plan,tmp_path / "export")
    parsed = _read(paths[0])
    np.testing.assert_allclose(parsed["xform"][0],
        np.asarray(settings.domain_min,dtype=np.float32).astype(float)+0.5*np.float32(settings.voxel_size),atol=1e-12,rtol=0)


def test_old_cache_requires_proven_origin_but_not_a_format_upgrade(tmp_path):
    module = api()
    base, settings = build_sdf_cache(tmp_path / "cache",include_origin=False)
    with pytest.raises(module.LegacyOriginRequired):
        module.plan_export(base)
    assert module.plan_export(base,settings).frames == (1,2)


@pytest.mark.parametrize("origin", [np.zeros(2,dtype=np.float32),
    np.array([0,np.nan,0],dtype=np.float32),np.zeros(3,dtype=np.float64)])
def test_optional_origin_must_be_finite_float32_vector(origin):
    from fluidSim_warp_blender51.io import cache
    payload = cache.empty_liquid_sdf()
    payload["liquid_sdf_origin"] = origin
    with pytest.raises(ValueError,match="liquid_sdf_origin"):
        cache.validate_liquid_sdf(payload,enabled=False)


def test_self_contained_export_has_no_warp_dependency_in_fresh_process(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    base, _ = build_sdf_cache(tmp_path / "cache")
    code = """
import importlib.abc
import sys
class RejectWarp(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'warp' or fullname.startswith('warp.'):
            raise ImportError('Warp deliberately unavailable in export test')
sys.meta_path.insert(0,RejectWarp())
from fluidSim_warp_blender51.io.cache_vdb_export import plan_export,export_sequence
assert len(export_sequence(plan_export(sys.argv[1]),sys.argv[2])) == 2
assert 'warp' not in sys.modules
print('VDB-WARP-FREE-PASS')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH","")
    result = subprocess.run([sys.executable,"-c",code,str(base),str(tmp_path/"export")],
        capture_output=True,text=True,env=env,timeout=30)
    assert result.returncode == 0,result.stdout + result.stderr
    assert "VDB-WARP-FREE-PASS" in result.stdout
