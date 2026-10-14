import gc
import sys
import types
import weakref

import numpy as np

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation, SourcePayloadKind,
)
from fluidSim_warp_blender51.sim.source_regions import (
    CompileContext, PayloadPlan, SourceCompiler, SourceKey, SourcePlan, VolumeRegionPlan,
)


class Accessor:
    def __init__(self, role): self.role = role
    def getValue(self, coord):
        x,y,z = coord
        if self.role == "fog":
            return 1.0 if x < 2 else 0.0
        if self.role == "sdf":
            return float(x) - 1.5
        if self.role == "scalar":
            return float(x+y+z)
        if self.role == "vx":
            return float(x)
        if self.role == "vy":
            return float(y)
        if self.role == "vz":
            return float(z)
        return (float(x),float(y),float(z))


class Grid:
    def __init__(self, role): self.role = role
    def evalActiveVoxelBoundingBox(self): return ((0,0,0),(3,3,3))
    def getConstAccessor(self): return Accessor(self.role)


def plan(path, **grids):
    payload = PayloadPlan(kind=SourcePayloadKind.SCALAR)
    return SourcePlan(SourceKey("volume"),SourceOperation.SCALAR_FIELD,
                      SourceLifecycle.EVERY_FRAME,SourceEvaluationCadence.STATIC,
                      VolumeRegionPlan(str(path),**grids),payload)


def staggered_plan(path):
    transform = tuple(np.diag((0.25, 0.25, 0.25, 1.0)).reshape(-1))
    return SourcePlan(
        SourceKey("staggered"), SourceOperation.STAGGERED_VELOCITY,
        SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
        VolumeRegionPlan(str(path), mask_grid="fog", vector_grid="vector",
                         transform=transform),
        PayloadPlan(kind=SourcePayloadKind.STAGGERED_VECTOR),
    )


def context():
    return CompileContext(1,0,1,"cpu",(0,0,0),0.25,(0,0,0),(1,1,1),upload=False)


def test_volume_grids_flatten_mask_sdf_scalar_vector_and_release_trees(tmp_path,monkeypatch):
    path = tmp_path/"fixture.vdb"
    path.write_bytes(b"fixture")
    refs = []
    def read(_path,name):
        grid = Grid(name)
        refs.append(weakref.ref(grid))
        return grid
    monkeypatch.setitem(sys.modules,"openvdb",types.SimpleNamespace(read=read))
    region = SourceCompiler().compile((plan(
        path,mask_grid="fog",sdf_grid="sdf",scalar_grid="scalar",
        vector_grid="vector"),),context()).sources[0].region
    assert region.kind.name == "VOLUME_GRID"
    assert region.coverage.shape == (1,512)
    assert region.scalar is not None and region.vector.shape == (1,512,3)
    assert np.count_nonzero(region.coverage) > 0
    gc.collect()
    assert refs and all(ref() is None for ref in refs)


def test_volume_staggered_compilation_samples_each_face_position(tmp_path, monkeypatch):
    path = tmp_path / "staggered.vdb"
    path.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "openvdb", types.SimpleNamespace(
        read=lambda _path, name: Grid(name)))
    region = SourceCompiler().compile((staggered_plan(path),), context()).sources[0].region
    assert region.staggered_u is not None
    assert region.staggered_v is not None
    assert region.staggered_w is not None
    # Vector grid value is index coordinate. X component sampled at x-face is
    # half an index lower than the centered vector sample, and likewise y/z.
    assert np.max(np.abs(region.staggered_u - region.vector[..., 0])) > 0.0
    assert np.max(np.abs(region.staggered_v - region.vector[..., 1])) > 0.0
    assert np.max(np.abs(region.staggered_w - region.vector[..., 2])) > 0.0


def test_volume_plan_bounds_contribute_transformed_active_topology(tmp_path, monkeypatch):
    path = tmp_path / "bounds.vdb"
    path.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "openvdb", types.SimpleNamespace(
        read=lambda _path, _name: Grid("fog")))
    transform = tuple(np.array(((2,0,0,1),(0,3,0,2),(0,0,4,3),(0,0,0,1)),
                               np.float64).reshape(-1))
    bounds = VolumeRegionPlan(str(path), mask_grid="fog", transform=transform).bounds()
    assert np.allclose(bounds[0], (1,2,3))
    assert np.allclose(bounds[1], (9,14,19))


def test_volume_vector_component_grids_compile_to_vector_and_staggered(tmp_path, monkeypatch):
    path = tmp_path / "components.vdb"
    path.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "openvdb", types.SimpleNamespace(
        read=lambda _path, name: Grid(name)))
    region_plan = VolumeRegionPlan(
        str(path), mask_grid="fog", vector_x_grid="vx",
        vector_y_grid="vy", vector_z_grid="vz",
        transform=tuple(np.diag((0.25,0.25,0.25,1.0)).reshape(-1)))
    source = SourcePlan(SourceKey("components"), SourceOperation.STAGGERED_VELOCITY,
                        SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
                        region_plan, PayloadPlan(kind=SourcePayloadKind.STAGGERED_VECTOR))
    region = SourceCompiler().compile((source,), context()).sources[0].region
    assert region.vector is not None
    assert all(value is not None for value in (
        region.staggered_u, region.staggered_v, region.staggered_w))


def test_vdb_authored_grid_transform_composes_with_plan_transform(tmp_path, monkeypatch):
    path = tmp_path / "transformed.vdb"
    path.write_bytes(b"fixture")

    class Transform:
        def isLinear(self): return True
        def indexToWorld(self, coord):
            return tuple(np.asarray(coord, np.float64) * 0.5 + (2.0, 3.0, 4.0))

    class TransformedGrid(Grid):
        transform = Transform()

    monkeypatch.setitem(sys.modules, "openvdb", types.SimpleNamespace(
        read=lambda _path, _name: TransformedGrid("fog")))
    plan_transform = tuple(np.array(
        ((1,0,0,10),(0,1,0,20),(0,0,1,30),(0,0,0,1)),
        np.float64).reshape(-1))
    bounds = VolumeRegionPlan(
        str(path), mask_grid="fog", transform=plan_transform).bounds()
    assert np.allclose(bounds[0], (12.0, 23.0, 34.0))
    assert np.allclose(bounds[1], (14.0, 25.0, 36.0))


def test_missing_grid_is_actionable_not_silently_partial(tmp_path,monkeypatch):
    path = tmp_path/"fixture.vdb"
    path.write_bytes(b"fixture")
    def read(_path,name): raise RuntimeError("grid not found")
    monkeypatch.setitem(sys.modules,"openvdb",types.SimpleNamespace(read=read))
    region = SourceCompiler().compile((plan(path,scalar_grid="missing"),),context()).sources[0].region
    assert region.leaf_count == 0
    assert any(d.code == "SOURCE_COMPILE_FAILED" and "missing" in d.message
               for d in region.diagnostics)
