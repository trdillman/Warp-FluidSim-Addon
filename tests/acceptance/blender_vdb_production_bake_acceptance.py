"""Real CUDA production bake -> committed SDF -> public VDB exporter -> Blender reader."""
from pathlib import Path
import json
import sys
import tempfile

import bpy
import numpy as np


def main(output_dir):
    from fluidSim_warp_blender51.blender import production_bake, scene_bridge
    from fluidSim_warp_blender51.io import cache, cache_schema
    from fluidSim_warp_blender51.preferences import probe_runtime

    runtime = probe_runtime()
    if not runtime.cuda_devices:
        raise RuntimeError("Production VDB acceptance requires certified CUDA; not a CPU qualification")
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    scene = bpy.data.scenes.new("VDBProductionBake")
    bpy.context.window.scene = scene
    scene.fluidsim.live = False
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.4, 0.2, 0.5))
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    domain = scene_bridge.active_domain_settings(scene)
    domain.resolution_divisions = 32
    domain.max_particles = 100000
    domain.auto_grow_particles = False
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    domain.export_attributes = False
    domain.narrow_band_enabled = True
    domain.profiling_enabled = False
    scene.fluidsim.live = False
    scene.fluidsim.cache_path = str(out / "cache")
    scene.fluidsim.cache_use_custom_range = True
    scene.fluidsim.cache_frame_start = 1
    scene.fluidsim.cache_frame_end = 2
    bpy.context.view_layer.update()
    backend = production_bake.ProductionBakeBackend()
    first = backend.initialize(scene,1,2)
    try:
        assert first == 1
        corner = np.asarray(tuple(backend.sess.solver.grid.origin),dtype=np.float32)
        for frame in range(first,3):
            backend.step_and_commit(scene,frame)
    finally:
        backend.finish(scene)
    base = Path(scene.fluidsim.cache_path)
    manifest = cache_schema.load_manifest(cache_schema.manifest_path(base))
    assert manifest.committed_frames == (1,2)
    counts = []
    for number in manifest.committed_frames:
        payload = cache.read_frame(str(base),number)
        np.testing.assert_array_equal(payload["liquid_sdf_origin"],corner)
        assert payload["liquid_sdf_sparse"].min() < 0
        assert len(payload["positions"]) > 0
        counts.append(len(payload["positions"]))
    digest = cache_schema.sha256_file(cache_schema.manifest_path(base))
    assert bpy.ops.fluidsim.export_vdb(directory=str(out),sequence_name="production-vdb") == {"FINISHED"}
    files = sorted((out / "production-vdb" / "frames").glob("*.vdb"))
    assert len(files) == 2
    for path in files:
        volume = bpy.data.volumes.new("ProductionReadback")
        volume.filepath = str(path)
        volume.grids.load()
        assert volume.grids.is_loaded and "surface" in volume.grids, volume.grids.error_message
    assert digest == cache_schema.sha256_file(cache_schema.manifest_path(base))
    result = {"passed":True,"cache":str(base),"frames":[1,2],"particle_counts":counts,
              "corner_origin":corner.tolist(),"manifest_sha256":digest,
              "blender":bpy.app.version_string,"device":runtime.cuda_devices[0].alias,
              "proof":"real ProductionBakeBackend, real CUDA, public operator, Blender OpenVDB reader"}
    (out / "report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("VDB-PRODUCTION-BAKE-ACCEPTANCE-PASS",flush=True)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root))
    from tools.repo_addon_loader import enable_repo_addon
    enable_repo_addon(str(root),default_set=True,persistent=True)
    with tempfile.TemporaryDirectory(prefix="fluidsim-vdb-production-") as directory:
        main(directory)
