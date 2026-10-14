"""Actual VDB operator -> committed cache -> independent Blender OpenVDB reader.

The moving sphere is a synthetic cache fixture, not a production solver bake.
The public exporter itself is not mocked. Run through the isolated gate.
"""
from pathlib import Path
import json
import sys

import bpy
import numpy as np


def main(output_dir):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from tests.cache_io.fixtures.vdb_cache import build_sdf_cache
    from fluidSim_warp_blender51.blender import interchange
    from fluidSim_warp_blender51.blender.menus import create_domain_object
    from fluidSim_warp_blender51.blender.solver_settings import build_settings
    from fluidSim_warp_blender51.io import cache_schema

    work = Path(output_dir)
    work.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name, condition, detail=None):
        checks.append({"name":name,"passed":bool(condition),"detail":detail})
        print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}",flush=True)

    scene = bpy.data.scenes.new("VDBExportAcceptance")
    bpy.context.window.scene = scene
    scene.fluidsim.live = False
    domain = create_domain_object(scene)
    domain.location = (-0.4,0.2,0.6)
    domain.fluidsim_domain.resolution_divisions = 80
    domain.fluidsim_domain.export_attributes = False
    domain.fluidsim_domain.narrow_band_enabled = True
    bpy.context.view_layer.update()
    settings = build_settings(scene)
    base, _ = build_sdf_cache(work / "cache",settings)
    scene.fluidsim.cache_path = str(base)
    scene.frame_set(7)
    before = (scene.frame_current,scene.fluidsim.live,scene.fluidsim.cache_playback_enabled,
              tuple(o.name for o in bpy.context.selected_objects))
    cache_digest = cache_schema.sha256_file(cache_schema.manifest_path(base))
    result = bpy.ops.fluidsim.export_vdb(directory=str(work),sequence_name="vdb_sequence")
    check("public exporter finishes", "FINISHED" in result, sorted(result))
    paths = sorted((work / "vdb_sequence" / "frames").glob("*.vdb"))
    check("two numbered cache frames exported", len(paths) == 2,[p.name for p in paths])
    after = (scene.frame_current,scene.fluidsim.live,scene.fluidsim.cache_playback_enabled,
             tuple(o.name for o in bpy.context.selected_objects))
    check("timeline, playback mode and selection unchanged", before == after)
    check("baked cache unchanged", cache_schema.sha256_file(cache_schema.manifest_path(base)) == cache_digest)
    for frame,path in enumerate(paths,1):
        volume = bpy.data.volumes.new(f"VDB{frame}")
        volume.filepath = str(path)
        volume.grids.load()
        check(f"frame {frame} independent OpenVDB reader", volume.grids.is_loaded and "surface" in volume.grids,
              volume.grids.error_message)
        obj = bpy.data.objects.new(f"Volume{frame}",volume)
        scene.collection.objects.link(obj)
        mesh_obj = bpy.data.objects.new(f"Mesh{frame}",bpy.data.meshes.new(f"mesh{frame}"))
        scene.collection.objects.link(mesh_obj)
        modifier = mesh_obj.modifiers.new("ReadBack","VOLUME_TO_MESH")
        modifier.object = obj
        modifier.grid_name = "surface"
        modifier.threshold = 0.0
        modifier.resolution_mode = "GRID"
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated = mesh_obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            count = len(mesh.vertices)
            check(f"frame {frame} surface meshes", count > 0,count)
            if count:
                coordinates = np.empty(count*3,dtype=np.float32)
                mesh.vertices.foreach_get("co",coordinates)
                coordinates = coordinates.reshape(-1,3)
                center = np.array([0.1*frame,0.3,0.1])
                error = float(max(np.abs(coordinates.min(0)-(center-0.3)).max(),
                                  np.abs(coordinates.max(0)-(center+0.3)).max()))
                check(f"frame {frame} world-space bounds", error <= settings.voxel_size,
                      {"error":error,"voxel_size":settings.voxel_size})
        finally:
            evaluated.to_mesh_clear()
    domain.location.x += 10.0
    bpy.context.view_layer.update()
    moved_plan = interchange.plan_vdb_export(scene)
    check("changed scene does not override baked origin", moved_plan.origin is None)
    legacy, _ = build_sdf_cache(work / "legacy-cache", settings, include_origin=False)
    scene.fluidsim.cache_path = str(legacy)
    refused = False
    try:
        interchange.plan_vdb_export(scene)
    except interchange.ExportRefused:
        refused = True
    check("old origin-less cache refuses a moved domain",refused)
    result = {"passed":all(c["passed"] for c in checks),"checks":checks,
              "fixture":"synthetic committed cache; actual exporter and Blender reader",
              "blender":bpy.app.version_string,"warp_imported":"warp" in sys.modules}
    (work / "report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("CACHE-VDB-EXPORT-ACCEPTANCE-PASS" if result["passed"] else "CACHE-VDB-EXPORT-ACCEPTANCE-FAIL",flush=True)
    return result


if __name__ == "__main__":
    import tempfile
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root))
    from tools.repo_addon_loader import enable_repo_addon
    enable_repo_addon(str(root),default_set=True,persistent=True)
    with tempfile.TemporaryDirectory(prefix="fluidsim-vdb-acceptance-") as directory:
        if not main(directory)["passed"]:
            raise RuntimeError("Cache VDB export acceptance failed")
