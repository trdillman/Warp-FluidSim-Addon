"""Blender --background gate for cache sequence export and playback.

Run:
  blender --background --factory-startup --python tests/acceptance/blender_cache_export_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []

def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def surface_evidence(payload):
    verts = np.asarray(payload["surface_verts"])
    tris = np.asarray(payload["surface_tris"])
    valid = bool(
        verts.ndim == 2 and verts.shape[1:] == (3,)
        and tris.ndim == 2 and tris.shape[1:] == (3,)
        and np.issubdtype(verts.dtype, np.floating)
        and np.issubdtype(tris.dtype, np.integer)
        and len(verts) > 0 and len(tris) > 0
        and np.all(np.isfinite(verts))
        and int(np.min(tris)) >= 0 and int(np.max(tris)) < len(verts)
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(verts).tobytes())
    digest.update(np.ascontiguousarray(tris).tobytes())
    return {
        "valid": valid,
        "vertices": len(verts),
        "triangles": len(tris),
        # Diagnostic only: CUDA marching-cubes order/topology equality is not
        # an acceptance invariant.
        "diagnostic_hash": digest.hexdigest(),
    }


def payload_diagnostic_hash(payload):
    """Hash exact payload bytes for reporting, never for acceptance."""
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = np.ascontiguousarray(payload[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def point_count(obj):
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication.live_point_count(obj)


def point_sample(obj, limit=32):
    count = point_count(obj)
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32)
    n = min(count, limit)
    if obj.type == "POINTCLOUD":
        attr = obj.data.attributes.get("position")
        values = np.empty(len(attr.data) * 3, dtype=np.float32)
        attr.data.foreach_get("vector", values)
    else:
        values = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", values)
    # Live points occupy the head of the buffer; the tail is padding.
    return values.reshape(-1, 3)[:n].copy()


def main():
    try:
        enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
        check("addon enable", True)
    except Exception as exc:  # noqa: BLE001
        check("addon enable", False, repr(exc))
        return

    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.cache_enabled = False
    fs.cache_playback_enabled = False
    fs.cache_use_custom_range = True
    fs.cache_frame_start = 1
    fs.cache_frame_end = 3

    cache_dir = tempfile.mkdtemp(prefix="fluidsim_cache_pipeline_")
    fs.cache_path = cache_dir
    original_cache_write = None
    original_handle_frame_change = None
    original_update_outputs = None
    try:
        check("setup_scene", bpy.ops.fluidsim.setup_scene() == {"FINISHED"})
        from fluidSim_warp_blender51.blender import scene_bridge

        ds = scene_bridge.active_domain_settings(scene)
        check("active domain settings", ds is not None)
        assert ds is not None, (
            "cache export acceptance requires a designated domain")
        ds.voxel_size = 0.12
        ds.max_particles = 100_000
        ds.auto_grow_particles = False
        ds.max_whitewater = 5_000
        ds.whitewater_enabled = True
        ds.ww_churn_gain = 1_000.0
        ds.ww_churn_threshold = 0.0
        ds.ww_emission_rate = 2_000.0
        ds.ww_min_speed = 0.0
        ds.surface_enabled = True
        # Give the short three-frame fixture an explicit energetic continuous
        # inflow so authoritative whitewater coverage cannot pass vacuously.
        source = next(
            (
                obj for obj in scene.objects
                if getattr(getattr(obj, "fluidsim", None), "role", "NONE")
                == "SOURCE"
            ),
            None,
        )
        check("cache fixture source exists", source is not None)
        source_peer = None
        if source is not None:
            # Two deterministic one-shot source records with opposing velocity
            # create immediate churn without relying on a top-up race.
            source.fluidsim.emission_mode = "INITIAL_FILL"
            source.fluidsim.use_object_velocity = False
            source.fluidsim.initial_velocity = (4.0, 0.0, 1.0)
            source.location.x -= 0.5
            source_peer = source.copy()
            source_peer.data = source.data.copy()
            source_peer.name = "FluidSim Cache Whitewater Peer"
            source_peer["fluidsim_persistent_id"] = "cache-whitewater-peer"
            source_peer.location.x += 1.0
            source_peer.fluidsim.initial_velocity = (-4.0, 0.0, 1.0)
            scene.collection.objects.link(source_peer)

        def normalize_source_diagnostics():
            # Status mirrors are outputs, not canonical source inputs. Restore
            # their pre-bake values when reconstructing the same fixture.
            for item in (source, source_peer):
                if item is None:
                    continue
                item.fluidsim.source_status_code = ""
                item.fluidsim.source_status_message = ""
                item.fluidsim.source_status_emitted = 0
                item.fluidsim.source_status_capacity_limited = False

        normalize_source_diagnostics()
        from fluidSim_warp_blender51.blender import output_publication, playback
        from fluidSim_warp_blender51.io import cache

        # Poison the two preview paths at their real runtime seams. Production
        # Bake must suppress timeline interpretation and step without Output
        # Publication even though scene.frame_set() delivers Blender events.
        bypass_calls = {"frame_handler": 0, "output_publication": 0}
        original_handle_frame_change = playback.handle_frame_change
        original_update_outputs = output_publication.update_outputs

        def forbidden_frame_handler(*_args, **_kwargs):
            bypass_calls["frame_handler"] += 1
            raise AssertionError("Production Cache Bake entered Preview Playback")

        def forbidden_output_publication(*_args, **_kwargs):
            bypass_calls["output_publication"] += 1
            raise AssertionError("Production Cache Bake entered Output Publication")

        playback.handle_frame_change = forbidden_frame_handler
        output_publication.update_outputs = forbidden_output_publication

        # Observe the real production write seam.  This proves each on-disk
        # payload has the same full counts as the immutable simulation/output
        # snapshot, rather than merely proving that a self-consistent decimated
        # payload was written.
        write_observations = []
        active_bake = ["first"]
        original_cache_write = cache.write_frame_v3_atomic

        def observed_cache_write(base_dir, frame_payload, frame, **kwargs):
            solver_particles = len(frame_payload.positions)
            solver_whitewater = len(frame_payload.whitewater)
            result = original_cache_write(
                base_dir, frame_payload, frame, **kwargs)
            path = result[0]
            with np.load(path, allow_pickle=False) as data:
                payload = {name: data[name] for name in data.files}
            write_observations.append({
                "bake": active_bake[0],
                "frame": int(frame),
                "solver_frame": int(frame_payload.frame),
                "preview_policy": {
                    "fraction": float(fs.particle_display_fraction),
                    "scale": float(fs.particle_display_scale),
                    "show_particles": bool(fs.show_particles),
                    "show_surface": bool(fs.show_surface),
                    "show_whitewater": bool(fs.show_whitewater),
                    "particle_attribute": str(
                        fs.particle_attribute_mode),
                    "surface_attribute": str(
                        fs.surface_attribute_mode),
                    "surface_cadence": int(
                        ds.live_surface_publish_cadence),
                },
                "solver_particles": solver_particles,
                "solver_whitewater": solver_whitewater,
                "cache_counts": cache.authoritative_payload_counts(payload),
                "whitewater_required": int(payload["whitewater_required"]),
                "whitewater_available": int(payload["whitewater_available"]),
                "whitewater_admitted": int(payload["whitewater_admitted"]),
                "whitewater_dropped": int(payload["whitewater_dropped"]),
                "whitewater_policy": str(payload["whitewater_policy"]),
                "whitewater_status": str(payload["whitewater_status"]),
            })
            return result

        cache.write_frame_v3_atomic = observed_cache_write
        check(
            "export cache sequence",
            bpy.ops.fluidsim.export_cache_sequence() == {"FINISHED"},
            fs.status,
        )
        first_bake_bypass_calls = dict(bypass_calls)
        playback.handle_frame_change = original_handle_frame_change
        output_publication.update_outputs = original_update_outputs

        metadata_path = os.path.join(cache_dir, "fluidsim_cache.json")
        check("cache-v3 manifest exists", os.path.exists(metadata_path), metadata_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            check(
                "cache-v3 manifest range",
                (
                    metadata.get("format_version") == 3
                    and metadata.get("frame_start") == 1
                    and metadata.get("frame_end") == 3
                    and sorted(int(frame) for frame in metadata.get("frames", {}))
                    == [1, 2, 3]
                ),
                str(metadata),
            )
            stamps = [
                metadata["frames"][str(frame)].get("source_frame_stamp", "")
                for frame in (1, 2, 3)
            ]
            check(
                "every committed frame records a source stamp",
                all(len(stamp) == 64 for stamp in stamps),
                str(stamps),
            )

        frame_paths = [
            os.path.join(cache_dir, "frames", f"frame_{frame:06d}.npz")
            for frame in range(1, 4)
        ]
        check(
            "cache frame files exist",
            all(os.path.exists(path) for path in frame_paths),
            ", ".join(os.path.basename(path) for path in frame_paths),
        )

        frame_data = []
        first_payload_hashes = []
        first_payload_schemas = []
        first_surfaces = []
        for frame, path in enumerate(frame_paths, start=1):
            if not os.path.exists(path):
                continue
            with np.load(path, allow_pickle=False) as data:
                positions = data["positions"]
                surface_verts = data["surface_verts"]
                payload = {name: data[name] for name in data.files}
                first_payload_hashes.append(payload_diagnostic_hash(payload))
                first_payload_schemas.append(tuple(sorted(data.files)))
                first_surfaces.append(surface_evidence(payload))
                frame_data.append(
                    {
                        "frame": int(np.asarray(data["frame"]).item()),
                        "particles": len(positions),
                        "surface_verts": len(surface_verts),
                        "position_sample": positions[:32].copy(),
                    }
                )
                check(
                    f"frame {frame} snapshot payload",
                    (
                        int(np.asarray(data["format_version"]).item()) == 3
                        and int(np.asarray(data["frame"]).item()) == frame
                        and int(np.asarray(data["solver_frame"]).item()) == frame
                        and len(positions) > 1_000
                        and len(data["velocities"]) == len(positions)
                        and len(data["whitewater_velocities"])
                        == len(data["whitewater"])
                        and len(data["whitewater_age"]) == len(data["whitewater"])
                        and len(data["whitewater_life"]) == len(data["whitewater"])
                        and "voxel_size" in data.files
                    ),
                    f"{len(positions)} particles, {len(surface_verts)} verts",
                )

        if len(frame_data) >= 2:
            changed = any(
                not np.allclose(frame_data[0]["position_sample"], item["position_sample"])
                for item in frame_data[1:]
                if item["position_sample"].shape == frame_data[0]["position_sample"].shape
            )
            check("cache frames differ over time", changed)

        first_metadata = cache.read_sequence_metadata(cache_dir)
        # These are Output Publication policy only. Presentation remains on the
        # scene while live surface cadence is owned by the designated domain.
        # A deterministic Production Cache Bake must neither consume them nor
        # acquire published outputs.
        fs.particle_display_fraction = 0.25
        fs.particle_display_scale = 2.5
        fs.show_particles = False
        fs.show_surface = False
        fs.show_whitewater = True
        fs.particle_attribute_mode = "HEIGHT"
        fs.surface_attribute_mode = "SPEED"
        ds.live_surface_publish_cadence = 7
        # Production Bake restores the pre-bake session state. Since this
        # fixture starts inactive, it must remain inactive between exports.
        from fluidSim_warp_blender51.blender.scene_bridge import session

        normalize_source_diagnostics()
        playback.handle_frame_change = original_handle_frame_change
        output_publication.update_outputs = original_update_outputs
        check("production bake restores inactive solver state",
              not session().active)
        bypass_calls.update(frame_handler=0, output_publication=0)
        playback.handle_frame_change = forbidden_frame_handler
        output_publication.update_outputs = forbidden_output_publication
        active_bake[0] = "second"
        from fluidSim_warp_blender51.blender import cache_bake

        second_bake = cache_bake.run_synchronous(
            scene, session(), 1, 3, restart=True)
        check(
            "explicitly restart export with changed preview policy",
            second_bake.status.value == "COMPLETE",
            second_bake.message,
        )
        second_payload_hashes = []
        second_payload_schemas = []
        second_surfaces = []
        for path in frame_paths:
            if not os.path.exists(path):
                continue
            with np.load(path, allow_pickle=False) as data:
                payload = {name: data[name] for name in data.files}
                second_payload_hashes.append(payload_diagnostic_hash(payload))
                second_payload_schemas.append(tuple(sorted(data.files)))
                second_surfaces.append(surface_evidence(payload))
        second_metadata = cache.read_sequence_metadata(cache_dir)

        check(
            "exact payload hashes recorded as diagnostics only",
            (len(first_payload_hashes) == len(second_payload_hashes) == 3
             and all(len(value) == 64
                     for value in first_payload_hashes + second_payload_hashes)),
            "no cross-bake hash or numerical equality is an acceptance condition; "
            f"first={first_payload_hashes}, second={second_payload_hashes}",
        )

        required_schema = {
            "format_version", "settings_signature", "scene_key", "positions",
            "velocities", "whitewater", "surface_verts", "surface_tris", "frame",
            "solver_frame", "simulation_time", "source_signature",
            "voxel_size", "surface_voxel_size", "surface_resolution_multiplier",
            "surface_resolution_mode", "surface_update_cadence",
            "surface_smooth_iterations", "particle_age", "particle_life",
            "particle_temperature", "particle_viscosity", "particle_density",
            "particle_time_offset", "whitewater_velocities", "whitewater_age",
            "whitewater_life", "whitewater_required", "whitewater_available",
            "whitewater_admitted", "whitewater_dropped", "whitewater_policy",
            "whitewater_status",
            "attribute_schema_version", "attribute_inventory_json",
            "surface_attribute_transfer_method",
            "source_frame_stamp", "liquid_sdf_sparse", "liquid_sdf_leaf_coords",
            "liquid_sdf_voxel_size",
        }
        schema_valid = bool(
            len(first_payload_schemas) == len(second_payload_schemas) == 3
            and all(required_schema <= set(schema)
                    for schema in first_payload_schemas + second_payload_schemas)
            and len(set(first_payload_schemas + second_payload_schemas)) == 1)
        check(
            "production payload schemas independently valid",
            schema_valid,
            f"required={sorted(required_schema)}, first={first_payload_schemas}, "
            f"second={second_payload_schemas}",
        )

        first_write = [item for item in write_observations
                       if item["bake"] == "first"]
        second_write = [item for item in write_observations
                        if item["bake"] == "second"]

        def counts_match_solver(observation):
            counts = observation["cache_counts"]
            return bool(
                set(counts["particles"].values())
                == {observation["solver_particles"]}
                and set(counts["whitewater"].values())
                == {observation["solver_whitewater"]})

        count_evidence = {
            "first": first_write,
            "second": second_write,
        }
        check(
            "each bake writes full solver particle/whitewater attribute counts",
            (len(first_write) == len(second_write) == 3
             and all(counts_match_solver(item)
                     for item in first_write + second_write)),
            json.dumps(count_evidence, sort_keys=True),
        )

        def whitewater_admission_is_coherent(observation):
            required = observation["whitewater_required"]
            available = observation["whitewater_available"]
            admitted = observation["whitewater_admitted"]
            dropped = observation["whitewater_dropped"]
            status = observation["whitewater_status"]
            return bool(
                observation["whitewater_policy"]
                == "FIXED_CAPACITY_DETERMINISTIC_DROPS"
                and min(required, available, admitted, dropped) >= 0
                and required == admitted + dropped
                and admitted <= available
                and ((dropped == 0 and status == "OK")
                     or (dropped > 0 and status == "CAPACITY_EXHAUSTED"))
            )

        check(
            "every production write records coherent whitewater admission state",
            all(whitewater_admission_is_coherent(item)
                for item in first_write + second_write),
            json.dumps(count_evidence, sort_keys=True),
        )
        sequential_control_flow = bool(
            [item["frame"] for item in first_write] == [1, 2, 3]
            and [item["solver_frame"] for item in first_write] == [1, 2, 3]
            and [item["frame"] for item in second_write] == [1, 2, 3]
            and [item["solver_frame"] for item in second_write] == [1, 2, 3])
        preview_varied = bool(
            first_write and second_write
            and {item["preview_policy"]["fraction"] for item in first_write}
            == {1.0}
            and {item["preview_policy"]["fraction"] for item in second_write}
            == {0.25}
            and first_write[0]["preview_policy"]
            != second_write[0]["preview_policy"])
        check(
            "control flow is sequential under varied preview settings",
            sequential_control_flow and preview_varied,
            json.dumps(count_evidence, sort_keys=True),
        )
        second_bake_bypass_calls = dict(bypass_calls)
        check(
            "Production Cache Bake bypasses handlers and Output Publication",
            (first_bake_bypass_calls
             == {"frame_handler": 0, "output_publication": 0}
             and second_bake_bypass_calls
             == {"frame_handler": 0, "output_publication": 0}),
            f"first={first_bake_bypass_calls}, second={second_bake_bypass_calls}",
        )

        surfaces_valid = bool(
            len(first_surfaces) == len(second_surfaces) == 3
            and all(item["valid"] for item in first_surfaces + second_surfaces))
        same_surface_counts = [
            (item["vertices"], item["triangles"]) for item in first_surfaces
        ] == [
            (item["vertices"], item["triangles"]) for item in second_surfaces
        ]
        same_surface_topology_bytes = [
            item["diagnostic_hash"] for item in first_surfaces
        ] == [
            item["diagnostic_hash"] for item in second_surfaces
        ]
        check(
            "derived surfaces independently structurally valid",
            surfaces_valid,
            f"first={first_surfaces}, second={second_surfaces}",
        )
        check(
            "CUDA derived surface nondeterminism reported without equality claim",
            len(first_surfaces) == len(second_surfaces) == 3,
            f"count_equality_observed={same_surface_counts}, "
            f"topology_byte_equality_observed={same_surface_topology_bytes}; "
            f"first={first_surfaces}, second={second_surfaces}",
        )
        check(
            "production metadata independently records the sequential range",
            all(
                (metadata or {}).get("source") == "production_bake"
                and (metadata or {}).get("format_version") == 3
                and (metadata or {}).get("frame_start") == 1
                and (metadata or {}).get("frame_end") == 3
                and (metadata or {}).get("frame_count") == 3
                for metadata in (first_metadata, second_metadata)),
            f"first={first_metadata}, second={second_metadata}",
        )

        # Restore poisoned preview seams before cache-playback assertions.
        playback.handle_frame_change = original_handle_frame_change
        output_publication.update_outputs = original_update_outputs
        original_handle_frame_change = None
        original_update_outputs = None

        # Restore presentation for the cache-playback output assertions below.
        fs.particle_display_fraction = 1.0
        fs.particle_display_scale = 1.0
        fs.show_particles = True
        fs.show_surface = True
        fs.show_whitewater = False
        fs.particle_attribute_mode = "NONE"
        fs.surface_attribute_mode = "NONE"
        ds.live_surface_publish_cadence = 1

        from fluidSim_warp_blender51.blender.scene_bridge import session

        normalize_source_diagnostics()
        if session().active:
            bpy.ops.fluidsim.release()
        check("solver released before cache playback", not session().active)
        check(
            "arm cache playback through the owned workflow",
            bpy.ops.fluidsim.cache_playback_start() == {"FINISHED"},
            fs.status,
        )

        session().last_frame = -1
        scene.frame_set(1)
        parts_1 = bpy.data.objects.get("FluidSim Particles")
        sample_1 = point_sample(parts_1)
        count_1 = point_count(parts_1)
        check("cache playback frame 1 output", count_1 > 1_000, f"{count_1} points")
        check("cache playback does not require live solver", not session().active)

        scene.frame_set(3)
        parts_3 = bpy.data.objects.get("FluidSim Particles")
        sample_3 = point_sample(parts_3)
        count_3 = point_count(parts_3)
        check("cache playback frame 3 output", count_3 > 1_000, f"{count_3} points")
        if sample_1.shape == sample_3.shape and len(sample_1):
            check("cache playback changes visible output", not np.allclose(sample_1, sample_3))
        else:
            check("cache playback changes visible output", count_1 != count_3)
        check("cache playback status", fs.status.startswith("Cache frame 3"), fs.status)
    finally:
        if original_cache_write is not None:
            cache.write_frame_v3_atomic = original_cache_write
        if original_handle_frame_change is not None:
            playback.handle_frame_change = original_handle_frame_change
        if original_update_outputs is not None:
            output_publication.update_outputs = original_update_outputs
        shutil.rmtree(cache_dir, ignore_errors=True)


main()
print()
if FAILURES:
    print(f"CACHE-EXPORT-PIPELINE-FAIL ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("CACHE-EXPORT-PIPELINE-PASS")
sys.exit(0)
