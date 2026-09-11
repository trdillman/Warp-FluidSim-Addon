"""Open generated demo .blend files in Blender and verify saved scene contents.

Run with Blender:
    blender --background --python tools/check_demo_blends.py
"""

import json
import os
import sys

import bpy
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon

DEMOS = os.path.join(REPO, "demos")
SCENES = (
    "dam_break", "crown_splash", "chromatic_vortex", "force_fields",
    "custom_effects_vortex",
    "adhesive_calender",
    "cache_export_playback_review", "paddle_stir",
    "mesh_solid_source", "mesh_shell_source", "point_source",
    "volume_source", "liquid_sink", "height_fill",
)
EXPECTED_AUTO_SEEDS = {
    "crown_splash": {"Crown Pool Seed", "Crown Droplet Seed"},
    "force_fields": {"Force Fields Particle Seed"},
    "adhesive_calender": {
        "Low Viscosity Adhesive Seed", "High Viscosity Adhesive Seed"},
    "paddle_stir": {"Paddle Stir Pool Seed"},
}

# An enabled effect whose own coefficients zero it out is a UI-only claim
# (DEMO_SCENE_CONTRACT.md, "Enabled effects are reachable"). Each entry is a
# necessary condition for the effect to change the solve at all.
EFFECT_PRECONDITIONS = {
    "whitewater_enabled": (
        ("ww_churn_gain > 0", lambda s: s.ww_churn_gain > 0.0),
        ("ww_emission_rate > 0", lambda s: s.ww_emission_rate > 0.0),
    ),
    "viscosity_enabled": (
        ("viscosity_from_particles or viscosity_constant > 0",
         lambda s: s.viscosity_from_particles or s.viscosity_constant > 0.0),
    ),
    "surface_tension_enabled": (
        ("surface_tension_coefficient > 0",
         lambda s: s.surface_tension_coefficient > 0.0),
    ),
    "species_enabled": (
        ("species_mixing_rate > 0", lambda s: s.species_mixing_rate > 0.0),
    ),
}
# Whitewater also has a runtime-only gate: a candidate emits only where churn
# reaches ww_churn_threshold, and churn depends on the flow. No static bound
# exists, so every whitewater demo declares the timeline window (frames after
# frame_start) within which live whitewater must appear on a fresh reopen.
# Each value is the measured first-emission frame plus margin.
WHITEWATER_ACTIVATION_FRAMES = {
    "dam_break": 32,          # first live whitewater at frame 16
    "paddle_stir": 32,        # frame 15 (needs the keyframed paddle moving)
    "crown_splash": 12,       # frame 3 on reopen (4-5 in the build run)
    "adhesive_calender": 250,  # its whole simulated timeline
}


def open_demo_blend(path):
    """Open ``path`` as an add-on session loads it, load migration included.

    The add-on is enabled only after the file is open, so its ``load_post``
    domain migration never fires here. A pre-split .blend (``schema_version``
    0) keeps its authored domain settings on the legacy scene group until that
    migration moves them onto the domain object, so run it explicitly. The
    checker never saves, so the demo on disk is untouched.
    """
    bpy.ops.wm.open_mainfile(filepath=path)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    # Import only after enable_repo_addon, which re-imports the add-on.
    from fluidSim_warp_blender51.blender import migration

    migration._on_load_post(None)


def effect_precondition_failures(settings):
    """Map each enabled effect to the preconditions its own settings violate."""
    failures = {}
    for flag, preconditions in EFFECT_PRECONDITIONS.items():
        if getattr(settings, flag, False):
            failed = [label for label, holds in preconditions if not holds(settings)]
            if failed:
                failures[flag] = failed
    return failures


def check_whitewater_reachable(name):
    """Reopen ``name`` and prove its enabled whitewater emits during playback.

    Drives the saved live timeline through ``scene.frame_set`` -- the
    production frame handler, with keyframed colliders evaluated -- and reads
    the live count from ``frame_stats``. The ``FluidSim Whitewater`` point
    cloud keeps a spare-capacity floor, so its point count proves nothing.
    """
    open_demo_blend(os.path.join(DEMOS, name + ".blend"))
    # Import only after enable_repo_addon: it re-imports the add-on, so an
    # earlier module reference would hold a dead session the handler never uses.
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    settings = scene_bridge.active_domain_settings(scene)
    window = WHITEWATER_ACTIVATION_FRAMES.get(name)
    report = {
        "window_frames": window,
        "churn_threshold": float(settings.ww_churn_threshold),
        "emission_rate": float(settings.ww_emission_rate),
        "peak_churn": 0.0,
        "first_live_frame": None,
    }
    if window is None:
        report["error"] = ("whitewater_enabled but no WHITEWATER_ACTIVATION_FRAMES "
                           "entry declares when it must emit")
        return False, report
    if not scene.fluidsim.live:
        report["error"] = "saved scene is not in Live mode"
        return False, report
    applied_ok = False
    try:
        for frame in range(scene.frame_start + 1, scene.frame_start + window + 1):
            scene.frame_set(frame)
            sess = scene_bridge.session()
            if not (sess.active and sess.frame_stats):
                report["error"] = f"solver inactive at frame {frame}"
                break
            report["peak_churn"] = max(
                report["peak_churn"], float(sess.solver.churn.data.numpy().max()))
            if int(sess.frame_stats[-1].get("whitewaterParticles") or 0) > 0:
                report["first_live_frame"] = frame
                break
        sess = scene_bridge.session()
        if sess.active:
            applied = sess.solver.settings
            applied_ok = bool(
                applied.whitewater_enabled
                and abs(applied.ww_churn_threshold - settings.ww_churn_threshold) < 1.0e-6
                and abs(applied.ww_emission_rate - settings.ww_emission_rate) < 1.0e-6
                and abs(applied.churn_gain - settings.ww_churn_gain) < 1.0e-6)
            report["applied_matches_authored"] = applied_ok
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        scene.fluidsim.live = False
        bpy.ops.fluidsim.release()
    return applied_ok and report["first_live_frame"] is not None, report


def point_count(obj):
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication.live_point_count(obj)


def output_snapshot(limit=32):
    """Return a bounded sample of whichever owned viewport outputs exist."""
    snapshot = []
    visible_count = 0
    for name in ("FluidSim Particles", "FluidSim Whitewater", "FluidSim Surface"):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        if obj.type == "POINTCLOUD":
            count = point_count(obj)
            position = obj.data.attributes.get("position")
            values = (() if position is None else tuple(
                round(float(value), 6)
                for item in position.data[:min(limit, count)]
                for value in item.vector
            ))
        elif obj.type == "MESH":
            count = len(obj.data.vertices)
            values = tuple(
                round(float(value), 6)
                for vertex in obj.data.vertices[:min(limit, count)]
                for value in vertex.co
            )
        else:
            continue
        visible_count += count
        snapshot.append((name, count, values))
    return visible_count, tuple(snapshot)


def check_playback_contract(scene, fs, *, cache_mode=False):
    """Exercise the saved timeline contract through real frame handlers."""
    result = {"ready": False, "advances": False}
    if fs is None:
        return result
    expected_mode = bool(fs.live and bool(fs.cache_playback_enabled) == cache_mode)
    result["ready"] = bool(
        expected_mode
        and scene.frame_current == scene.frame_start
        and scene.get("fluidsim_playback_ready") is True)
    if not result["ready"]:
        return result

    from fluidSim_warp_blender51.blender.scene_bridge import session

    _before_count, before = output_snapshot()
    try:
        scene.frame_set(scene.frame_start + 1)
        middle_count, middle = output_snapshot()
        scene.frame_set(scene.frame_start + 2)
        after_count, after = output_snapshot()
        sess = session()
        if cache_mode:
            state_valid = not sess.active
        else:
            state_valid = bool(
                sess.active
                and sess.solver.particles.refresh_host_count() > 0)
        result["advances"] = bool(
            state_valid
            and max(middle_count, after_count) > 0
            and (middle != before or after != middle))
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def check_scene(name):
    path = os.path.join(DEMOS, name + ".blend")
    result = {
        "scene": name,
        "path": path,
        "exists": os.path.exists(path),
        "opened": False,
        "checks": {},
    }
    if not result["exists"]:
        return result

    open_demo_blend(path)
    scene = bpy.context.scene
    saved_frame = scene.frame_current
    result["opened"] = True

    fs = getattr(scene, "fluidsim", None)
    # Domain settings live on the designated domain object, never on
    # scene.fluidsim (FLUIDSIM_PG_scene declares none of them).
    from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings

    ds = active_domain_settings(scene)
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")

    checks = {
        "fluidsim_settings": fs is not None,
        "domain_object": bool(fs and fs.domain_object),
        "camera": bool(scene.camera),
        "valid_domain_boundary_policy": bool(
            ds and (ds.open_boundaries or ds.domain_sdf_walls)),
    }
    actual_auto_seeds = {
        obj.name for obj in scene.objects
        if getattr(getattr(obj, "fluidsim", None), "auto_inject_seed", False)
    }
    checks["owned_auto_seeds"] = (
        actual_auto_seeds == EXPECTED_AUTO_SEEDS.get(name, set()))
    if not checks["owned_auto_seeds"]:
        result["actual_auto_seeds"] = sorted(actual_auto_seeds)
    playback = check_playback_contract(scene, fs, cache_mode=False)
    checks["playback_ready"] = playback["ready"]
    checks["playback_advances"] = playback["advances"]
    if playback.get("error"):
        result["playback_contract_error"] = playback["error"]
    if name == "dam_break":
        domain = bpy.data.objects.get("DamBreak Domain")
        source = bpy.data.objects.get("DamBreak Source Column")
        collider = bpy.data.objects.get("Sphere")
        near = lambda actual, expected: all(
            abs(float(a) - float(e)) < 1.0e-4
            for a, e in zip(actual, expected)
        )
        checks.update({
            "canonical_domain_scale": bool(
                domain and near(domain.dimensions, (5.6, 2.00292993, 1.6))),
            "canonical_source_transform": bool(
                source
                and near(source.location, (0.3, 0.3, 0.6))
                and near(source.dimensions, (0.6, 0.6, 1.2))
                and source.fluidsim.emission_mode == "INITIAL_FILL"
                and source.fluidsim.source_region_kind == "ANALYTIC"),
            "sphere_collider": bool(
                collider and collider.fluidsim.role == "COLLIDER"
                and near(collider.location, (2.41923952, 0.0, 0.0))),
            "canonical_friction": bool(
                collider
                and abs(float(collider.fluidsim.collider_friction) - 1.0) < 1.0e-6),
            "no_redundant_empties": not any(
                obj.type == "EMPTY" for obj in scene.objects),
        })
    elif name == "crown_splash":
        pool_seed = bpy.data.objects.get("Crown Pool Seed")
        drop_seed = bpy.data.objects.get("Crown Droplet Seed")
        seed_contract = bool(
            pool_seed and drop_seed
            and pool_seed.type == drop_seed.type == "MESH"
            and pool_seed.fluidsim.enabled
            and drop_seed.fluidsim.enabled
            and pool_seed.fluidsim.auto_inject_seed
            and drop_seed.fluidsim.auto_inject_seed
            and abs(float(pool_seed.fluidsim.initial_velocity.z)) < 1.0e-6
            and abs(float(drop_seed.fluidsim.initial_velocity.z) + 3.5) < 1.0e-6
        )
        checks.update({
            "restartable_seed_contract": seed_contract,
            "live_playback_saved": bool(
                fs and fs.live and saved_frame == 1
                and scene.get("fluidsim_playback_ready") is True),
        })
        if seed_contract and checks["live_playback_saved"]:
            try:
                from fluidSim_warp_blender51.blender.scene_bridge import session

                scene.frame_set(2)
                sess = session()
                before = sess.solver.particles.position[:1024].numpy().copy()
                scene.frame_set(3)
                after = sess.solver.particles.position[:1024].numpy()
                delta = float(np.max(np.linalg.norm(after - before, axis=1)))
                checks["viewport_playback_advances"] = bool(
                    sess.active
                    and sess.solver.particles.refresh_host_count() > 0
                    and point_count(bpy.data.objects.get("FluidSim Particles")) > 0
                    and delta > 1.0e-6)
                result["playback_position_delta"] = delta
            except Exception as exc:  # noqa: BLE001
                checks["viewport_playback_advances"] = False
                result["playback_error"] = repr(exc)
            finally:
                fs.live = False
                bpy.ops.fluidsim.release()
        else:
            checks["viewport_playback_advances"] = False
    elif name == "chromatic_vortex":
        red_source = bpy.data.objects.get("Red Dye Initial Fill")
        blue_source = bpy.data.objects.get("Blue Dye Initial Fill")
        vortex = bpy.data.objects.get("Chromatic Mixing Vortex Force")
        live_surface = bpy.data.objects.get("FluidSim Surface")
        color_attr = (
            live_surface.data.color_attributes.get("display_color")
            if live_surface and live_surface.type == "MESH" else None)
        live_colors = np.asarray(
            [item.color[:3] for item in color_attr.data], dtype=np.float32
        ) if color_attr and len(color_attr.data) else np.empty((0, 3))
        checks.update({
            "color_mixing_enabled": bool(
                ds and ds.species_enabled and ds.species_mixing_rate > 0.0),
            "two_colored_initial_fills": bool(
                red_source and blue_source
                and red_source.fluidsim.role == "SOURCE"
                and blue_source.fluidsim.role == "SOURCE"
                and red_source.fluidsim.emission_mode == "INITIAL_FILL"
                and blue_source.fluidsim.emission_mode == "INITIAL_FILL"
                and red_source.fluidsim.source_color[0]
                > red_source.fluidsim.source_color[2]
                and blue_source.fluidsim.source_color[2]
                > blue_source.fluidsim.source_color[0]),
            "owned_spin_force": bool(
                vortex and vortex.fluidsim.role == "FORCE"
                and vortex.fluidsim.force_type == "SPIN"
                and vortex.fluidsim.force_strength > 0.0),
            "live_two_color_output": bool(
                len(live_colors)
                and np.any(live_colors[:, 0] > live_colors[:, 2] + 0.15)
                and np.any(live_colors[:, 2] > live_colors[:, 0] + 0.15)),
        })
    elif name == "custom_effects_vortex":
        domain = bpy.data.objects.get("Custom Effects Vortex Domain")
        source = bpy.data.objects.get("Custom Effects Vortex Initial Fill")
        controller = bpy.data.objects.get("Custom Effects Vortex Controller")
        stack = getattr(scene, "fluidsim_effects", None)
        stack_items = list(stack.items) if stack is not None else []
        checks.update({
            "custom_effect_demo_identity": (
                scene.get("fluidsim_custom_effect_demo") == "vortex_force"),
            "custom_effect_domain": bool(
                domain and fs and fs.domain_object == domain),
            "custom_effect_source": bool(
                source
                and source.fluidsim.role == "SOURCE"
                and source.fluidsim.enabled
                and source.fluidsim.emission_mode == "INITIAL_FILL"),
            "custom_effect_controller": bool(
                controller
                and controller.get("fluidsim_effect_controller") is True),
            "custom_effect_stack": bool(
                len(stack_items) == 1
                and stack_items[0].effect_id == "fluidsim.vortex_force"),
            "custom_effect_applied": bool(
                stack and stack.status == "APPLIED"
                and stack_items and stack_items[0].status == "APPLIED"),
        })
        try:
            from fluidSim_warp_blender51.blender.scene_bridge import session

            sess = session()
            effect_stats = list(sess.solver.effects.stats.values())
            checks["custom_effect_executed"] = any(
                item.effect_id == "fluidsim.vortex_force" and item.calls > 0
                for item in effect_stats
            )
            result["custom_effect_calls"] = {
                item.effect_id: int(item.calls) for item in effect_stats
            }
            runtime_entries = [
                entry
                for entries in sess.solver.effects.by_stage.values()
                for entry in entries
                if entry.instance.definition.effect_id == "fluidsim.vortex_force"
            ]
            entry_before = runtime_entries[0]
            compiled_kernel = entry_before.compiled_kernel
            solver_identity = id(sess.solver)
            calls_before = sum(item.calls for item in effect_stats)
            suction = next(
                parameter for parameter in stack_items[0].parameters
                if parameter.key == "suction"
            )
            suction.float_value = 48.54
            runtime_entries = [
                entry
                for entries in sess.solver.effects.by_stage.values()
                for entry in entries
                if entry.instance.definition.effect_id == "fluidsim.vortex_force"
            ]
            entry_after = runtime_entries[0]
            runtime_suction = float(
                entry_after.instance.effective_parameter_values()["suction"]
            )
            checks["custom_effect_parameter_hot_update"] = bool(
                abs(runtime_suction - 48.54) < 1.0e-4
                and stack.status == "APPLIED"
                and stack_items[0].status == "APPLIED"
                and not stack.blocking_reason
                and id(sess.solver) == solver_identity
                and entry_after.compiled_kernel is compiled_kernel
            )
            scene.frame_set(scene.frame_current + 1)
            calls_after = sum(
                item.calls for item in sess.solver.effects.stats.values()
            )
            checks["custom_effect_hot_update_executes"] = calls_after > calls_before
            result["custom_effect_runtime_suction"] = runtime_suction
        except Exception as exc:  # noqa: BLE001
            checks["custom_effect_executed"] = False
            checks["custom_effect_parameter_hot_update"] = False
            checks["custom_effect_hot_update_executes"] = False
            result["custom_effect_error"] = repr(exc)
    elif name == "force_fields":
        force_fields = [
            obj for obj in scene.objects
            if getattr(getattr(obj, "field", None), "type", "NONE") != "NONE"
        ]
        expected_types = {
            "FORCE", "WIND", "VORTEX", "MAGNET", "HARMONIC", "CHARGE",
            "LENNARDJ", "TEXTURE", "GUIDE", "BOID", "TURBULENCE",
            "DRAG", "FLUID_FLOW",
        }
        texture = next(
            (obj for obj in force_fields if obj.field.type == "TEXTURE"), None)
        guide = next(
            (obj for obj in force_fields if obj.field.type == "GUIDE"), None)
        checks.update({
            "all_13_force_types": (
                len(force_fields) == 13
                and {obj.field.type for obj in force_fields} == expected_types),
            "force_proxy_metadata": (
                scene.get("fluidsim_force_field_types") == 13
                and bpy.data.texts.get("README_Force_Fields") is not None),
            "texture_proxy_configured": bool(
                texture and texture.field.texture
                and texture.field.texture_mode == "CURL"),
            "curve_guide_has_controls": bool(
                guide and guide.type == "CURVE" and guide.data.splines
                and len(guide.data.splines[0].points) == 4),
            "saved_live_frame": bool(
                saved_frame == 1
                and scene.get("fluidsim_playback_ready") is True),
        })
    elif name in {
            "mesh_solid_source", "mesh_shell_source", "point_source",
            "volume_source", "liquid_sink", "height_fill"}:
        checks["phase3_workflow_metadata"] = scene.get("fluidsim_phase3_workflow") == name
        if name == "mesh_solid_source":
            obj = bpy.data.objects.get("Watertight Mesh Solid Emitter")
            checks["real_mesh_solid_emitter"] = bool(
                obj and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "MESH_SOLID"
                and obj.fluidsim.emission_mode == "INITIAL_FILL")
        elif name == "mesh_shell_source":
            obj = bpy.data.objects.get("Open Mesh Surface Shell Emitter")
            checks["real_mesh_shell_emitter"] = bool(
                obj and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "MESH_SHELL"
                and obj.fluidsim.mesh_surface_thickness > 0.0)
        elif name == "point_source":
            obj = bpy.data.objects.get("Rasterized Point Emitter")
            checks["real_rasterized_point_source"] = bool(
                obj and obj.type == "POINTCLOUD" and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "POINTS"
                and obj.fluidsim.point_source_representation == "RASTERIZED_REGION")
        elif name == "volume_source":
            obj = bpy.data.objects.get("Fog Grid Liquid Emitter")
            checks["real_volume_source"] = bool(
                obj and obj.type == "VOLUME" and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "VOLUME_GRID"
                and obj.fluidsim.volume_mask_grid == "density")
        elif name == "liquid_sink":
            obj = bpy.data.objects.get("Mesh Solid Liquid Sink")
            checks["real_liquid_sink"] = bool(
                obj and obj.fluidsim.role == "SINK"
                and obj.fluidsim.sink_region_kind == "MESH_SOLID")
        elif name == "height_fill":
            checks["real_profile_height_fill"] = bool(
                ds and ds.height_enabled and ds.height_back_mode == "PROFILE"
                and ds.height_profile_image is not None
                and ds.height_emission_mode == "INITIAL_FILL")
    elif name == "adhesive_calender":
        low_seed = bpy.data.objects.get("Low Viscosity Adhesive Seed")
        high_seed = bpy.data.objects.get("High Viscosity Adhesive Seed")
        divider = bpy.data.objects.get("Viscosity Lane Divider")
        force_fields = [
            obj for obj in scene.objects
            if getattr(getattr(obj, "field", None), "type", "NONE") != "NONE"
        ]
        field_types = [obj.field.type for obj in force_fields]
        checks.update({
            "million_particle_contract": bool(
                ds and ds.max_particles == 1_000_000
                and scene.get("fluidsim_initial_particle_count") == 1_000_000),
            "250_frame_timeline": bool(
                scene.frame_end == 250 and saved_frame == 1
                and scene.get("fluidsim_simulated_frames") == 250),
            "four_max_substeps": bool(
                ds and ds.max_substeps == 4
                and 1 <= scene.get("fluidsim_max_observed_substeps", 0) <= 4),
            "whitewater_enabled": bool(ds and ds.whitewater_enabled),
            "mirrored_viscosity_seeds": bool(
                low_seed and high_seed
                and low_seed.type == high_seed.type == "MESH"
                and len(low_seed.data.vertices) == 500_000
                and len(high_seed.data.vertices) == 500_000
                and low_seed.fluidsim.enabled
                and high_seed.fluidsim.enabled
                and low_seed.fluidsim.auto_inject_seed
                and high_seed.fluidsim.auto_inject_seed
                and abs(float(low_seed.fluidsim.viscosity) - 0.002) < 1.0e-8
                and abs(float(high_seed.fluidsim.viscosity) - 0.5) < 1.0e-8),
            "live_playback_saved": bool(fs and fs.live),
            "fluid_flow_and_drag_fields": bool(
                len(force_fields) == 3
                and field_types.count("FLUID_FLOW") == 2
                and field_types.count("DRAG") == 1
                and not any(kind in field_types for kind in (
                    "VORTEX", "HARMONIC", "TURBULENCE"))),
            "per_particle_viscosity": bool(
                ds and ds.viscosity_enabled and ds.viscosity_from_particles
                and scene.get("fluidsim_low_viscosity") == 0.002
                and scene.get("fluidsim_high_viscosity") == 0.5),
            "separated_lanes": bool(
                divider and divider.fluidsim.role == "COLLIDER"
                and float(divider.dimensions.y) >= 0.239),
            "measured_final_state": bool(
                scene.get("fluidsim_final_particle_count", 0) > 0
                and scene.get("fluidsim_final_low_core_coupling", 0.0) >= 0.0
                and scene.get("fluidsim_final_high_core_coupling", 0.0) >= 0.0
                and scene.get("fluidsim_measured_viscosity_effect") is True),
            "review_workspace_and_clips": bool(
                bpy.data.texts.get("README_Adhesive_Calender")
                and all(bpy.data.objects.get(camera) for camera in (
                    "Camera Main", "Camera Top", "Camera Profile"))
                and "Adhesive Calender" in bpy.context.workspace.name),
        })
        if checks["mirrored_viscosity_seeds"] and checks["live_playback_saved"]:
            try:
                from fluidSim_warp_blender51.blender.scene_bridge import session

                scene.frame_set(2)
                sess = session()
                before = sess.solver.particles.position[:1024].numpy().copy()
                scene.frame_set(3)
                after = sess.solver.particles.position[:1024].numpy()
                delta = float(np.max(np.linalg.norm(after - before, axis=1)))
                checks["viewport_playback_advances"] = bool(
                    sess.active
                    and sess.solver.particles.refresh_host_count() == 1_000_000
                    and delta > 1.0e-6)
                result["playback_position_delta"] = delta
            except Exception as exc:  # noqa: BLE001
                checks["viewport_playback_advances"] = False
                result["playback_error"] = repr(exc)
            finally:
                fs.live = False
                bpy.ops.fluidsim.release()
        else:
            checks["viewport_playback_advances"] = False

    result["checks"] = checks
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")
    result["surface_vertices"] = len(surface.data.vertices) if surface and surface.type == "MESH" else 0
    result["particle_points"] = point_count(particles)
    result["whitewater_points"] = point_count(whitewater)

    settings = ds
    unmet = effect_precondition_failures(settings) if settings is not None else {}
    checks["enabled_effects_have_driving_settings"] = not unmet
    if unmet:
        result["unreachable_effects"] = unmet
    if settings is not None and settings.whitewater_enabled:
        reachable, report = check_whitewater_reachable(name)
        checks["whitewater_emits_in_declared_window"] = reachable
        result["whitewater_reachability"] = report
    return result


def main():
    disk_scenes = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(DEMOS)
        if filename.endswith(".blend")
    }
    declared_scenes = set(SCENES)
    if disk_scenes != declared_scenes:
        print(json.dumps({
            "undeclared_demo_blends": sorted(disk_scenes - declared_scenes),
            "missing_demo_blends": sorted(declared_scenes - disk_scenes),
        }, indent=2))
        raise SystemExit("DEMO-BLEND-CHECK-FAIL (scene inventory mismatch)")
    requested = tuple(sys.argv[sys.argv.index("--") + 1:]) if "--" in sys.argv else ()
    unknown = sorted(set(requested) - declared_scenes)
    if unknown:
        raise SystemExit(f"DEMO-BLEND-CHECK-FAIL (unknown scenes: {', '.join(unknown)})")
    results = [check_scene(name) for name in (requested or SCENES)]
    passed = all(
        item["exists"] and item["opened"] and all(item["checks"].values())
        for item in results
    )
    print(json.dumps(results, indent=2))
    print("DEMO-BLEND-CHECK-" + ("PASS" if passed else "FAIL"))
    sys.exit(0 if passed else 1)


main()
