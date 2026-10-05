"""Build the FluidSim Warp Custom Effect example scenes.

Generates 12 example scenes (add-on examples, not the canonical demos; see
docs/DEMO_SCENE_CONTRACT.md):
  10 dedicated single-effect scenes
  + 2 multi-effect layered scenes combining stack layers.

Effects are trusted only in a private temporary store, so the scenes ship
Untrusted Imported and the user's trust.json is never touched.

Usage:
  blender --background --python tools/release/build_custom_effect_example_scenes.py -- --all
  blender --background --python tools/release/build_custom_effect_example_scenes.py -- --scene 01_thermal_updraft_example

Do not add --factory-startup without a private BLENDER_USER_EXTENSIONS: it can
delete the installed Warp wheels from the real Blender profile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

DEFAULT_OUTPUT_DIR = os.path.join(
    REPO_ROOT, "fluidSim_warp_blender51", "resources", "examples", "custom_effects", "scenes"
)


def _log(msg: str) -> None:
    print(f"[CustomEffectExampleBuilder] {msg}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Build all 12 example scenes")
    parser.add_argument("--scene", type=str, help="Specific example scene name to build")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Destination directory")
    parser.add_argument("--device", type=str, default="cpu", help="Warp compile target device (cpu or cuda:0)")
    parser.add_argument("--dry-run", action="store_true", help="Configure without saving .blend files")
    return parser.parse_args(argv)


def fresh_scene(scene_name: str) -> bpy.types.Scene:
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)
    from fluidSim_warp_blender51.blender.scene_bridge import release_session

    scene = bpy.context.scene
    release_session(scene)
    scene.name = scene_name
    scene.property_unset("fluidsim")
    if hasattr(scene, "fluidsim_effects"):
        scene.fluidsim_effects.items.clear()
        scene.fluidsim_effects.active_index = 0
        scene.fluidsim_effects.applied_signature = ""

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.pointclouds, bpy.data.materials):
        for datablock in list(block):
            if datablock.users == 0:
                block.remove(datablock)

    scene.render.fps = 24
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = 120
    scene.frame_current = 1
    return scene


def add_camera_and_light(scene: bpy.types.Scene, cam_loc=(4.2, -4.2, 3.2), cam_rot=(1.05, 0.0, 0.785)):
    cam_data = bpy.data.cameras.new("Example Cam")
    cam = bpy.data.objects.new("Example Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = cam_loc
    cam.rotation_euler = cam_rot
    scene.camera = cam

    sun_data = bpy.data.lights.new("Example Sun", type="SUN")
    sun_data.energy = 3.5
    sun = bpy.data.objects.new("Example Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.75, 0.2, 0.5)


def make_domain(scene: bpy.types.Scene, lo: tuple[float, float, float], hi: tuple[float, float, float], *, voxel_size: float = 0.08, particle_radius: float = 0.04):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    dom = bpy.context.active_object
    dom.name = "FluidSim Domain"
    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    dimensions = tuple(hi[i] - lo[i] for i in range(3))
    dom.location = center
    dom.dimensions = dimensions
    dom.display_type = "WIRE"
    dom.hide_render = True
    scene.fluidsim.domain_object = dom

    ds = dom.fluidsim_domain
    ds.voxel_size = voxel_size
    ds.particle_radius = particle_radius
    return dom


def make_source_box(scene: bpy.types.Scene, lo: tuple[float, float, float], hi: tuple[float, float, float], name: str = "Fluid Reservoir", emission_mode: str = "INITIAL_FILL"):
    from fluidSim_warp_blender51.blender import sources as _sources

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    src = bpy.context.active_object
    src.name = name
    src.location = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    src.dimensions = tuple(hi[i] - lo[i] for i in range(3))
    _sources.tag_source(src, shape="BOX", mode="PARTICLES")
    src.fluidsim.emission_mode = emission_mode
    src.fluidsim.source_region_kind = "ANALYTIC"
    src.fluidsim.use_object_velocity = False
    return src


def make_controller(scene: bpy.types.Scene, name: str, location: tuple[float, float, float], display_type: str = "PLAIN_AXES", size: float = 0.6) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display_type
    empty.empty_display_size = size
    empty.location = location
    empty["fluidsim_owned"] = True
    scene.collection.objects.link(empty)
    return empty


def add_effect_layer(scene: bpy.types.Scene, effect_def, *, instance_id: str, parameters: dict | None = None, controller: bpy.types.Object | None = None):
    from fluidSim_warp_blender51.blender import effects
    from fluidSim_warp_blender51.sim.effects.enums import TrustState
    from fluidSim_warp_blender51.sim.effects.schema import EffectDefinitionV2, EffectInstanceV2
    from fluidSim_warp_blender51.sim.effects.trust import content_trust_digest

    defn = EffectDefinitionV2.from_dict(effect_def.to_dict()) if hasattr(effect_def, "to_dict") else EffectDefinitionV2.from_dict(effect_def)

    instance = EffectInstanceV2(
        instance_id=instance_id,
        definition=defn,
        enabled=True,
        order=len(scene.fluidsim_effects.items),
        display_name=defn.name,
    )
    pg = effects.add_instance_to_scene(scene, instance)

    if parameters:
        for key, value in parameters.items():
            param_pg = next((p for p in pg.parameters if p.key == key), None)
            if param_pg is not None:
                if param_pg.kind == "FLOAT":
                    param_pg.float_value = float(value)
                elif param_pg.kind == "INT":
                    param_pg.int_value = int(value)
                elif param_pg.kind == "BOOL":
                    param_pg.bool_value = bool(value)
                elif param_pg.kind == "VECTOR":
                    param_pg.vector_value = tuple(value)
                elif param_pg.kind == "COLOR":
                    param_pg.color_value = tuple(value)
                elif param_pg.kind == "OBJECT":
                    param_pg.object_value = value

    if controller is not None:
        for param_pg in pg.parameters:
            if param_pg.kind == "OBJECT":
                param_pg.object_value = controller
                break

    # Trust lives in the builder's private store (see main), never the user's
    # profile; a shipped scene therefore opens Untrusted Imported elsewhere.
    digest = content_trust_digest(effect_def)
    effects.trust_store_for_scene(scene).trust(digest, pg.display_name)
    pg.trust_state = TrustState.LOCAL_TRUSTED.value
    pg.status = "DRAFT"
    effects.refresh_stack_state(scene)
    return pg


def finalize_and_save(scene: bpy.types.Scene, filename: str, output_dir: Path, *, device: str = "cpu", dry_run: bool = False) -> Path:
    from fluidSim_warp_blender51.blender import effects

    candidate = effects.apply_stack(scene, device=device)
    if not candidate.success:
        raise RuntimeError(f"Failed to apply effect stack in {filename}: {candidate.issues}")

    effects.refresh_stack_state(scene)
    scene.fluidsim.live = True
    scene["fluidsim_playback_ready"] = True
    scene["fluidsim_custom_effect_example"] = filename.removesuffix(".blend")

    out_path = output_dir / filename
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res = bpy.ops.wm.save_as_mainfile(filepath=str(out_path), check_existing=False, compress=True)
        if "FINISHED" not in res:
            raise RuntimeError(f"Save failed for {out_path}: {res}")
        # Blender keeps the previous file as .blend1; build_extension rejects
        # any .blend1 under the package tree.
        backup = out_path.with_suffix(".blend1")
        if backup.exists():
            backup.unlink()
        _log(f"Saved example scene -> {out_path} ({out_path.stat().st_size:,} bytes)")
    else:
        _log(f"Dry-run configured successfully -> {filename}")
    return out_path


# -----------------------------------------------------------------------------
# Example Scene Definitions
# -----------------------------------------------------------------------------

def build_01_thermal_updraft(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Thermal Updraft")
    add_camera_and_light(scene, cam_loc=(3.6, -3.6, 2.8), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-1.2, -1.2, 0.0), (1.2, 1.2, 2.6), voxel_size=0.075)
    make_source_box(scene, (-0.45, -0.45, 0.1), (0.45, 0.45, 1.2), name="Heated Liquid Column")
    controller = make_controller(scene, "Updraft Origin", (0.0, 0.0, 0.0), display_type="ARROWS", size=0.5)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.thermal_updraft")
    add_effect_layer(
        scene, eff_def, instance_id="example-updraft",
        parameters={"ambient_temp": 20.0, "buoyancy": 0.8, "up_direction": (0.0, 0.0, 1.0), "max_accel": 25.0},
        controller=controller,
    )
    return finalize_and_save(scene, "01_thermal_updraft_example.blend", output_dir, device=device, dry_run=dry_run)


def build_02_point_attractor(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Point Attractor")
    add_camera_and_light(scene, cam_loc=(4.0, -4.0, 3.5), cam_rot=(1.0, 0.0, 0.785))
    make_domain(scene, (-2.0, -2.0, -0.2), (2.0, 2.0, 2.4), voxel_size=0.08)
    make_source_box(scene, (0.4, 0.4, 0.2), (1.5, 1.5, 1.2), name="Orbital Fluid Block")
    controller = make_controller(scene, "Attractor Center", (0.0, 0.0, 0.8), display_type="SPHERE", size=0.7)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.point_attractor")
    add_effect_layer(
        scene, eff_def, instance_id="example-attractor",
        parameters={"strength": 7.5, "radius": 2.8, "softening": 0.25},
        controller=controller,
    )
    return finalize_and_save(scene, "02_point_attractor_example.blend", output_dir, device=device, dry_run=dry_run)


def build_03_helical_vortex(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Helical Vortex")
    add_camera_and_light(scene, cam_loc=(3.8, -3.8, 3.0), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-1.5, -1.5, 0.0), (1.5, 1.5, 3.0), voxel_size=0.08)
    make_source_box(scene, (-1.0, -1.0, 0.1), (1.0, 1.0, 0.8), name="Fluid Basin")
    controller = make_controller(scene, "Vortex Cylinder Axis", (0.0, 0.0, 0.0), display_type="SINGLE_ARROW", size=1.0)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.helical_vortex")
    add_effect_layer(
        scene, eff_def, instance_id="example-helix",
        parameters={"axis": (0.0, 0.0, 1.0), "radius": 1.8, "swirl": 6.5, "axial_drive": 3.5},
        controller=controller,
    )
    return finalize_and_save(scene, "03_helical_vortex_example.blend", output_dir, device=device, dry_run=dry_run)


def build_04_directional_wind(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Directional Wind")
    add_camera_and_light(scene, cam_loc=(3.5, -4.5, 2.8), cam_rot=(1.1, 0.0, 0.65))
    make_domain(scene, (-2.0, -1.2, 0.0), (2.5, 1.2, 2.6), voxel_size=0.08)
    make_source_box(scene, (-1.5, -0.4, 1.4), (-0.8, 0.4, 2.2), name="Falling Fluid Inflow")
    controller = make_controller(scene, "Wind Zone", (0.0, 0.0, 1.2), display_type="ARROWS", size=0.8)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.directional_wind")
    add_effect_layer(
        scene, eff_def, instance_id="example-wind",
        parameters={"direction": (1.0, 0.0, 0.0), "speed": 6.5, "drag": 2.5, "influence_radius": 3.0, "gust_freq": 1.8, "gust_amp": 2.2},
        controller=controller,
    )
    return finalize_and_save(scene, "04_directional_wind_example.blend", output_dir, device=device, dry_run=dry_run)


def build_05_centrifugal_spinner(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Centrifugal Spinner")
    add_camera_and_light(scene, cam_loc=(4.0, -4.0, 3.2), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), voxel_size=0.08)
    make_source_box(scene, (-0.35, -0.35, 0.1), (0.35, 0.35, 1.2), name="Rotor Fluid Core")
    controller = make_controller(scene, "Spinner Shaft", (0.0, 0.0, 0.0), display_type="CIRCLE", size=0.8)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.centrifugal_spinner")
    add_effect_layer(
        scene, eff_def, instance_id="example-spinner",
        parameters={"axis": (0.0, 0.0, 1.0), "angular_speed": 6.0, "max_radius": 1.9},
        controller=controller,
    )
    return finalize_and_save(scene, "05_centrifugal_spinner_example.blend", output_dir, device=device, dry_run=dry_run)


def build_06_surface_tension_bead(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Surface Tension Bead")
    add_camera_and_light(scene, cam_loc=(3.2, -3.2, 2.5), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-1.2, -1.2, 0.0), (1.2, 1.2, 2.5), voxel_size=0.075)
    make_source_box(scene, (-0.35, -0.35, 1.2), (0.35, 0.35, 2.1), name="Discharge Column")

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.surface_tension_bead")
    add_effect_layer(
        scene, eff_def, instance_id="example-beading",
        parameters={"rest_density": 1000.0, "surface_threshold": 880.0, "damping_factor": 4.0},
    )
    return finalize_and_save(scene, "06_surface_tension_bead_example.blend", output_dir, device=device, dry_run=dry_run)


def build_07_viscous_heating(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Viscous Heating")
    add_camera_and_light(scene, cam_loc=(3.6, -3.6, 2.8), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-1.5, -1.5, 0.0), (1.5, 1.5, 2.6), voxel_size=0.08)
    make_source_box(scene, (-0.5, -0.5, 1.2), (0.5, 0.5, 2.2), name="Gravitational Jet Column")

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.viscous_heating")
    add_effect_layer(
        scene, eff_def, instance_id="example-visc-heating",
        parameters={"heating_rate": 0.8, "speed_threshold": 1.0, "max_temp": 500.0},
    )
    return finalize_and_save(scene, "07_viscous_heating_example.blend", output_dir, device=device, dry_run=dry_run)


def build_08_periodic_ripples(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Periodic Ripples")
    add_camera_and_light(scene, cam_loc=(3.5, -3.5, 3.2), cam_rot=(1.0, 0.0, 0.785))
    make_domain(scene, (-2.0, -2.0, 0.0), (2.0, 2.0, 1.4), voxel_size=0.08)
    make_source_box(scene, (-1.8, -1.8, 0.05), (1.8, 1.8, 0.35), name="Shallow Pool")
    controller = make_controller(scene, "Ripple Origin", (0.0, 0.0, 0.2), display_type="CIRCLE", size=0.6)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.periodic_ripples")
    add_effect_layer(
        scene, eff_def, instance_id="example-ripples",
        parameters={"amplitude": 2.5, "wavenumber": 3.8, "wave_speed": 2.6, "decay": 0.35, "radius": 2.2},
        controller=controller,
    )
    return finalize_and_save(scene, "08_periodic_ripples_example.blend", output_dir, device=device, dry_run=dry_run)


def build_09_bounding_box_confinement(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Bounding Box Confinement")
    add_camera_and_light(scene, cam_loc=(4.2, -4.2, 3.0), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-2.4, -2.4, 0.0), (2.4, 2.4, 2.4), voxel_size=0.08)
    make_source_box(scene, (-0.9, -0.9, 0.4), (0.1, 0.1, 1.4), name="High Energy Fluid Block")
    controller = make_controller(scene, "Confinement Box", (0.0, 0.0, 1.0), display_type="CUBE", size=1.0)

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.bounding_box_confinement")
    add_effect_layer(
        scene, eff_def, instance_id="example-box-confinement",
        parameters={"half_extents": (1.3, 1.3, 1.0), "margin": 0.25, "stiffness": 35.0},
        controller=controller,
    )
    return finalize_and_save(scene, "09_bounding_box_confinement_example.blend", output_dir, device=device, dry_run=dry_run)


def build_10_voxel_viscosity_decay(output_dir: Path, device: str, dry_run: bool) -> Path:
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Voxel Viscosity Decay")
    add_camera_and_light(scene, cam_loc=(3.6, -3.6, 2.8), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-1.5, -1.5, 0.0), (1.5, 1.5, 2.0), voxel_size=0.08)
    make_source_box(scene, (-0.8, -0.8, 0.1), (0.8, 0.8, 1.0), name="High Viscosity Fluid")

    eff_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.voxel_viscosity_decay")
    add_effect_layer(
        scene, eff_def, instance_id="example-visc-decay",
        parameters={"base_viscosity": 0.001, "decay_rate": 0.5},
    )
    return finalize_and_save(scene, "10_voxel_viscosity_decay_example.blend", output_dir, device=device, dry_run=dry_run)


# -----------------------------------------------------------------------------
# Layered Multi-Effect Example Scenes (2 Required)
# -----------------------------------------------------------------------------

def build_11_layered_atmospheric_tempest(output_dir: Path, device: str, dry_run: bool) -> Path:
    """Multi-effect layer example: Thermal Updraft + Directional Wind + Bounding Box Confinement."""
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Layered Atmospheric Tempest")
    add_camera_and_light(scene, cam_loc=(4.5, -4.5, 3.5), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-2.0, -2.0, 0.0), (2.0, 2.0, 2.6), voxel_size=0.08)
    make_source_box(scene, (-0.6, -0.6, 0.1), (0.6, 0.6, 1.2), name="Convective Tempest Source")

    updraft_ctrl = make_controller(scene, "Tempest Updraft Origin", (0.0, 0.0, 0.0), display_type="ARROWS", size=0.5)
    wind_ctrl = make_controller(scene, "Tempest Wind Stream", (0.0, 0.0, 1.2), display_type="SINGLE_ARROW", size=0.7)
    box_ctrl = make_controller(scene, "Tempest Boundary Box", (0.0, 0.0, 1.2), display_type="CUBE", size=1.0)

    # Layer 1: Thermal Updraft
    updraft_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.thermal_updraft")
    add_effect_layer(
        scene, updraft_def, instance_id="tempest-layer1-updraft",
        parameters={"ambient_temp": 20.0, "buoyancy": 0.65, "up_direction": (0.0, 0.0, 1.0), "max_accel": 22.0},
        controller=updraft_ctrl,
    )

    # Layer 2: Directional Wind
    wind_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.directional_wind")
    add_effect_layer(
        scene, wind_def, instance_id="tempest-layer2-wind",
        parameters={"direction": (1.0, 0.4, 0.0), "speed": 5.5, "drag": 2.2, "influence_radius": 2.8, "gust_freq": 1.5, "gust_amp": 2.0},
        controller=wind_ctrl,
    )

    # Layer 3: Bounding Box Confinement
    box_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.bounding_box_confinement")
    add_effect_layer(
        scene, box_def, instance_id="tempest-layer3-box",
        parameters={"half_extents": (1.6, 1.6, 1.2), "margin": 0.25, "stiffness": 30.0},
        controller=box_ctrl,
    )

    return finalize_and_save(scene, "11_layered_atmospheric_tempest_example.blend", output_dir, device=device, dry_run=dry_run)


def build_12_layered_hydro_centrifuge(output_dir: Path, device: str, dry_run: bool) -> Path:
    """Multi-effect layer example: Centrifugal Spinner + Helical Vortex + Viscous Heating."""
    from fluidSim_warp_blender51.resources.examples.custom_effects.example_effects import EXAMPLE_EFFECT_DEFINITIONS

    scene = fresh_scene("Example: Layered Hydro Centrifuge")
    add_camera_and_light(scene, cam_loc=(4.2, -4.2, 3.4), cam_rot=(1.05, 0.0, 0.785))
    make_domain(scene, (-2.0, -2.0, 0.0), (2.0, 2.0, 2.6), voxel_size=0.08)
    make_source_box(scene, (-0.5, -0.5, 0.2), (0.5, 0.5, 1.4), name="Centrifuge Liquid Core")

    shaft_ctrl = make_controller(scene, "Centrifuge Shaft", (0.0, 0.0, 0.0), display_type="CIRCLE", size=0.8)
    helix_ctrl = make_controller(scene, "Centrifuge Helix Axis", (0.0, 0.0, 0.0), display_type="SINGLE_ARROW", size=1.0)

    # Layer 1: Centrifugal Spinner
    spinner_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.centrifugal_spinner")
    add_effect_layer(
        scene, spinner_def, instance_id="centrifuge-layer1-spinner",
        parameters={"axis": (0.0, 0.0, 1.0), "angular_speed": 5.5, "max_radius": 1.8},
        controller=shaft_ctrl,
    )

    # Layer 2: Helical Vortex
    helix_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.helical_vortex")
    add_effect_layer(
        scene, helix_def, instance_id="centrifuge-layer2-helix",
        parameters={"axis": (0.0, 0.0, 1.0), "radius": 1.8, "swirl": 4.5, "axial_drive": 2.8},
        controller=helix_ctrl,
    )

    # Layer 3: Viscous Heating
    heating_def = next(d for d in EXAMPLE_EFFECT_DEFINITIONS if d.effect_id == "example.viscous_heating")
    add_effect_layer(
        scene, heating_def, instance_id="centrifuge-layer3-heating",
        parameters={"heating_rate": 0.75, "speed_threshold": 1.2, "max_temp": 500.0},
    )

    return finalize_and_save(scene, "12_layered_hydro_centrifuge_example.blend", output_dir, device=device, dry_run=dry_run)


BUILDERS = {
    "01_thermal_updraft_example": build_01_thermal_updraft,
    "02_point_attractor_example": build_02_point_attractor,
    "03_helical_vortex_example": build_03_helical_vortex,
    "04_directional_wind_example": build_04_directional_wind,
    "05_centrifugal_spinner_example": build_05_centrifugal_spinner,
    "06_surface_tension_bead_example": build_06_surface_tension_bead,
    "07_viscous_heating_example": build_07_viscous_heating,
    "08_periodic_ripples_example": build_08_periodic_ripples,
    "09_bounding_box_confinement_example": build_09_bounding_box_confinement,
    "10_voxel_viscosity_decay_example": build_10_voxel_viscosity_decay,
    "11_layered_atmospheric_tempest_example": build_11_layered_atmospheric_tempest,
    "12_layered_hydro_centrifuge_example": build_12_layered_hydro_centrifuge,
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    _log(f"Output directory: {output_dir}")

    targets = list(BUILDERS.keys()) if (args.all or not args.scene) else [args.scene]
    for name in targets:
        if name not in BUILDERS:
            raise SystemExit(f"Unknown scene {name!r}. Available: {list(BUILDERS.keys())}")

    t0 = time.perf_counter()
    built = []
    with tempfile.TemporaryDirectory(prefix="fluidsim-example-trust-") as trust_dir:
        # Build with a private trust store so the user's real one is untouched.
        from fluidSim_warp_blender51.blender import effects
        from fluidSim_warp_blender51.sim.effects.trust import TrustStore

        store = TrustStore(Path(trust_dir) / "trust.json")
        original = effects.trust_store_for_scene
        effects.trust_store_for_scene = lambda _scene=None: store
        try:
            for name in targets:
                _log(f"Building [{name}] on {args.device}...")
                built.append(BUILDERS[name](output_dir, args.device, args.dry_run))
        finally:
            effects.trust_store_for_scene = original

    elapsed = time.perf_counter() - t0
    _log(f"Successfully generated {len(built)} custom effect example scenes in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
