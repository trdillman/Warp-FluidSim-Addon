"""Background acceptance for viewport attribute visualization modes."""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon
from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings


CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), str(detail)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def _color_stats(attr):
    vals = [0.0] * (len(attr.data) * 4)
    attr.data.foreach_get("color", vals)
    rgba = np.asarray(vals, dtype=np.float32).reshape(-1, 4)
    rgb = rgba[:, :3]
    return float(rgb.min()), float(rgb.max()), float(np.max(np.std(rgb, axis=0)))


def _material_uses_display_color(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    for node in mat.node_tree.nodes:
        if getattr(node, "bl_idname", "") == "ShaderNodeAttribute":
            if getattr(node, "attribute_name", "") == "display_color":
                return True
    return False


def _check_attribute(data, name, data_type, domain, count):
    attr = data.attributes.get(name)
    check(f"{name} exists", attr is not None)
    if attr is not None:
        check(f"{name} type", attr.data_type == data_type, attr.data_type)
        check(f"{name} domain", attr.domain == domain, attr.domain)
        check(f"{name} aligned", len(attr.data) == count, len(attr.data))
    return attr


def _geometry_nodes_exists_probe(obj, attribute_name):
    probe_name = "fluidsim_gn_velocity_exists"
    modifier = obj.modifiers.new(
        name="FluidSim Attribute Access Probe", type="NODES")
    tree = bpy.data.node_groups.new(
        "FluidSim Attribute Access Probe", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = tree.nodes
    links = tree.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    named = nodes.new("GeometryNodeInputNamedAttribute")
    named.data_type = "FLOAT_VECTOR"
    named.inputs["Name"].default_value = attribute_name
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "BOOLEAN"
    store.domain = "POINT"
    store.inputs["Name"].default_value = probe_name
    links.new(group_in.outputs["Geometry"], store.inputs["Geometry"])
    links.new(named.outputs["Exists"], store.inputs["Value"])
    links.new(store.outputs["Geometry"], group_out.inputs["Geometry"])

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    attr = evaluated.data.attributes.get(probe_name)
    values = [False] * len(evaluated.data.vertices)
    if attr is not None and values:
        attr.data.foreach_get("value", values)
    return evaluated, attr, values


def main():
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    scene = bpy.context.scene
    fs = scene.fluidsim
    bpy.ops.fluidsim.setup_scene()
    domain = active_domain_settings(scene)
    domain.voxel_size = 0.08
    domain.max_particles = 500_000
    domain.whitewater_enabled = True
    # The per-attribute toggles only apply with Export Attributes on (off by
    # default), and this gate checks the exported attributes.
    domain.export_attributes = True
    domain.surface_attribute_velocity = True
    domain.surface_attribute_speed = True
    domain.surface_attribute_motion = True
    domain.ww_churn_gain = 10.0
    domain.ww_churn_threshold = 0.05
    domain.ww_emission_rate = 4.0
    # This gate validates whitewater color attributes, so configure an explicit
    # energetic Continuous Inflow instead of depending on the new P3 emitter
    # default (Initial Fill) to happen to generate whitewater.
    source = next(
        obj for obj in scene.objects
        if getattr(getattr(obj, "fluidsim", None), "role", "NONE") == "SOURCE"
    )
    source.fluidsim.emission_mode = "CONTINUOUS_INFLOW"
    source.fluidsim.initial_velocity = (2.0, 0.0, 1.0)
    bpy.ops.fluidsim.initialize()

    fs.liquid_view_mode = "PARTICLES"
    check("particle-only mode toggles flags", fs.show_particles and not fs.show_surface)
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    check("surface+particle mode toggles flags", fs.show_particles and fs.show_surface)

    fs.surface_attribute_mode = "SPEED"
    fs.particle_attribute_mode = "SPEED"
    fs.whitewater_attribute_mode = "AGE"
    fs.particle_display_fraction = 1.0

    bpy.ops.fluidsim.step(frames=8)

    surf = bpy.data.objects.get("FluidSim Surface")
    check("surface output exists", surf is not None and surf.type == "MESH")
    if surf is not None:
        surface_count = len(surf.data.vertices)
        _check_attribute(
            surf.data, "fluidsim_velocity", "FLOAT_VECTOR", "POINT",
            surface_count)
        _check_attribute(
            surf.data, "fluidsim_speed", "FLOAT", "POINT", surface_count)
        _check_attribute(
            surf.data, "velocity", "FLOAT_VECTOR", "POINT", surface_count)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = surf.evaluated_get(depsgraph)
        evaluated_count = len(evaluated.data.vertices)
        _check_attribute(
            evaluated.data, "fluidsim_velocity", "FLOAT_VECTOR", "POINT",
            evaluated_count)
        _check_attribute(
            evaluated.data, "fluidsim_speed", "FLOAT", "POINT",
            evaluated_count)
        _check_attribute(
            evaluated.data, "velocity", "FLOAT_VECTOR", "POINT",
            evaluated_count)

        evaluated, probe, probe_values = _geometry_nodes_exists_probe(
            surf, "fluidsim_velocity")
        check(
            "Geometry Nodes Named Attribute existence probe",
            probe is not None and bool(probe_values) and all(probe_values),
            f"{sum(bool(value) for value in probe_values)}/"
            f"{len(evaluated.data.vertices)} points",
        )
    sattr = surf.data.color_attributes.get("display_color") if surf else None
    check("surface display_color attribute", sattr is not None and len(sattr.data) > 0,
          f"{len(sattr.data) if sattr else 0} colors")
    if sattr is not None:
        lo, hi, spread = _color_stats(sattr)
        check("surface colors vary spatially", spread > 1.0e-4,
              f"rgb {lo:.3f}..{hi:.3f}, std {spread:.4f}")
    check("surface attribute material",
          surf.data.materials and surf.data.materials[0].name.startswith(
              "FluidSim Surface Attribute"))
    check("surface material binds display_color",
          _material_uses_display_color(surf.data.materials[0] if surf.data.materials else None))

    pts = bpy.data.objects.get("FluidSim Particles")
    check("particle output exists", pts is not None and pts.type == "POINTCLOUD")
    pattr = pts.data.attributes.get("display_color") if pts else None
    check("particle display_color attribute", pattr is not None and len(pattr.data) > 0,
          f"{len(pattr.data) if pattr else 0} colors")
    if pattr is not None:
        lo, hi, spread = _color_stats(pattr)
        check("particle colors vary spatially", spread > 1.0e-4,
              f"rgb {lo:.3f}..{hi:.3f}, std {spread:.4f}")
    check("particle attribute material",
          pts.data.materials and pts.data.materials[0].name.startswith(
              "FluidSim Particle Attribute"))
    check("particle material binds display_color",
          _material_uses_display_color(pts.data.materials[0] if pts.data.materials else None))

    fs.particle_attribute_mode = "AGE"
    bpy.ops.fluidsim.step(frames=1)
    pts = bpy.data.objects.get("FluidSim Particles")
    age_attr = pts.data.attributes.get("display_color") if pts else None
    check("particle age attribute mode works",
          age_attr is not None and len(age_attr.data) > 0,
          f"{len(age_attr.data) if age_attr else 0} colors")

    ww = bpy.data.objects.get("FluidSim Whitewater")
    check("whitewater output exists", ww is not None and ww.type == "POINTCLOUD")
    wattr = ww.data.attributes.get("display_color") if ww else None
    check("whitewater display_color attribute",
          wattr is not None and len(wattr.data) > 0,
          f"{len(wattr.data) if wattr else 0} colors")
    check("whitewater attribute material",
          ww.data.materials and ww.data.materials[0].name.startswith(
              "FluidSim Whitewater Attribute"))
    check("whitewater material binds display_color",
          _material_uses_display_color(ww.data.materials[0] if ww.data.materials else None))

    failed = [name for name, ok, _detail in CHECKS if not ok]
    if failed:
        print(f"ATTRIBUTE-VIZ-ACCEPTANCE-FAIL {failed}")
        return 1
    print("ATTRIBUTE-VIZ-ACCEPTANCE-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
