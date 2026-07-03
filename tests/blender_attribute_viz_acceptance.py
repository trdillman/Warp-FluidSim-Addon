"""Background acceptance for viewport attribute visualization modes."""

from __future__ import annotations

import addon_utils
import sys

import bpy
import numpy as np


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


def main():
    addon_utils.enable("fluidSim_warp_blender51", default_set=True, persistent=True)
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.voxel_size = 0.08
    fs.max_particles = 500_000
    fs.whitewater_enabled = True
    fs.ww_churn_threshold = 0.05
    fs.ww_emission_rate = 4.0

    bpy.ops.fluidsim.setup_scene()
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
