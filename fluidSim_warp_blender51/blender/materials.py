"""Default materials for liquid surface and whitewater outputs."""

from __future__ import annotations

import bpy

LIQUID_MATERIAL = "FluidSim Liquid"
WHITEWATER_MATERIAL = "FluidSim Whitewater"


def _new_liquid_material():
    mat = bpy.data.materials.new(LIQUID_MATERIAL)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.8, 0.95, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.02
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.33
    # Viewport look.
    mat.diffuse_color = (0.35, 0.65, 0.9, 0.65)
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND"
        except Exception:
            pass
    return mat


def _new_whitewater_material():
    mat = bpy.data.materials.new(WHITEWATER_MATERIAL)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.05
    mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    return mat


def ensure_surface_material(scene, obj):
    fs = scene.fluidsim
    mat = fs.surface_material
    if mat is None:
        mat = bpy.data.materials.get(LIQUID_MATERIAL) or _new_liquid_material()
    if obj.data and (not obj.data.materials or obj.data.materials[0] is not mat):
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def ensure_whitewater_material(scene, obj):
    fs = scene.fluidsim
    mat = fs.whitewater_material
    if mat is None:
        mat = (bpy.data.materials.get(WHITEWATER_MATERIAL)
               or _new_whitewater_material())
    if obj.data is None:
        return
    if hasattr(obj.data, "materials"):
        if not obj.data.materials or obj.data.materials[0] is not mat:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
