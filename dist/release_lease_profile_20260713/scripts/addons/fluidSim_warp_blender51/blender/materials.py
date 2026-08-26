"""Default materials for liquid surface and whitewater outputs."""

from __future__ import annotations

import bpy

LIQUID_MATERIAL = "FluidSim Liquid"
PARTICLE_MATERIAL = "FluidSim Particles"
WHITEWATER_MATERIAL = "FluidSim Whitewater"
SURFACE_ATTR_MATERIAL = "FluidSim Surface Attribute"
PARTICLE_ATTR_MATERIAL = "FluidSim Particle Attribute"
WHITEWATER_ATTR_MATERIAL = "FluidSim Whitewater Attribute"


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


def _new_particle_material():
    mat = bpy.data.materials.new(PARTICLE_MATERIAL)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.1, 0.45, 1.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.35
    mat.diffuse_color = (0.1, 0.45, 1.0, 1.0)
    return mat


def _new_attribute_material(name, fallback_color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = fallback_color
        bsdf.inputs["Roughness"].default_value = 0.2
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "display_color"
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    mat.diffuse_color = fallback_color
    return mat


def _assign_material(obj, mat):
    if obj.data is None or not hasattr(obj.data, "materials"):
        return
    if not obj.data.materials or obj.data.materials[0] is not mat:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    try:
        obj.color = tuple(mat.diffuse_color)
        obj.display_type = "TEXTURED"
    except Exception:
        pass


def ensure_surface_material(scene, obj):
    fs = scene.fluidsim
    if fs.surface_attribute_mode != "NONE":
        mat = (bpy.data.materials.get(SURFACE_ATTR_MATERIAL)
               or _new_attribute_material(SURFACE_ATTR_MATERIAL,
                                          (0.1, 0.55, 1.0, 1.0)))
    else:
        mat = fs.surface_material
        if mat is None:
            mat = bpy.data.materials.get(LIQUID_MATERIAL) or _new_liquid_material()
    _assign_material(obj, mat)


def ensure_particle_material(scene, obj):
    fs = scene.fluidsim
    if fs.particle_attribute_mode != "NONE":
        mat = (bpy.data.materials.get(PARTICLE_ATTR_MATERIAL)
               or _new_attribute_material(PARTICLE_ATTR_MATERIAL,
                                          (0.1, 0.55, 1.0, 1.0)))
    else:
        mat = fs.particle_material
        if mat is None:
            mat = bpy.data.materials.get(PARTICLE_MATERIAL) or _new_particle_material()
    _assign_material(obj, mat)


def ensure_whitewater_material(scene, obj):
    fs = scene.fluidsim
    if fs.whitewater_attribute_mode != "NONE":
        mat = (bpy.data.materials.get(WHITEWATER_ATTR_MATERIAL)
               or _new_attribute_material(WHITEWATER_ATTR_MATERIAL,
                                          (1.0, 0.95, 0.65, 1.0)))
    else:
        mat = fs.whitewater_material
        if mat is None:
            mat = (bpy.data.materials.get(WHITEWATER_MATERIAL)
                   or _new_whitewater_material())
    _assign_material(obj, mat)
