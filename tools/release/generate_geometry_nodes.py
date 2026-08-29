"""Generate the deterministic bundled FluidSim Geometry Nodes asset.

Run only with the supplied Blender package.  The generated .blend contains no
simulation/runtime binaries and is validated by ``validate_phase4_blender.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

# Resolve package source when Blender executes this file directly.
ROOT = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

ASSET_REVISION = "fluidsim-render-p4-v1"
GROUPS = (
    ("FluidSim Surface Render [P4 v1]", "SURFACE"),
    ("FluidSim Liquid Particles [P4 v1]", "PARTICLES"),
    ("FluidSim Whitewater [P4 v1]", "WHITEWATER"),
    ("FluidSim Foam [P4 v1]", "FOAM"),
    ("FluidSim Bubbles [P4 v1]", "BUBBLES"),
    ("FluidSim Spray [P4 v1]", "SPRAY"),
)


def _socket(tree, name, in_out, socket_type, *, default=None, minimum=None, maximum=None):
    socket = tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
    if default is not None and hasattr(socket, "default_value"):
        socket.default_value = default
    if minimum is not None and hasattr(socket, "min_value"):
        socket.min_value = minimum
    if maximum is not None and hasattr(socket, "max_value"):
        socket.max_value = maximum
    return socket


def _base_tree(name: str, kind: str):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.use_fake_user = True
    tree["fluidsim_asset_key"] = name
    tree["fluidsim_source_revision"] = ASSET_REVISION
    tree["fluidsim_output_kind"] = kind
    tree["fluidsim_generated_by"] = "tools/release/generate_geometry_nodes.py"
    _socket(tree, "Geometry", "INPUT", "NodeSocketGeometry")
    _socket(tree, "Material", "INPUT", "NodeSocketMaterial")
    _socket(tree, "Geometry", "OUTPUT", "NodeSocketGeometry")
    group_in = tree.nodes.new("NodeGroupInput")
    group_in.name = "FluidSim Inputs"
    group_in.location = (-520, 0)
    group_out = tree.nodes.new("NodeGroupOutput")
    group_out.name = "FluidSim Output"
    group_out.is_active_output = True
    group_out.location = (500, 0)
    return tree, group_in, group_out


def _set_material(tree, geometry_socket, material_socket, group_out, location=(220, 0)):
    node = tree.nodes.new("GeometryNodeSetMaterial")
    node.name = "FluidSim Material"
    node.location = location
    tree.links.new(geometry_socket, node.inputs["Geometry"])
    tree.links.new(material_socket, node.inputs["Material"])
    tree.links.new(node.outputs["Geometry"], group_out.inputs["Geometry"])
    return node


def _surface_group(name: str):
    tree, group_in, group_out = _base_tree(name, "SURFACE")
    _set_material(
        tree,
        group_in.outputs["Geometry"],
        group_in.outputs["Material"],
        group_out,
    )
    return tree


def _point_group(name: str, kind: str):
    tree, group_in, group_out = _base_tree(name, kind)
    _socket(
        tree,
        "Radius Scale",
        "INPUT",
        "NodeSocketFloat",
        default=1.0,
        minimum=0.0,
        maximum=100.0,
    )
    _socket(
        tree,
        "Lifetime Cutoff",
        "INPUT",
        "NodeSocketFloat",
        default=1.0,
        minimum=0.0,
        maximum=1.0,
    )

    radius = tree.nodes.new("GeometryNodeInputRadius")
    radius.name = "FluidSim Radius"
    radius.location = (-520, -180)
    scale = tree.nodes.new("ShaderNodeMath")
    scale.name = "Scale Radius"
    scale.operation = "MULTIPLY"
    scale.location = (-300, -120)
    set_radius = tree.nodes.new("GeometryNodeSetPointRadius")
    set_radius.name = "FluidSim Point Radius"
    set_radius.location = (-60, 40)
    tree.links.new(radius.outputs["Radius"], scale.inputs[0])
    tree.links.new(group_in.outputs["Radius Scale"], scale.inputs[1])
    tree.links.new(group_in.outputs["Geometry"], set_radius.inputs["Points"])
    tree.links.new(scale.outputs["Value"], set_radius.inputs["Radius"])

    age = tree.nodes.new("GeometryNodeInputNamedAttribute")
    age.name = "FluidSim Age"
    age.data_type = "FLOAT"
    age.inputs["Name"].default_value = "fluidsim_age"
    age.location = (-300, -330)
    lifetime = tree.nodes.new("GeometryNodeInputNamedAttribute")
    lifetime.name = "FluidSim Lifetime"
    lifetime.data_type = "FLOAT"
    lifetime.inputs["Name"].default_value = "fluidsim_lifetime"
    lifetime.location = (-300, -460)
    safe_life = tree.nodes.new("ShaderNodeMath")
    safe_life.name = "Safe Lifetime"
    safe_life.operation = "MAXIMUM"
    safe_life.inputs[1].default_value = 1.0e-8
    safe_life.location = (-70, -450)
    fraction = tree.nodes.new("ShaderNodeMath")
    fraction.name = "Life Fraction"
    fraction.operation = "DIVIDE"
    fraction.location = (130, -360)
    expired = tree.nodes.new("ShaderNodeMath")
    expired.name = "Lifetime Cutoff"
    expired.operation = "GREATER_THAN"
    expired.location = (330, -340)
    delete = tree.nodes.new("GeometryNodeDeleteGeometry")
    delete.name = "Cull Expired Points"
    delete.domain = "POINT"
    delete.location = (150, 40)
    tree.links.new(lifetime.outputs["Attribute"], safe_life.inputs[0])
    tree.links.new(age.outputs["Attribute"], fraction.inputs[0])
    tree.links.new(safe_life.outputs["Value"], fraction.inputs[1])
    tree.links.new(fraction.outputs["Value"], expired.inputs[0])
    tree.links.new(group_in.outputs["Lifetime Cutoff"], expired.inputs[1])
    tree.links.new(set_radius.outputs["Points"], delete.inputs["Geometry"])
    tree.links.new(expired.outputs["Value"], delete.inputs["Selection"])

    sphere = tree.nodes.new("GeometryNodeMeshIcoSphere")
    sphere.name = "FluidSim Render Sphere"
    sphere.location = (120, 210)
    sphere.inputs["Radius"].default_value = 1.0
    sphere.inputs["Subdivisions"].default_value = 2
    combine_scale = tree.nodes.new("ShaderNodeCombineXYZ")
    combine_scale.name = "FluidSim Instance Scale"
    combine_scale.location = (140, -110)
    for axis in ("X", "Y", "Z"):
        tree.links.new(scale.outputs["Value"], combine_scale.inputs[axis])
    instance = tree.nodes.new("GeometryNodeInstanceOnPoints")
    instance.name = "FluidSim Particle Instances"
    instance.location = (360, 80)
    tree.links.new(delete.outputs["Geometry"], instance.inputs["Points"])
    tree.links.new(sphere.outputs["Mesh"], instance.inputs["Instance"])
    tree.links.new(combine_scale.outputs["Vector"], instance.inputs["Scale"])

    # Realization is deliberate: Blender shader named attributes on the source
    # point domain are not visible on unrealized instance geometry. Realizing
    # transfers those fields onto the generated sphere vertices so materials
    # can consume fluidsim_color and every other enabled render channel in F12.
    realize = tree.nodes.new("GeometryNodeRealizeInstances")
    realize.name = "FluidSim Realize Render Instances"
    realize.location = (560, 80)
    tree.links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    _set_material(
        tree,
        realize.outputs["Geometry"],
        group_in.outputs["Material"],
        group_out,
        location=(780, 80),
    )
    return tree


def generate(output: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for name, kind in GROUPS:
        if kind == "SURFACE":
            _surface_group(name)
        else:
            _point_group(name, kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    generate(_parse(args).output.resolve())
