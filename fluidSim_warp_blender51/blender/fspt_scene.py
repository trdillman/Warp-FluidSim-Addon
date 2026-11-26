"""Read Blender's evaluated scene into owned FSPT records on its callback thread.

This module has no bpy import so material translation can be checked separately.
Its record tests do not qualify actual Blender extraction or material appearance.
"""
import numpy as np

from ..fspt.contracts import CameraSnapshot, Material, SceneSnapshot


class UnsupportedSceneError(RuntimeError):
    pass


def _fail(owner, reason):
    raise UnsupportedSceneError(f"{getattr(owner, 'name', 'Scene')}: {reason}")


def _input(node, name, default=None):
    socket = node.inputs.get(name)
    return default if socket is None else socket.default_value


def _inputs(node):
    return node.inputs.values() if isinstance(node.inputs, dict) else node.inputs


def _shader(owner, output_type):
    outputs = [n for n in owner.node_tree.nodes if n.bl_idname == output_type
               and getattr(n, "is_active_output", False)]
    if not outputs:
        _fail(owner, "no active shader output")
    output = next((n for n in outputs if getattr(n, "target", "ALL") == "ALL"), None)
    if output is None:
        _fail(owner, "engine-specific output graphs are not supported by P1")
    for name in ("Volume", "Displacement"):
        socket = output.inputs.get(name)
        if socket is not None and socket.is_linked:
            _fail(owner, f"linked {name} graph is not supported by P1")
    surface = output.inputs.get("Surface")
    if surface is None or not surface.is_linked or len(surface.links) != 1:
        _fail(owner, "a single linked Surface shader is required")
    link = surface.links[0]
    if not getattr(link, "is_valid", True):
        _fail(owner, "invalid Surface graph link")
    if getattr(link, "is_muted", False):
        _fail(owner, "muted Surface graph link is not supported by P1")
    shader = link.from_node
    if getattr(shader, "mute", False):
        _fail(owner, "muted shader graph is not supported by P1")
    for socket in _inputs(shader):
        if socket.is_linked:
            _fail(owner, f"linked input {shader.name}.{socket.name} is not supported by P1")
    return shader


def translate_material(material):
    if material is None:
        return Material()
    if not material.use_nodes or material.node_tree is None:
        _fail(material, "P1 requires an explicit Diffuse, smooth Glass or Emission node graph")
    node = _shader(material, "ShaderNodeOutputMaterial")
    kind = node.bl_idname
    try:
        if kind in {"ShaderNodeBsdfDiffuse", "ShaderNodeBsdfGlass"}:
            if float(_input(node, "Roughness", 0)) != 0:
                _fail(material, f"{node.name}: rough BSDF is not supported by P1")
            color = tuple(_input(node, "Color")[:3])
            if kind == "ShaderNodeBsdfGlass":
                return Material(kind="glass", color=color, ior=float(_input(node, "IOR", 1.45)))
            return Material(color=color)
        if kind == "ShaderNodeEmission":
            strength = float(_input(node, "Strength", 1.0))
            color = tuple(float(x) * strength for x in _input(node, "Color")[:3])
            return Material(kind="emission", emission=color)
        # Full Principled is intentionally not silently approximated by Lambert.
        _fail(material, f"unsupported shader {node.name} ({kind}); P1 supports Diffuse, smooth Glass and Emission")
    except (TypeError, ValueError) as exc:
        _fail(material, f"invalid material parameters: {exc}")


def world_color(world):
    if world is None:
        return (0.0, 0.0, 0.0)
    if not world.use_nodes or world.node_tree is None:
        color = tuple(float(x) for x in world.color)
    else:
        node = _shader(world, "ShaderNodeOutputWorld")
        if node.bl_idname != "ShaderNodeBackground":
            _fail(world, "only a constant Background world is supported by P1")
        strength = float(_input(node, "Strength", 1.0))
        color = tuple(float(x) * strength for x in _input(node, "Color")[:3])
    if len(color) != 3 or not np.isfinite(color).all() or min(color) < 0:
        _fail(world, "world radiance must be finite and nonnegative")
    return color


def _validate_evaluated_geometry(obj):
    geometry = obj.evaluated_geometry()
    for component in ("pointcloud", "curves", "volume", "grease_pencil"):
        if getattr(geometry, component) is not None:
            _fail(obj, f"evaluated {component} geometry is not supported by P1")
    # Object/collection instances are exported by depsgraph.object_instances.
    # Anonymous geometry references require a separate extraction contract.
    for reference in geometry.instance_references():
        identifier = getattr(getattr(reference, "bl_rna", None), "identifier", None)
        if reference is not None and identifier not in {"Object", "Collection"}:
            _fail(obj, "anonymous geometry instances are not supported by P1; use Realize Instances")


def extract_scene(depsgraph, for_viewport=False):
    from .fspt_bridge import SurfaceNotReadyError, capture_publications, validate_publications
    from .surface_publication import id_token

    scene = depsgraph.scene
    publications = capture_publications(scene, for_viewport=for_viewport,
                                        exact_frame=not for_viewport,
                                        view_layer=depsgraph.view_layer)
    layer = depsgraph.view_layer
    if layer.material_override is not None or getattr(layer, "world_override", None) is not None:
        _fail(layer, "view-layer material/world overrides are not supported by P1")
    world = world_color(scene.world)
    vertices, triangles, assignments, materials = [], [], [], [Material()]
    material_cache = {}
    captured_outputs = set()
    expected_outputs = {p.object_token for p in publications}
    vertex_offset = 0
    for instance in depsgraph.object_instances:
        obj = instance.object
        if not for_viewport and obj.hide_render:
            continue
        if for_viewport and not instance.is_instance and not obj.original.visible_get(view_layer=layer):
            continue
        role = getattr(getattr(obj.original, "fluidsim", None), "role", "NONE")
        if role in {"DOMAIN", "SOURCE", "SINK", "FORCE"}:
            continue
        if obj.type in {"CAMERA", "EMPTY", "ARMATURE", "LATTICE"}:
            continue
        if obj.type != "MESH":
            _fail(obj, f"unsupported {obj.type} object; P1 accepts evaluated triangle meshes and mesh emitters")
        if (obj.original.get("fluidsim_output_kind") == "FluidSim Surface"
                and id_token(obj.original) not in expected_outputs):
            _fail(obj, "fluid output has no eligible publication from this scene owner")
        _validate_evaluated_geometry(obj)
        if not getattr(instance, "show_self", True):
            continue
        mesh = obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            if id_token(obj.original) in expected_outputs:
                captured_outputs.add(id_token(obj.original))
            if not len(mesh.loop_triangles):
                continue
            points = np.empty((len(mesh.vertices), 3), dtype=np.float32)
            mesh.vertices.foreach_get("co", points.ravel())
            transform = np.array(instance.matrix_world, dtype=np.float64)
            points = (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
            faces = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", faces.ravel())
            if np.linalg.det(transform[:3, :3]) < 0:
                faces = faces[:, [0, 2, 1]]
            slots = np.empty(len(faces), dtype=np.int32)
            mesh.loop_triangles.foreach_get("material_index", slots)
            ids = np.empty(len(faces), dtype=np.int32)
            for slot in np.unique(slots):
                material = mesh.materials[int(slot)] if int(slot) < len(mesh.materials) else None
                if int(slot) < len(obj.material_slots) and obj.material_slots[int(slot)].link == "OBJECT":
                    material = obj.material_slots[int(slot)].material
                if material is None:
                    index = 0
                else:
                    identity = material.as_pointer()
                    if identity not in material_cache:
                        material_cache[identity] = len(materials)
                        materials.append(translate_material(material))
                    index = material_cache[identity]
                ids[slots == slot] = index
            vertices.append(points)
            triangles.append(faces + vertex_offset)
            assignments.append(ids)
            vertex_offset += len(points)
        finally:
            obj.to_mesh_clear()
    validate_publications(scene, publications, for_viewport=for_viewport,
                          exact_frame=not for_viewport, view_layer=layer)
    if any(p.triangle_count and p.object_token not in captured_outputs for p in publications):
        raise SurfaceNotReadyError("Committed fluid surface was not captured from the evaluated scene")
    return SceneSnapshot(np.concatenate(vertices) if vertices else [],
                         np.concatenate(triangles) if triangles else [],
                         np.concatenate(assignments) if assignments else [],
                         tuple(materials), world, int(scene.frame_current), publications)


def camera_snapshot(depsgraph, width, height):
    scene = depsgraph.scene
    if scene.camera is None:
        _fail(scene, "a camera is required for final rendering")
    camera = scene.camera.evaluated_get(depsgraph)
    if camera.data.type not in {"PERSP", "ORTHO"}:
        _fail(camera, "P1 supports perspective and orthographic cameras")
    if camera.data.dof.use_dof:
        _fail(camera, "depth of field is not supported by P1")
    projection = camera.calc_matrix_camera(depsgraph, x=width, y=height,
                                          scale_x=scene.render.pixel_aspect_x,
                                          scale_y=scene.render.pixel_aspect_y)
    view = camera.matrix_world.normalized().inverted()
    return CameraSnapshot(np.array((projection @ view).inverted(), dtype=np.float32))
