"""Acceptance: Quick Setup is transactional, composable, and idempotent."""

from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fluidSim_warp_blender51 as addon  # noqa: E402
from fluidSim_warp_blender51.blender import setup_wizard  # noqa: E402
from fluidSim_warp_blender51.blender.scene_bridge import session  # noqa: E402


def snapshot(scene):
    return tuple(sorted((obj.name, obj.type) for obj in scene.objects))


def datablock_identities():
    names = ("objects", "collections", "meshes", "pointclouds", "materials", "node_groups")
    return {
        name: tuple(sorted(item.as_pointer() for item in getattr(bpy.data, name)))
        for name in names
        if hasattr(bpy.data, name)
    }


def source_state(scene, source, material):
    nodes = material.node_tree.nodes if material.node_tree else ()
    return {
        "ids": datablock_identities(),
        "objects": snapshot(scene),
        "role": source.fluidsim.role,
        "location": tuple(source.location),
        "display_type": source.display_type,
        "collections": tuple(sorted(coll.as_pointer() for coll in source.users_collection)),
        "materials": tuple(slot.as_pointer() for slot in source.data.materials),
        "node_labels": tuple(sorted((node.name, node.label) for node in nodes)),
        "domain": scene.fluidsim.domain_object,
        "output_collection_name": scene.fluidsim.output_collection_name,
        "setup_id": scene.get(setup_wizard.SCENE_SETUP_ID),
    }


def output_state(obj, material):
    return {
        "ids": datablock_identities(),
        "material_slots": tuple(slot.as_pointer() for slot in obj.data.materials),
        "material_nodes": tuple(
            sorted((node.name, node.label) for node in material.node_tree.nodes)
        ),
        "object_properties": tuple(sorted((key, str(value)) for key, value in obj.items())),
        "display_type": obj.display_type,
        "color": tuple(obj.color),
        "modifiers": tuple(
            (
                modifier.name,
                modifier.type,
                modifier.node_group.as_pointer() if getattr(modifier, "node_group", None) else 0,
                tuple(sorted((key, str(value)) for key, value in modifier.items())),
            )
            for modifier in obj.modifiers
        ),
    }


def main():
    addon.register()
    scene = bpy.context.scene
    source = bpy.context.active_object
    assert source is not None and source.type == "MESH"
    source.location = (1.25, -0.5, 2.0)
    source["artist_setup_sentinel"] = "preserve"
    material = bpy.data.materials.new("Artist Setup Sentinel")
    material.use_nodes = True
    principled = next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )
    principled.label = "USER NODE EDIT"
    source.data.materials.append(material)

    baseline = source_state(scene, source, material)
    for fault_stage in (
        "after_source_tagging",
        "after_domain_fitting",
        "after_output_setup",
    ):
        helper_name = {
            "after_source_tagging": "_tag_source_mesh",
            "after_domain_fitting": "_fit_domain",
            "after_output_setup": "_capacity_summary",
        }[fault_stage]
        original_helper = getattr(setup_wizard, helper_name)

        def fail_after_helper(*args, _original=original_helper,
                              _stage=fault_stage, **kwargs):
            _original(*args, **kwargs)
            raise RuntimeError(f"injected setup failure: {_stage}")

        setattr(setup_wizard, helper_name, fail_after_helper)
        try:
            setup_wizard.setup_selected(bpy.context, "QUICK_LIQUID")
        except RuntimeError as exc:
            assert str(exc) == f"injected setup failure: {fault_stage}", str(exc)
        else:
            raise AssertionError(f"failure was not injected at {fault_stage}")
        finally:
            setattr(setup_wizard, helper_name, original_helper)
        assert source_state(scene, source, material) == baseline, fault_stage
        assert source["artist_setup_sentinel"] == "preserve"

    assert bpy.ops.fluidsim.setup_scene(start_preview=False) == {"FINISHED"}
    first = snapshot(scene)
    assert scene.fluidsim.domain_object is not None
    assert any(obj.fluidsim.role == "SOURCE" for obj in scene.objects)
    assert len(tuple(obj for obj in scene.objects if obj.get("fluidsim_kind") == "OUTPUT")) == 6
    assert not scene.fluidsim.live
    assert not session().active

    surface = bpy.data.objects["FluidSim Surface"]
    output_material = bpy.data.materials.new("Artist Output Sentinel")
    output_material.use_nodes = True
    output_material.node_tree.nodes.new("ShaderNodeValue").label = "USER NODE EDIT"
    surface.data.materials.clear()
    surface.data.materials.append(output_material)
    surface.modifiers.new("Artist Non-Nodes Sentinel", "BEVEL")
    output_baseline = output_state(surface, output_material)

    original_capacity_summary = setup_wizard._capacity_summary
    callback_sentinel = None

    def fail_after_outputs(scene):
        nonlocal callback_sentinel
        original_capacity_summary(scene)
        callback_sentinel = bpy.data.materials.new("Setup Callback Sentinel")
        scene["setup_callback_sentinel"] = "preserve"
        raise RuntimeError("injected rerun failure after output setup")

    setup_wizard._capacity_summary = fail_after_outputs
    try:
        setup_wizard.setup_selected(bpy.context, "QUICK_LIQUID")
    except RuntimeError as exc:
        assert str(exc) == "injected rerun failure after output setup", str(exc)
    else:
        raise AssertionError("configured-scene output failure was not injected")
    finally:
        setup_wizard._capacity_summary = original_capacity_summary
    assert output_state(surface, output_material) == output_baseline
    assert callback_sentinel is not None
    assert bpy.data.materials.get(callback_sentinel.name) is callback_sentinel
    assert scene["setup_callback_sentinel"] == "preserve"

    assert bpy.ops.fluidsim.setup_scene(start_preview=False) == {"FINISHED"}
    assert snapshot(scene) == first
    assert len(tuple(obj for obj in scene.objects if obj.get("fluidsim_kind") == "OUTPUT")) == 6

    preview_result = bpy.ops.fluidsim.setup_scene(start_preview=True)
    assert preview_result == {"FINISHED"}
    assert snapshot(scene) == first
    if scene.fluidsim.status.startswith("Scene configured; preview could not start:"):
        # Background Blender has no playable window. The composited operator
        # must retain the valid setup and leave it paused for a later GUI start.
        assert not scene.fluidsim.live
    else:
        before = session().solver.frame
        scene.frame_set(scene.frame_start + 1)
        assert session().solver.frame > before
        assert scene.fluidsim.stat_particles > 0

    addon.unregister()
    print("QUICK-SETUP-IDEMPOTENCE-PASS")


if __name__ == "__main__":
    main()
