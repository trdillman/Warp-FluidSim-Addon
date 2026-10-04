"""Acceptance: Quick Setup is transactional, composable, and idempotent."""

from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[2]
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
    def modifier_properties(modifier):
        try:
            return tuple(sorted((key, str(value)) for key, value in modifier.items()))
        except TypeError:
            return ()

    return {
        "ids": datablock_identities(),
        "material_slots": tuple(slot.as_pointer() for slot in obj.data.materials),
        "material_slot_names": tuple(slot.name for slot in obj.data.materials),
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
                modifier_properties(modifier),
            )
            for modifier in obj.modifiers
        ),
    }


def assert_state_equal(before, after, label):
    if before == after:
        return
    print(f"SETUP-ROLLBACK-DELTA {label}")
    for key in sorted(before.keys() | after.keys()):
        if before.get(key) == after.get(key):
            continue
        if key == "ids":
            for registry in sorted(before[key].keys() | after[key].keys()):
                old = set(before[key].get(registry, ()))
                new = set(after[key].get(registry, ()))
                if old != new:
                    added = sorted(new - old)[:8]
                    added_details = []
                    for item in getattr(bpy.data, registry):
                        if item.as_pointer() in added:
                            added_details.append(
                                (
                                    item.name,
                                    item.get(setup_wizard.OWNER_TAG),
                                    item.get(setup_wizard.OWNER_ID_TAG),
                                    item.get(setup_wizard.KIND_TAG),
                                )
                            )
                    print(
                        f"  ids.{registry}: added={added} details={added_details} "
                        f"removed={sorted(old - new)[:8]}"
                    )
        else:
            print(f"  {key}: before={before.get(key)!r} after={after.get(key)!r}")
    raise AssertionError(label)


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
        assert_state_equal(baseline, source_state(scene, source, material), fault_stage)
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
    assert callback_sentinel is not None
    expected_output_state = dict(output_baseline)
    expected_ids = dict(output_baseline["ids"])
    expected_ids["materials"] = tuple(sorted((
        *expected_ids["materials"], callback_sentinel.as_pointer(),
    )))
    expected_output_state["ids"] = expected_ids
    assert_state_equal(
        expected_output_state,
        output_state(surface, output_material),
        "configured output rollback",
    )
    assert bpy.data.materials.get(callback_sentinel.name) is callback_sentinel
    assert scene["setup_callback_sentinel"] == "preserve"

    assert bpy.ops.fluidsim.setup_scene(start_preview=False) == {"FINISHED"}
    assert snapshot(scene) == first
    assert len(tuple(obj for obj in scene.objects if obj.get("fluidsim_kind") == "OUTPUT")) == 6

    addon.unregister()
    print("QUICK-SETUP-IDEMPOTENCE-PASS")


if __name__ == "__main__":
    main()
