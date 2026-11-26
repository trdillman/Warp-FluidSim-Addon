"""Pure translation semantics on small socket records; Blender owns runtime coverage."""
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender.fspt_scene import (
    UnsupportedSceneError,
    translate_material,
    world_color,
)

RGB = (0.2, 0.4, 0.6)


def socket(name, value=None, source=None):
    links = [] if source is None else [SimpleNamespace(from_node=source, is_valid=True)]
    return SimpleNamespace(name=name, default_value=value, is_linked=bool(links), links=links)


def node(kind, **values):
    return SimpleNamespace(bl_idname="ShaderNode" + kind, name=kind, mute=False,
                           is_active_output=False,
                           inputs={k: socket(k, v) for k, v in values.items()})


def graph(shader, *, world=False, disconnected=()):
    output = node("OutputWorld" if world else "OutputMaterial", Volume=None)
    if not world:
        output.inputs["Displacement"] = socket("Displacement")
    output.inputs["Surface"] = socket("Surface", source=shader)
    output.is_active_output = True
    output.target = "ALL"
    tree = SimpleNamespace(nodes=[*disconnected, shader, output])
    return SimpleNamespace(name="Validation graph", use_nodes=True, node_tree=tree), output


def diffuse():
    return node("BsdfDiffuse", Color=(*RGB, 1), Roughness=0.0, Normal=(0, 0, 0))


def test_diffuse_color_and_disconnected_texture():
    material, _ = graph(diffuse(), disconnected=[node("TexNoise")])
    translated = translate_material(material)
    assert translated.kind == "diffuse"
    assert translated.color == pytest.approx(RGB)


def test_emission_scales_each_channel():
    material, _ = graph(node("Emission", Color=(*RGB, 1), Strength=2.5))
    translated = translate_material(material)
    assert translated.kind == "emission"
    assert translated.emission == pytest.approx((0.5, 1.0, 1.5))


def test_glass_preserves_tint_and_ior():
    material, _ = graph(node("BsdfGlass", Color=(*RGB, 1), Roughness=0.0, IOR=1.52,
                             Normal=(0, 0, 0)))
    translated = translate_material(material)
    assert translated.kind == "glass"
    assert translated.ior == pytest.approx(1.52)
    assert translated.color == pytest.approx(RGB)


def test_linked_color_is_rejected():
    shader = diffuse()
    noise = node("TexNoise")
    shader.inputs["Color"] = socket("Color", (*RGB, 1), source=noise)
    material, _ = graph(shader, disconnected=[noise])
    with pytest.raises(UnsupportedSceneError):
        translate_material(material)


def test_rough_glass_is_rejected():
    material, _ = graph(node("BsdfGlass", Color=(*RGB, 1), Roughness=0.25, IOR=1.52,
                             Normal=(0, 0, 0)))
    with pytest.raises(UnsupportedSceneError):
        translate_material(material)


@pytest.mark.parametrize(("destination", "kind"), [
    ("Volume", "VolumePrincipled"), ("Displacement", "Displacement"),
])
def test_linked_auxiliary_output_is_rejected(destination, kind):
    source = node(kind)
    material, output = graph(diffuse(), disconnected=[source])
    output.inputs[destination] = socket(destination, source=source)
    with pytest.raises(UnsupportedSceneError):
        translate_material(material)


def test_principled_with_default_specular_is_rejected():
    shader = node("BsdfPrincipled", **{"Base Color": (*RGB, 1), "Roughness": 0.5,
                  "Metallic": 0.0, "IOR": 1.5, "Specular IOR Level": 0.5,
                  "Transmission Weight": 0.0})
    material, _ = graph(shader)
    with pytest.raises(UnsupportedSceneError):
        translate_material(material)


def test_world_scales_each_background_channel():
    world, _ = graph(node("Background", Color=(*RGB, 1), Strength=3.0), world=True)
    assert world_color(world) == pytest.approx((0.6, 1.2, 1.8))
