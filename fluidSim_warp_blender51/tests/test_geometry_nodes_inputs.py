"""Geometry Nodes modifier input assignment must never fail silently."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fluidsim_testpkg.blender import geometry_nodes as gn


def _group(*socket_names: str, name: str = "FluidSim Test Group"):
    items = [
        SimpleNamespace(
            item_type="SOCKET",
            in_out="INPUT",
            name=socket_name,
            identifier=f"Socket_{index}",
        )
        for index, socket_name in enumerate(socket_names, start=2)
    ]
    items.append(
        SimpleNamespace(item_type="SOCKET", in_out="OUTPUT", name="Geometry", identifier="Socket_0")
    )
    return SimpleNamespace(name=name, interface=SimpleNamespace(items_tree=items))


class _TypedModifier:
    """Blender 5.2 style: ``modifier.properties.inputs.<identifier>.value``."""

    def __init__(self, *identifiers: str):
        self.name = "Typed"
        self.properties = SimpleNamespace(
            inputs=SimpleNamespace(**{key: SimpleNamespace(value=None) for key in identifiers})
        )
        self.id_properties: dict = {}

    def __setitem__(self, key, value):
        self.id_properties[key] = value


class _IdPropertyModifier(dict):
    """Blender 5.1 style: no typed inputs, ID properties keyed by identifier."""

    name = "IdProps"


class _BrokenModifier:
    """Neither API path accepts the value."""

    name = "Broken"
    properties = SimpleNamespace(inputs=SimpleNamespace())

    def __setitem__(self, key, value):
        raise TypeError(f"property {key!r} is read-only")


def test_typed_inputs_path_assigns_value():
    group = _group(gn.MATERIAL_SOCKET)
    modifier = _TypedModifier("Socket_2")
    material = object()

    gn._set_modifier_input(modifier, group, gn.MATERIAL_SOCKET, material)

    assert modifier.properties.inputs.Socket_2.value is material
    assert modifier.id_properties == {}
    assert gn._modifier_input_value(modifier, group, gn.MATERIAL_SOCKET) is material


def test_id_property_fallback_assigns_value():
    group = _group(gn.MATERIAL_SOCKET, gn.RADIUS_SOCKET)
    modifier = _IdPropertyModifier()

    gn._set_modifier_input(modifier, group, gn.RADIUS_SOCKET, 2.5)

    assert modifier == {"Socket_3": 2.5}
    assert gn._modifier_input_value(modifier, group, gn.RADIUS_SOCKET) == 2.5


def test_both_paths_failing_is_logged_and_raised(capsys):
    group = _group(gn.RADIUS_SOCKET, name="FluidSim Foam [P4 v1]")

    with pytest.raises(gn.GeometryNodesInputError) as raised:
        gn._set_modifier_input(_BrokenModifier(), group, gn.RADIUS_SOCKET, 1.0)

    message = str(raised.value)
    assert "'Radius Scale'" in message
    assert "'Broken'" in message
    assert "FluidSim Foam [P4 v1]" in message
    assert "typed inputs API failed (AttributeError" in message
    assert "ID-property fallback failed (TypeError" in message
    assert isinstance(raised.value.__cause__, TypeError)
    assert f"FluidSim: {message}" in capsys.readouterr().out


def test_undeclared_required_socket_is_loud(capsys):
    group = _group(gn.MATERIAL_SOCKET)

    with pytest.raises(gn.GeometryNodesInputError, match="does not declare"):
        gn._set_modifier_input(_TypedModifier("Socket_2"), group, gn.RADIUS_SOCKET, 1.0)
    assert "FluidSim: cannot set Geometry Nodes input 'Radius Scale'" in capsys.readouterr().out


def test_undeclared_optional_socket_is_skipped(capsys):
    group = _group(gn.MATERIAL_SOCKET)
    modifier = _TypedModifier("Socket_2")

    gn._set_modifier_input(modifier, group, gn.RADIUS_SOCKET, 1.0, required=False)

    assert modifier.properties.inputs.Socket_2.value is None
    assert capsys.readouterr().out == ""


def test_only_the_surface_output_may_omit_radius():
    assert gn._radius_required("FluidSim Surface") is False
    for output_name in gn.GROUP_BY_OUTPUT:
        if output_name != "FluidSim Surface":
            assert gn._radius_required(output_name) is True


class _FakeObject(dict):
    def __init__(self, name, modifier_factory):
        super().__init__()
        self.name = name
        self.data = SimpleNamespace(materials=["surface-material"])
        self._factory = modifier_factory
        self._modifiers: dict = {}
        self.modifiers = SimpleNamespace(get=self._modifiers.get, new=self._new)

    def _new(self, *, name, type):
        modifier = self._factory()
        modifier.name = name
        modifier.node_group = None
        self._modifiers[name] = modifier
        return modifier


def _tagged(group):
    group.get = {gn._REVISION_TAG: gn.ASSET_REVISION}.get
    return group


def test_attach_surface_without_radius_socket_succeeds():
    group = _tagged(_group(gn.MATERIAL_SOCKET, name=gn.SURFACE_GROUP))
    obj = _FakeObject("FluidSim Surface", lambda: _TypedModifier("Socket_2"))

    ok, warning = gn._attach_group(
        obj, group, "FluidSim Surface", force=False, radius_scale=1.0
    )

    assert (ok, warning) == (True, "")
    modifier = obj.modifiers.get(gn.MODIFIER_BY_OUTPUT["FluidSim Surface"])
    assert modifier.properties.inputs.Socket_2.value == "surface-material"


def test_attach_propagates_input_failure_to_installer_callers():
    group = _tagged(_group(gn.MATERIAL_SOCKET, gn.RADIUS_SOCKET, name=gn.FOAM_GROUP))
    obj = _FakeObject("FluidSim Foam", _BrokenModifier)

    with pytest.raises(gn.GeometryNodesInputError, match="'Material'"):
        gn._attach_group(obj, group, "FluidSim Foam", force=False, radius_scale=1.0)
