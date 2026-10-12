"""Animated and moved Custom Effect values reach the running solver (B66).

F-curve/driver evaluation and controller-object motion change effect values
without an RNA update callback, so before this fix they stayed frozen at the
values captured at admission. Preview now re-reads runtime-bindable values
each step; a Production Bake keeps its admitted values (decision D2).
"""
from __future__ import annotations

from types import SimpleNamespace

from test_operator_panel_behavior import _install_bpy_stub, _reload_module
from fluidSim_warp_blender51.sim.effects.builtins import builtin_effect_definitions
from fluidSim_warp_blender51.sim.effects.runtime import EffectRuntimeManager
from fluidSim_warp_blender51.sim.effects.schema import EffectInstanceV2
from fluidSim_warp_blender51.sim.effects.source import parameter_runtime_inputs


class _Matrix(list):
    """A 4x4 row-major matrix with Blender's ``translation`` accessor."""

    def __init__(self, x, y, z):
        super().__init__([[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y],
                          [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]])

    @property
    def translation(self):
        return (self[0][3], self[1][3], self[2][3])


def _vortex():
    return next(item for item in builtin_effect_definitions()
                if item.effect_id == "fluidsim.vortex_force")


def _parameter_pgs(definition, controller):
    items = []
    for parameter in definition.parameters:
        item = SimpleNamespace(key=parameter.key, kind=parameter.kind.value,
                               float_value=0.0, vector_value=(0.0, 0.0, 0.0),
                               object_value=None)
        if parameter.kind.value == "FLOAT":
            item.float_value = float(parameter.default)
        elif parameter.kind.value == "VECTOR":
            item.vector_value = tuple(parameter.default)
        elif parameter.kind.value == "OBJECT":
            item.object_value = controller
        items.append(item)
    return items


def _setup(monkeypatch, *, bake_active=False, animated=True):
    _install_bpy_stub(monkeypatch)
    effects = _reload_module("fluidSim_warp_blender51.blender.effects")
    from fluidSim_warp_blender51.blender import production_bake

    monkeypatch.setattr(production_bake, "any_bake_active", lambda: bake_active)
    definition = _vortex()
    controller = SimpleNamespace(name="Vortex Controller", matrix_world=_Matrix(1.0, 2.0, 3.0))
    pg = SimpleNamespace(instance_id="vortex-1", parameters=_parameter_pgs(definition, controller))
    stack = SimpleNamespace(items=[pg], applied_signature="admitted")
    scene = SimpleNamespace(
        fluidsim_effects=stack,
        animation_data=SimpleNamespace(action="EffectsAction", drivers=(), nla_tracks=())
        if animated else None)
    values ={item.key: effects._parameter_value(item) for item in pg.parameters}
    controllers = {"controller": effects._controller_snapshot(pg.parameters[0])}
    admitted = EffectInstanceV2("vortex-1", definition, parameter_values=values,
                                controller_bindings=controllers)
    manager = EffectRuntimeManager(device="cpu")
    kernel = definition.kernels[0]
    compiled = SimpleNamespace(
        success=True, instance=admitted,
        kernels=(SimpleNamespace(kernel_definition=kernel,
                                 generated=SimpleNamespace(kernel_index=0)),))
    assert manager.swap_candidate(SimpleNamespace(success=True, compiled_effects=(compiled,)))
    session = SimpleNamespace(solver=SimpleNamespace(features=SimpleNamespace(effects=manager)))
    published = []
    monkeypatch.setattr(effects, "scene_effect_instances",
                        lambda scene: (manager.applied_instances()[0],))
    monkeypatch.setattr(effects, "_publish_applied_state",
                        lambda scene, instances: published.append(instances))
    return effects, scene, pg, controller, manager, session, published


def _running_inputs(manager):
    (instance,) = manager.applied_instances()
    keys = instance.definition.kernels[0].parameter_keys
    return dict(zip(keys, parameter_runtime_inputs(instance, 0), strict=True))


def test_keyframed_parameter_follows_the_preview_frame(monkeypatch):
    effects, scene, pg, _controller, manager, session, published = _setup(monkeypatch)
    strength = next(item for item in pg.parameters if item.key == "strength")
    strength.float_value = 9.5  # F-curve evaluation: no update callback fires

    assert effects.sync_animated_effect_bindings(scene, session) is True

    assert _running_inputs(manager)["strength"] == 9.5
    assert len(published) == 1


def test_moved_controller_follows_the_preview_frame(monkeypatch):
    effects, scene, _pg, controller, manager, session, _published = _setup(monkeypatch)
    assert _running_inputs(manager)["controller"] == (1.0, 2.0, 3.0)
    controller.matrix_world = _Matrix(-4.0, 0.5, 2.0)

    assert effects.sync_animated_effect_bindings(scene, session) is True

    assert _running_inputs(manager)["controller"] == (-4.0, 0.5, 2.0)


def test_unchanged_values_do_not_rebind(monkeypatch):
    effects, scene, _pg, _controller, manager, session, published = _setup(monkeypatch)
    calls = []
    original = manager.update_runtime_bindings
    monkeypatch.setattr(manager, "update_runtime_bindings",
                        lambda instances: calls.append(1) or original(instances))

    assert effects.sync_animated_effect_bindings(scene, session) is False
    assert calls == [] and published == []


def test_no_animation_and_no_controller_reads_nothing(monkeypatch):
    effects, scene, pg, _controller, manager, session, _published = _setup(
        monkeypatch, animated=False)
    for item in pg.parameters:
        item.object_value = None
    monkeypatch.setattr(effects, "_parameter_value",
                        lambda item: (_ for _ in ()).throw(AssertionError("read")))

    assert effects.sync_animated_effect_bindings(scene, session) is False


def test_production_bake_keeps_admitted_values(monkeypatch):
    # D2: a bake keeps the values its cache identity was recorded with; the
    # keyframe-following part of B66 for bakes is deferred.
    effects, scene, pg, controller, manager, session, published = _setup(
        monkeypatch, bake_active=True)
    next(item for item in pg.parameters if item.key == "strength").float_value = 9.5
    controller.matrix_world = _Matrix(-4.0, 0.5, 2.0)

    assert effects.sync_animated_effect_bindings(scene, session) is False

    assert _running_inputs(manager)["strength"] == 4.0
    assert _running_inputs(manager)["controller"] == (1.0, 2.0, 3.0)
    assert published == []
