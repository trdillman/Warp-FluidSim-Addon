"""Liquid Preview: settings, handler lifecycle, payload handoff and pass math.

Pure Python: Blender's ``bpy`` and ``gpu`` are replaced by small recording
doubles, so these run in the regular regression lane. The pixel truth is
``test_liquid_preview_reference.py``, which needs a real Blender.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_operator_panel_behavior import (
    _draw_stage,
    _install_bpy_stub,
    _reload_module,
    _stage_context,
)

ROOT = Path(__file__).resolve().parents[2]
PROPERTIES = ROOT / "fluidSim_warp_blender51" / "properties.py"
PREVIEW_SETTINGS = (
    "liquid_preview_enabled",
    "liquid_preview_scale",
    "liquid_preview_radius_multiplier",
    "liquid_preview_absorption",
    "liquid_preview_whitewater",
)
HANDLER = "fluidSim_warp_blender51.blender.liquid_preview.handler"


# --- settings -------------------------------------------------------------

def _scene_group_fields():
    tree = ast.parse(PROPERTIES.read_text(encoding="utf-8"))
    group = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef)
                 and node.name == "FLUIDSIM_PG_scene")
    fields = {}
    for node in group.body:
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.annotation, ast.Call)):
            keywords = {kw.arg: kw.value for kw in node.annotation.keywords}
            fields[node.target.id] = (node.annotation.func.id, keywords)
    return fields


def _literal(node):
    return ast.literal_eval(node)


def test_preview_settings_live_on_the_scene_display_group():
    fields = _scene_group_fields()
    factories = {name: fields[name][0] for name in PREVIEW_SETTINGS}
    assert factories == {
        "liquid_preview_enabled": "BoolProperty",
        "liquid_preview_scale": "FloatProperty",
        "liquid_preview_radius_multiplier": "FloatProperty",
        "liquid_preview_absorption": "FloatVectorProperty",
        "liquid_preview_whitewater": "BoolProperty",
    }
    kw = {name: fields[name][1] for name in PREVIEW_SETTINGS}
    assert _literal(kw["liquid_preview_enabled"]["default"]) is False
    assert _literal(kw["liquid_preview_scale"]["default"]) == 0.5
    assert 0.0 < _literal(kw["liquid_preview_scale"]["min"])
    assert _literal(kw["liquid_preview_scale"]["max"]) == 1.0
    assert _literal(kw["liquid_preview_radius_multiplier"]["default"]) > 0.0
    assert _literal(kw["liquid_preview_absorption"]["subtype"]) == "COLOR"
    assert _literal(kw["liquid_preview_absorption"]["size"]) == 3
    assert _literal(kw["liquid_preview_whitewater"]["default"]) is True


def test_preview_settings_redraw_without_republishing_outputs():
    """An output refresh re-reads the solver; a display toggle must not."""
    fields = _scene_group_fields()
    updates = {name: fields[name][1]["update"].id for name in PREVIEW_SETTINGS}
    assert updates.pop("liquid_preview_enabled") == "_on_liquid_preview_changed"
    assert set(updates.values()) == {"_redraw_liquid_preview"}


def test_preview_settings_are_output_only_and_scene_owned(monkeypatch):
    _install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.blender.pending_settings import (
        SETTING_TRANSITION_REGISTRY,
        SettingTier,
    )
    from fluidSim_warp_blender51.presets.schema import SCENE_OWNED_SETTINGS

    for name in PREVIEW_SETTINGS:
        assert SETTING_TRANSITION_REGISTRY[name] is SettingTier.OUTPUT, name
        assert name in SCENE_OWNED_SETTINGS, name


def test_enable_callback_syncs_the_handler_and_contains_failures(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    handler = importlib.import_module(HANDLER)
    calls = []
    monkeypatch.setattr(handler, "sync_registration",
                        lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(handler, "tag_redraw", lambda: calls.append("redraw"))
    group = SimpleNamespace(status="")
    properties._on_liquid_preview_changed(group, SimpleNamespace(scene=None))
    assert calls == ["sync", "redraw"]

    def broken(*_args, **_kwargs):
        raise RuntimeError("no window manager")

    monkeypatch.setattr(handler, "sync_registration", broken)
    properties._on_liquid_preview_changed(group, SimpleNamespace(scene=None))
    assert "Liquid Preview" in group.status


# --- operator and Look stage -----------------------------------------------

def test_toggle_operator_flips_the_scene_setting(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    op_cls = operators.FLUIDSIM_OT_toggle_liquid_preview
    assert op_cls.bl_idname == "fluidsim.toggle_liquid_preview"
    fs = SimpleNamespace(liquid_preview_enabled=False,
                         liquid_view_mode="PARTICLES")
    context = SimpleNamespace(scene=SimpleNamespace(fluidsim=fs))
    assert op_cls.poll(context)
    assert not op_cls.poll(SimpleNamespace(scene=None))
    op = op_cls()
    assert op.execute(context) == {"FINISHED"}
    assert fs.liquid_preview_enabled is True
    assert op.execute(context) == {"FINISHED"}
    assert fs.liquid_preview_enabled is False


def _preview_scene_overrides(**overrides):
    values = {
        "liquid_preview_enabled": False, "liquid_preview_scale": 0.5,
        "liquid_preview_radius_multiplier": 2.0,
        "liquid_preview_absorption": (0.25, 0.6, 0.75),
        "liquid_preview_whitewater": True}
    values.update(overrides)
    return values


def test_look_stage_mirrors_the_toggle_on_the_scene_group(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_stages = _reload_module("fluidSim_warp_blender51.blender.ui_stages")
    context, scene_settings, _domain = _stage_context(
        scene_overrides=_preview_scene_overrides(liquid_display_mode="SURFACE"))
    look = _draw_stage(ui_stages.FLUIDSIM_STAGE_PT_look, context)
    drawn = {call["name"]: call["data"] for call in look.prop_calls}
    assert drawn["liquid_display_mode"] is scene_settings
    assert not {"liquid_preview_scale", "liquid_preview_absorption"} & set(drawn)

    context, scene_settings, _domain = _stage_context(
        scene_overrides=_preview_scene_overrides(liquid_preview_enabled=True, liquid_display_mode="PREVIEW"))
    look = _draw_stage(ui_stages.FLUIDSIM_STAGE_PT_look, context)
    drawn = {call["name"]: call["data"] for call in look.prop_calls}
    for name in set(PREVIEW_SETTINGS) - {"liquid_preview_enabled"}:
        assert drawn[name] is scene_settings, name


def test_look_stage_explains_a_preview_with_no_published_particles(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_stages = _reload_module("fluidSim_warp_blender51.blender.ui_stages")
    context, _scene, _domain = _stage_context(
        scene_overrides=_preview_scene_overrides(
            liquid_preview_enabled=True, liquid_view_mode="SURFACE"))
    look = _draw_stage(ui_stages.FLUIDSIM_STAGE_PT_look, context)
    assert not any("pick a particle view" in label["text"].lower() for label in look.labels)


# --- handler lifecycle -----------------------------------------------------

class _KeymapItems(list):
    def new(self, idname, type, value, **kwargs):
        item = SimpleNamespace(idname=idname, type=type, value=value, **kwargs)
        self.append(item)
        return item


class _Keymaps(list):
    def new(self, name, space_type="EMPTY", **_kwargs):
        keymap = SimpleNamespace(name=name, space_type=space_type,
                                 keymap_items=_KeymapItems())
        self.append(keymap)
        return keymap


def _install_handler_bpy(monkeypatch, scenes=(), *, keyconfig=True):
    draw_handlers = []

    class SpaceView3D:
        @staticmethod
        def draw_handler_add(callback, args, region, event):
            handle = object()
            draw_handlers.append((handle, callback, region, event))
            return handle

        @staticmethod
        def draw_handler_remove(handle, region):
            for entry in list(draw_handlers):
                if entry[0] is handle and entry[2] == region:
                    draw_handlers.remove(entry)
                    return
            raise ValueError("unknown draw handle")

    keymaps = _Keymaps()
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.types = SimpleNamespace(SpaceView3D=SpaceView3D)
    fake_bpy.app = SimpleNamespace(handlers=SimpleNamespace(
        load_post=[], persistent=lambda fn: fn))
    fake_bpy.data = SimpleNamespace(scenes=list(scenes))
    fake_bpy.context = SimpleNamespace(
        window_manager=SimpleNamespace(keyconfigs=SimpleNamespace(
            addon=SimpleNamespace(keymaps=keymaps) if keyconfig else None)),
        screen=SimpleNamespace(areas=[]),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    sys.modules.pop(HANDLER, None)
    handler = importlib.import_module(HANDLER)
    return handler, fake_bpy, draw_handlers, keymaps


def _scene(enabled):
    return SimpleNamespace(fluidsim=SimpleNamespace(
        **_preview_scene_overrides(liquid_preview_enabled=enabled)))


def test_mode_off_registers_no_draw_handler(monkeypatch):
    handler, _bpy, draw_handlers, _keymaps = _install_handler_bpy(
        monkeypatch, [_scene(False), SimpleNamespace(fluidsim=None)])
    handler.register()
    assert draw_handlers == []
    assert handler.is_draw_registered() is False
    handler.unregister()


def test_enable_disable_keeps_exactly_one_or_zero_handlers(monkeypatch):
    scene = _scene(False)
    handler, _bpy, draw_handlers, _keymaps = _install_handler_bpy(
        monkeypatch, [scene])
    handler.register()
    scene.fluidsim.liquid_preview_enabled = True
    handler.sync_registration()
    handler.sync_registration()
    assert len(draw_handlers) == 1
    _handle, _callback, region, event = draw_handlers[0]
    assert (region, event) == ("WINDOW", "POST_VIEW")
    scene.fluidsim.liquid_preview_enabled = False
    handler.sync_registration()
    assert draw_handlers == []
    handler.unregister()


def test_unregister_leaves_zero_handlers_keymaps_and_load_hooks(monkeypatch):
    handler, fake_bpy, draw_handlers, keymaps = _install_handler_bpy(
        monkeypatch, [_scene(True)])
    handler.register()
    assert len(draw_handlers) == 1
    items = [item for keymap in keymaps for item in keymap.keymap_items]
    assert [(i.idname, i.type, i.value) for i in items] == [
        ("fluidsim.toggle_liquid_preview", "L", "PRESS")]
    assert keymaps[0].space_type == "VIEW_3D"
    assert len(fake_bpy.app.handlers.load_post) == 1

    handler.unregister()
    handler.unregister()  # Reload Scripts may call it twice
    assert draw_handlers == []
    assert [item for keymap in keymaps for item in keymap.keymap_items] == []
    assert fake_bpy.app.handlers.load_post == []
    assert handler.is_draw_registered() is False


def test_register_without_an_addon_keyconfig_is_background_safe(monkeypatch):
    handler, _bpy, draw_handlers, _keymaps = _install_handler_bpy(
        monkeypatch, [_scene(False)], keyconfig=False)
    handler.register()
    handler.unregister()
    assert draw_handlers == []


def test_file_load_resyncs_registration(monkeypatch):
    handler, fake_bpy, draw_handlers, _keymaps = _install_handler_bpy(
        monkeypatch, [_scene(False)])
    handler.register()
    fake_bpy.data.scenes = [_scene(True)]
    for callback in fake_bpy.app.handlers.load_post:
        callback(None)
    assert len(draw_handlers) == 1
    handler.unregister()


def test_draw_failure_is_logged_once_and_never_raises(monkeypatch, capsys):
    handler, _fake_bpy, draw_handlers, _keymaps = _install_handler_bpy(
        monkeypatch, [_scene(True)])
    handler.register()
    calls = []

    def exploding_frame(_context):
        calls.append(1)
        raise RuntimeError("shader compile failed: backend said no")

    monkeypatch.setattr(handler, "_draw_frame", exploding_frame)
    _handle, callback, _region, _event = draw_handlers[0]
    assert callback() is None
    assert callback() is None
    assert calls == [1], "a failed session draws nothing afterwards"
    assert "backend said no" in handler.failure()
    assert capsys.readouterr().out.count("Liquid Preview") == 1
    handler.unregister()
    assert handler.failure() is None


def test_shader_compile_failure_sets_the_session_flag(monkeypatch, capsys):
    handler, fake_bpy, draw_handlers, _keymaps = _install_handler_bpy(
        monkeypatch, [_scene(True)])
    _install_gpu_double(monkeypatch, fail_compile=True)
    from fluidSim_warp_blender51.blender import output_publication

    _publish(output_publication, _payload(particles=np.ones((4, 3), np.float32)))
    fake_bpy.context.scene = fake_bpy.data.scenes[0]
    fake_bpy.context.region = SimpleNamespace(width=64, height=32)
    fake_bpy.context.region_data = _region_data()
    handler.register()
    _handle, callback, _region, _event = draw_handlers[0]
    callback()
    callback()
    assert "compile" in handler.failure()
    assert capsys.readouterr().out.count("Liquid Preview") == 1
    handler.unregister()


# --- payload handoff -------------------------------------------------------

def _payload(*, particles=None, whitewater=None, types_=None,
             particles_visible=True, whitewater_visible=True):
    from fluidSim_warp_blender51.blender import output_publication

    particles = (np.zeros((0, 3), np.float32) if particles is None
                 else particles)
    whitewater = (np.zeros((0, 3), np.float32) if whitewater is None
                  else whitewater)
    attributes = {} if types_ is None else {"type": types_}
    return output_publication.PublicationPayload(
        voxel_size=.1, publish_surface=False, surface_visible=False,
        surface_attribute_mode="NONE",
        surface_verts=np.zeros((0, 3), np.float32),
        surface_tris=np.zeros((0, 3), np.int32), surface_attributes={},
        particles_visible=particles_visible, particle_attribute_mode="NONE",
        particle_radius=.025, particles=particles, particle_attributes={},
        whitewater_visible=whitewater_visible,
        whitewater_output_mode="COMBINED", whitewater_attribute_mode="NONE",
        whitewater_radius=.015, whitewater=whitewater,
        whitewater_attributes=attributes)


def _publish(output_publication, payload, scene=None, monkeypatch=None, mutations=None):
    """Run publish_payload with every Blender ID mutation stubbed out."""
    patched = {
        "_require_bpy": lambda: None,
        "output_object": lambda *_a: None,
        "_clear_points": lambda *_a: None,
        "_ensure_point_object": lambda *_a: SimpleNamespace(),
        "_supports_pointcloud": lambda: True,
        "_swap_points": lambda *_a, **_k: None,
        "_ensure_material": lambda *_a: None,
        "_publish_whitewater_points": lambda *_a, **_k: None,
        "_hide_and_clear_output": lambda *_a: None,
        "_sync_render_inputs": lambda _scene: None,
    }
    if mutations is not None:
        patched.update({
            "_ensure_object": lambda _scene, name, *_a: mutations.append(name) or SimpleNamespace(),
            "_ensure_point_object": lambda _scene, name, *_a: mutations.append(name) or SimpleNamespace(),
            "_swap_mesh": lambda *_a, **_k: None,
        })
    saved = {name: getattr(output_publication, name) for name in patched}
    sys.modules.setdefault(
        "fluidSim_warp_blender51.blender.materials",
        SimpleNamespace(**{name: None for name in (
            "ensure_surface_material", "ensure_particle_material",
            "ensure_whitewater_material", "ensure_foam_material",
            "ensure_bubble_material", "ensure_spray_material")}))
    try:
        for name, value in patched.items():
            setattr(output_publication, name, value)
        output_publication.publish_payload(
            scene if scene is not None else SimpleNamespace(frame_current=7),
            payload)
    finally:
        for name, value in saved.items():
            setattr(output_publication, name, value)
    return output_publication.latest_published_points()


def test_publication_hands_over_an_immutable_read_only_record():
    from fluidSim_warp_blender51.blender import output_publication

    particles = np.arange(12, dtype=np.float32).reshape(4, 3)
    whitewater = np.ones((2, 3), np.float32)
    types_ = np.array([0, 2], np.int32)
    record = _publish(output_publication, _payload(
        particles=particles, whitewater=whitewater, types_=types_))

    assert record.particle_count == 4
    assert record.whitewater_count == 2
    assert record.scene_frame == 7
    assert record.particle_radius == pytest.approx(.025)
    assert np.shares_memory(record.positions, particles), "no copy"
    for array in (record.positions, record.whitewater,
                  record.whitewater_types):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.frame_id = 0
    assert particles.flags.writeable, "the publisher's array is untouched"

    again = _publish(output_publication, _payload(particles=particles))
    assert again.frame_id > record.frame_id
    assert again.whitewater_types is None


def test_hidden_outputs_hand_over_empty_points():
    from fluidSim_warp_blender51.blender import output_publication

    record = _publish(output_publication, _payload(
        particles=np.ones((3, 3), np.float32), particles_visible=False,
        whitewater=np.ones((3, 3), np.float32), whitewater_visible=False))
    assert record.particle_count == 0
    assert record.whitewater_count == 0
    output_publication.clear_published_points()
    assert output_publication.latest_published_points() is None


# --- GPU double: upload policy and readback pin ----------------------------

def _install_gpu_double(monkeypatch, *, fail_compile=False):
    log = SimpleNamespace(vertbufs=[], batches=[], textures=[], draws=[])
    # Models Blender 5.2: creating a GPUOffScreen leaves no framebuffer bound,
    # and bind() contexts restore whatever was active when they were entered.
    destination = None

    class GPUVertFormat:
        def attr_add(self, **kwargs):
            return None

    class GPUVertBuf:
        def __init__(self, fmt, length):
            self.length = length
            self.filled = {}
            log.vertbufs.append(self)

        def attr_fill(self, name, data):
            data = np.asarray(data)
            assert len(data) == self.length
            self.filled[name] = data.copy()

    class GPUBatch:
        def __init__(self, type, buf):
            self.type = type
            self.buf = buf
            log.batches.append(self)

        def draw(self, shader):
            log.draws.append(("draw", self.type, log.active))

        def draw_range(self, shader, elem_start=0, elem_count=0):
            log.draws.append(("range", self.type, elem_count))
            log.targets.append((self.type, log.active))

    class _Shader:
        def __getattr__(self, name):
            return lambda *a, **k: None

    def create_from_info(_info):
        if fail_compile:
            raise RuntimeError("shader compile failed on this backend")
        return _Shader()

    class _Info:
        def __getattr__(self, name):
            return lambda *a, **k: None

    class GPUTexture:
        def __init__(self, size, format="RGBA8", data=None, **_kwargs):
            log.textures.append((tuple(size), format))

        def filter_mode(self, _value):
            return None

    class _Bound:
        def __init__(self, target):
            self.target = target

        def __enter__(self):
            self.previous = log.active
            log.active = self.target
            return self

        def __exit__(self, *_exc):
            log.active = self.previous
            return False

    class _Framebuffer:
        def clear(self, **_kwargs):
            return None

        def bind(self):
            return _Bound(self)

    class GPUOffScreen:
        def __init__(self, width, height, format="RGBA8"):
            self.width, self.height = width, height
            self.texture_color = GPUTexture((width, height), format)
            log.active = None

        def bind(self):
            return _Bound(self)

        def clear(self, **_kwargs):  # stands in for its bound framebuffer
            return None

        def free(self):
            return None

    class GPUFrameBuffer(_Framebuffer):
        def __init__(self, depth_slot=None, color_slots=None):
            self.depth_slot, self.color_slots = depth_slot, color_slots

    class GPUUniformBuf:
        def __init__(self, data):
            log.ubo_updates = 1

        def update(self, data):
            log.ubo_updates += 1

    state = SimpleNamespace(
        blend_get=lambda: "NONE", blend_set=lambda _v: None,
        depth_test_get=lambda: "NONE", depth_test_set=lambda _v: None,
        depth_mask_get=lambda: False, depth_mask_set=lambda _v: None,
        program_point_size_set=lambda _v: None,
        viewport_get=lambda: (0, 0, 64, 32),
        viewport_set=lambda *_a: None,
        active_framebuffer_get=lambda: log.active,
    )
    fake_gpu = types.ModuleType("gpu")
    fake_gpu.types = SimpleNamespace(
        GPUVertFormat=GPUVertFormat, GPUVertBuf=GPUVertBuf, GPUBatch=GPUBatch,
        GPUTexture=GPUTexture, GPUOffScreen=GPUOffScreen,
        GPUFrameBuffer=GPUFrameBuffer,
        GPUShaderCreateInfo=_Info, GPUStageInterfaceInfo=lambda _n: _Info(),
        GPUUniformBuf=GPUUniformBuf,
        Buffer=lambda fmt, dims, data=None: data)
    fake_gpu.shader = SimpleNamespace(create_from_info=create_from_info)
    fake_gpu.state = state
    monkeypatch.setitem(sys.modules, "gpu", fake_gpu)
    destination = _Framebuffer()
    log.destination = destination
    log.active = destination
    log.targets = []
    return log


def _region_data():
    view = np.eye(4)
    proj = np.array([[1.5, 0, 0, 0], [0, 2.0, 0, 0],
                     [0, 0, -1.0002, -0.20002], [0, 0, -1.0, 0]])
    return SimpleNamespace(view_matrix=view, window_matrix=proj,
                           is_perspective=True)


def test_upload_happens_once_per_frame_id_not_per_redraw(monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.blender.liquid_preview import passes

    log = _install_gpu_double(monkeypatch)
    renderer = passes.LiquidPreviewRenderer()
    first = _publish(output_publication, _payload(
        particles=np.ones((5000, 3), np.float32),
        whitewater=np.ones((10, 3), np.float32)))
    settings = passes.PreviewSettings()
    for _ in range(5):
        renderer.draw(first, view_matrix=np.eye(4),
                      projection_matrix=_region_data().window_matrix,
                      viewport=(0, 0, 640, 360), settings=settings)
    assert renderer.uploads == 1
    vbo = next(buf for buf in log.vertbufs
               if "pos" in buf.filled and "wtype" not in buf.filled)
    assert vbo.length == output_publication._point_capacity_for(5000)
    assert ("range", "POINTS", 5000) in log.draws

    second = _publish(output_publication, _payload(
        particles=np.ones((5100, 3), np.float32)))
    renderer.draw(second, view_matrix=np.eye(4),
                  projection_matrix=_region_data().window_matrix,
                  viewport=(0, 0, 640, 360), settings=settings)
    assert renderer.uploads == 2
    renderer.free()


def test_liquid_draws_land_in_the_framebuffer_bound_at_entry(monkeypatch):
    """Blender 5.2 unbinds the viewport when a GPUOffScreen is created, so
    the intermediate targets must not be offscreens or the composite is lost."""
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.blender.liquid_preview import passes

    log = _install_gpu_double(monkeypatch)
    renderer = passes.LiquidPreviewRenderer()
    record = _publish(output_publication, _payload(
        particles=np.ones((500, 3), np.float32),
        whitewater=np.ones((10, 3), np.float32)))
    renderer.draw(record, view_matrix=np.eye(4),
                  projection_matrix=_region_data().window_matrix,
                  viewport=(0, 0, 640, 360), settings=passes.PreviewSettings())
    composite = [d for d in log.draws if d[:2] == ("draw", "TRI_STRIP")][-1]
    assert composite[2] is log.destination
    assert log.targets[-1] == ("POINTS", log.destination)  # whitewater
    assert log.active is log.destination
    renderer.free()


class _CountingSolver:
    """A solver without ``particles``, so HostFrame takes the counted path."""

    frame = 3
    settings = SimpleNamespace(voxel_size=.1)

    def particle_positions(self):
        return np.ones((256, 3), np.float32)


def _live_scene(preview_enabled):
    fs = SimpleNamespace(
        show_surface=False, surface_attribute_mode="NONE",
        show_particles=True, particle_attribute_mode="NONE",
        particle_display_fraction=1.0, particle_display_scale=1.0,
        show_whitewater=False, whitewater_attribute_mode="NONE",
        whitewater_display_scale=1.0, whitewater_display_fraction=1.0,
        whitewater_output_mode="COMBINED",
        liquid_preview_enabled=preview_enabled)
    domain = SimpleNamespace(surface_enabled=False, whitewater_enabled=False)
    return SimpleNamespace(fluidsim=fs, frame_current=3), domain


def _one_published_frame(monkeypatch, output_publication, preview_enabled):
    from fluidSim_warp_blender51.blender.liquid_preview import passes
    from fluidSim_warp_blender51.sim.frame_output import (
        HostFrame,
        host_output_operation_counts,
    )

    scene, domain = _live_scene(preview_enabled)
    monkeypatch.setattr(output_publication, "_domain_settings",
                        lambda _scene: domain)
    monkeypatch.setattr(output_publication, "_particle_values",
                        lambda *_a, **_k: None)
    before = host_output_operation_counts()
    output_publication.output_cost_counts(reset=True)
    payload = output_publication.prepare_live_payload(
        scene, HostFrame(_CountingSolver()), publish_surface=False)
    record = _publish(output_publication, payload, scene=scene)
    if preview_enabled:
        renderer = passes.LiquidPreviewRenderer()
        for _ in range(3):
            renderer.draw(record, view_matrix=np.eye(4),
                          projection_matrix=_region_data().window_matrix,
                          viewport=(0, 0, 320, 180),
                          settings=passes.PreviewSettings())
        renderer.free()
    after = host_output_operation_counts()
    costs = output_publication.output_cost_counts()["live_adaptation"]
    return {
        "device_readbacks": after["device_readbacks"]
        - before["device_readbacks"],
        "host_frame_acquisitions": costs["host_frame_acquisitions"],
        "live_device_readbacks": costs["device_readbacks"],
    }


def test_liquid_preview_adds_no_device_readback(monkeypatch):
    from fluidSim_warp_blender51.blender import output_publication

    _install_gpu_double(monkeypatch)
    off = _one_published_frame(monkeypatch, output_publication, False)
    on = _one_published_frame(monkeypatch, output_publication, True)
    assert off["device_readbacks"] == 1, off
    assert on == off


# --- pure pass math --------------------------------------------------------

def test_filter_radius_matches_the_screen_space_formula():
    from fluidSim_warp_blender51.blender.liquid_preview.passes import (
        filter_radius_pixels,
    )

    tan_half = math.tan(math.radians(25.0))
    expected = math.ceil(540 * 0.05 / (2 * 2.0 * tan_half))
    assert filter_radius_pixels(0.05, 540, 2.0, tan_half) == expected
    assert filter_radius_pixels(0.05, 540, -2.0, tan_half) == expected, "|z|"
    near = filter_radius_pixels(0.05, 540, 0.5, tan_half, max_radius=10_000)
    far = filter_radius_pixels(0.05, 540, 8.0, tan_half)
    assert near > expected > far >= 1
    assert filter_radius_pixels(0.05, 540, 0.01, tan_half, max_radius=16) == 16
    assert filter_radius_pixels(0.05, 540, 0.0, tan_half) == 0
    assert filter_radius_pixels(0.0, 540, 2.0, tan_half) == 0


def test_offscreen_size_scales_and_never_collapses():
    from fluidSim_warp_blender51.blender.liquid_preview.passes import (
        offscreen_size,
    )

    assert offscreen_size(1920, 1080, 0.5) == (960, 540)
    assert offscreen_size(1920, 1080, 1.0) == (1920, 1080)
    assert offscreen_size(3, 1, 0.1) == (1, 1)
    assert offscreen_size(1920, 1080, 5.0) == (1920, 1080), "clamped to 1"


def test_absorption_follows_beer_lambert():
    from fluidSim_warp_blender51.blender.liquid_preview.passes import (
        absorption_coefficients,
    )

    coefficients = absorption_coefficients((1.0, 0.5, 0.0))
    assert coefficients[0] == pytest.approx(0.0)
    assert math.exp(-coefficients[1] * 1.0) == pytest.approx(0.5)
    assert math.isfinite(coefficients[2]) and coefficients[2] > 0


def test_blue_noise_tile_is_a_deterministic_rank_permutation():
    from fluidSim_warp_blender51.blender.liquid_preview.passes import (
        blue_noise_tile,
    )

    tile = blue_noise_tile()
    assert tile.shape == (64, 64)
    assert tile.dtype == np.float32
    ranks = np.sort((tile * 4096).round().astype(int).ravel())
    assert np.array_equal(ranks, np.arange(4096))
    assert np.array_equal(tile, blue_noise_tile())
    # Blue noise: little low-frequency energy compared with white noise.
    def low_frequency_share(values):
        spectrum = np.abs(np.fft.fft2(values - values.mean())) ** 2
        freq = np.hypot(*np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64)))
        return spectrum[(freq > 0) & (freq < 0.1)].sum() / spectrum.sum()

    white = np.random.default_rng(0).random((64, 64))
    assert low_frequency_share(tile) < 0.5 * low_frequency_share(white)


# --- reference helpers -----------------------------------------------------

def test_ssim_is_one_for_identical_images_and_drops_with_damage():
    from fluidSim_warp_blender51.blender.liquid_preview.reference import ssim

    rng = np.random.default_rng(1)
    image = (rng.random((60, 80, 4)) * 255).astype(np.uint8)
    assert ssim(image, image) == pytest.approx(1.0)
    noisy = np.clip(image + rng.normal(0, 40, image.shape), 0, 255)
    assert ssim(image, noisy.astype(np.uint8)) < 0.9
    with pytest.raises(ValueError):
        ssim(image, image[:-1])


def test_png_round_trip(tmp_path):
    from fluidSim_warp_blender51.blender.liquid_preview.reference import (
        read_png,
        write_png,
    )

    image = (np.random.default_rng(2).random((9, 7, 4)) * 255).astype(np.uint8)
    path = tmp_path / "round.png"
    write_png(path, image)
    assert np.array_equal(read_png(path), image)


def test_synthetic_payload_is_deterministic():
    from fluidSim_warp_blender51.blender.liquid_preview.reference import (
        synthetic_points,
    )

    first = synthetic_points(20_000, seed=4)
    second = synthetic_points(20_000, seed=4)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
    positions, whitewater, types_ = first
    assert positions.shape == (20_000, 3) and positions.dtype == np.float32
    assert set(np.unique(types_)) <= {0, 1, 2}
    assert len(whitewater) == len(types_)


def test_display_modes_change_only_liquid_viewport_objects(monkeypatch):
    from fluidSim_warp_blender51.blender import liquid_display, output_publication

    class Obj:
        def __init__(self):
            self.hidden = None
            self.hide_render = False
        def hide_set(self, value):
            self.hidden = value
        def hide_get(self):
            return self.hidden

    surface, particles, whitewater = Obj(), Obj(), Obj()
    objects = {output_publication.SURFACE_NAME: surface,
               output_publication.PARTICLES_NAME: particles,
               output_publication.WHITEWATER_NAME: whitewater}
    monkeypatch.setattr(output_publication, "output_object", lambda _s, name: objects.get(name))
    monkeypatch.setattr(output_publication, "_domain_settings",
                        lambda _scene: SimpleNamespace(surface_enabled=True))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        liquid_display_mode="PREVIEW", show_surface=True, show_particles=True))
    for mode, expected in (("PREVIEW", True), ("SURFACE", False), ("OFF", True)):
        scene.fluidsim.liquid_display_mode = mode
        liquid_display.apply_viewport(scene)
        assert surface.hidden is expected
        assert particles.hidden is True
        assert whitewater.hidden is None
        assert not surface.hide_render and not particles.hide_render


def test_cached_hidden_particle_record_can_supply_preview():
    from fluidSim_warp_blender51.blender import output_publication
    scene = SimpleNamespace(frame_current=7, fluidsim=SimpleNamespace(liquid_preview_enabled=False))
    payload = _payload(particles=np.ones((3, 3), np.float32), particles_visible=False)
    output_publication._record_published_points(scene, payload, preserve_preview_points=True)
    assert output_publication.latest_published_points().particle_count == 3


def test_saved_mode_wins_and_legacy_preview_is_migrated(monkeypatch):
    from fluidSim_warp_blender51.blender import liquid_display
    monkeypatch.setattr(liquid_display, "apply_viewport", lambda _scene: None)

    class Settings:
        def __init__(self, mode, preview, saved):
            self.liquid_display_mode = mode
            self.liquid_preview_enabled = preview
            self.saved = saved
        def is_property_set(self, name):
            return self.saved

    legacy = Settings("SURFACE", True, False)
    liquid_display.restore_scene(SimpleNamespace(fluidsim=legacy))
    assert legacy.liquid_display_mode == "PREVIEW"
    assert legacy.liquid_preview_enabled is True
    saved = Settings("OFF", True, True)
    liquid_display.restore_scene(SimpleNamespace(fluidsim=saved))
    assert saved.liquid_display_mode == "OFF"
    assert saved.liquid_preview_enabled is False


def test_restricted_registration_defers_scene_restore(monkeypatch):
    handler, fake_bpy, draw_handlers, _keymaps = _install_handler_bpy(monkeypatch)
    pending = []
    fake_bpy.app.timers = SimpleNamespace(
        is_registered=lambda fn: fn in pending,
        register=lambda fn, **_kw: pending.append(fn),
        unregister=lambda fn: pending.remove(fn))
    fake_bpy.data = SimpleNamespace()  # Blender registration restricts bpy.data
    handler.register()
    assert pending == [handler._restore_after_registration]
    fake_bpy.data = SimpleNamespace(scenes=[_scene(True)])
    pending.pop()()
    assert handler.is_draw_registered()
    handler.unregister()
    assert not draw_handlers


def test_display_applies_to_each_scene_view_layer(monkeypatch):
    from fluidSim_warp_blender51.blender import liquid_display, output_publication
    class Obj:
        name = "liquid"
        def __init__(self):
            self.calls = []
        def hide_set(self, value, *, view_layer):
            self.calls.append((value, view_layer))
        def hide_get(self, *, view_layer):
            return next((v for v, l in reversed(self.calls) if l is view_layer), None)
    surface, particles = Obj(), Obj()
    layers = [SimpleNamespace(objects={"liquid"}), SimpleNamespace(objects={"liquid"})]
    scene = SimpleNamespace(fluidsim=SimpleNamespace(liquid_display_mode="OFF"), view_layers=layers)
    monkeypatch.setattr(output_publication, "output_object", lambda _scene, name:
                        surface if name == output_publication.SURFACE_NAME else particles)
    liquid_display.apply_viewport(scene)
    assert surface.calls == [(True, layers[0]), (True, layers[1])]
    assert particles.calls == [(True, layers[0]), (True, layers[1])]


def test_display_writes_view_layer_visibility_only_when_it_changes(monkeypatch):
    """hide_set re-tags the depsgraph even for an equal value; publication runs
    per frame and inside Alembic export's frame updates, which crashed Blender
    5.2.0 (deg_check_base_in_depsgraph), so unchanged flags must not be rewritten."""
    from fluidSim_warp_blender51.blender import liquid_display, output_publication
    class Obj:
        name = "liquid"
        def __init__(self):
            self.calls = []
        def hide_set(self, value, *, view_layer):
            self.calls.append((value, view_layer))
        def hide_get(self, *, view_layer):
            return next((v for v, l in reversed(self.calls) if l is view_layer), None)
    surface, particles = Obj(), Obj()
    layers = [SimpleNamespace(objects={"liquid"}), SimpleNamespace(objects={"liquid"})]
    scene = SimpleNamespace(fluidsim=SimpleNamespace(liquid_display_mode="OFF"), view_layers=layers)
    monkeypatch.setattr(output_publication, "output_object", lambda _scene, name:
                        surface if name == output_publication.SURFACE_NAME else particles)
    monkeypatch.setattr(output_publication, "_domain_settings",
                        lambda _scene: SimpleNamespace(surface_enabled=True))
    liquid_display.apply_viewport(scene)
    assert len(surface.calls) == len(particles.calls) == 2
    for _ in range(3):
        liquid_display.apply_viewport(scene)
    assert len(surface.calls) == len(particles.calls) == 2
    scene.fluidsim.liquid_display_mode = "SURFACE"
    liquid_display.apply_viewport(scene)
    assert [c[0] for c in surface.calls[2:]] == [False, False]


def test_export_hold_keeps_display_policy_from_rehiding_outputs(monkeypatch):
    """Exporters un-hide the objects they sample; per-frame publication must not
    re-apply the display policy until they release (it crashed Blender 5.2.0)."""
    from fluidSim_warp_blender51.blender import liquid_display, output_publication
    class Obj:
        name = "liquid"
        def __init__(self):
            self.calls = []
        def hide_set(self, value, *, view_layer):
            self.calls.append((value, view_layer))
        def hide_get(self, *, view_layer):
            return next((v for v, l in reversed(self.calls) if l is view_layer), None)
    surface, particles = Obj(), Obj()
    layers = [SimpleNamespace(objects={"liquid"})]
    scene = SimpleNamespace(fluidsim=SimpleNamespace(liquid_display_mode="OFF"), view_layers=layers)
    monkeypatch.setattr(output_publication, "output_object", lambda _scene, name:
                        surface if name == output_publication.SURFACE_NAME else particles)
    liquid_display.hold_viewport()
    try:
        liquid_display.apply_viewport(scene)
        assert surface.calls == [] and particles.calls == []
    finally:
        liquid_display.release_viewport()
    liquid_display.release_viewport()  # an unbalanced release never goes negative
    liquid_display.apply_viewport(scene)
    assert len(surface.calls) == len(particles.calls) == 1


def test_view_layer_write_is_deferred_out_of_the_frame_update(monkeypatch):
    """Changing a base's hide flag inside the frame-change handler crashed
    Blender 5.2.0 (deg_check_base_in_depsgraph), so a changed flag is written
    from a timer instead, and an unchanged one schedules nothing."""
    from fluidSim_warp_blender51.blender import liquid_display, output_publication
    class Obj:
        name = "liquid"
        def __init__(self):
            self.calls = []
        def hide_set(self, value, *, view_layer):
            self.calls.append((value, view_layer))
        def hide_get(self, *, view_layer):
            return next((v for v, l in reversed(self.calls) if l is view_layer), False)
    surface, particles = Obj(), Obj()
    layers = [SimpleNamespace(objects={"liquid"})]
    scene = SimpleNamespace(name="Scene", view_layers=layers,
                            fluidsim=SimpleNamespace(liquid_display_mode="OFF"))
    timers = []
    fake_bpy = SimpleNamespace(
        app=SimpleNamespace(timers=SimpleNamespace(
            is_registered=lambda fn: fn in timers,
            register=lambda fn, **_kw: timers.append(fn))),
        data=SimpleNamespace(scenes=[scene]))
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setattr(output_publication, "output_object", lambda _scene, name:
                        surface if name == output_publication.SURFACE_NAME else particles)
    liquid_display.apply_viewport(scene)
    liquid_display.apply_viewport(scene)
    assert surface.calls == [] and particles.calls == []
    assert timers == [liquid_display._flush_pending]
    assert timers.pop()() is None
    assert surface.calls == [(True, layers[0])] and particles.calls == [(True, layers[0])]
    liquid_display.apply_viewport(scene)  # nothing changed: nothing scheduled
    assert timers == []


def test_particle_only_legacy_scene_migrates_to_preview(monkeypatch):
    from fluidSim_warp_blender51.blender import liquid_display, scene_access
    scene, domain = _live_scene(False)
    scene.fluidsim.liquid_display_mode = "SURFACE"
    scene.fluidsim.is_property_set = lambda _name: False
    monkeypatch.setattr(scene_access, "active_domain_settings", lambda _scene: domain)
    monkeypatch.setattr(liquid_display, "apply_viewport", lambda _scene: None)
    liquid_display.restore_scene(scene)
    assert scene.fluidsim.liquid_display_mode == "PREVIEW"
    assert scene.fluidsim.liquid_preview_enabled


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("surface_enabled", [False, True])
def test_viewport_geometry_is_independent_of_render_flags(monkeypatch, cached, surface_enabled):
    from fluidSim_warp_blender51.blender import output_publication as output
    scene, domain = _live_scene(False)
    scene.fluidsim.liquid_display_mode = "SURFACE"
    scene.fluidsim.show_particles = False
    domain.surface_enabled = surface_enabled
    monkeypatch.setattr(output, "_domain_settings", lambda _scene: domain)
    monkeypatch.setattr(output, "_particle_values", lambda *_a, **_k: None)
    verts = np.ones((3, 3), np.float32)
    tris = np.array([[0, 1, 2]], np.int32)

    class Host:
        solver = _CountingSolver()
        surface_mesh = (verts, tris)
        def positions_strided(self, _step):
            return np.ones((4, 3), np.float32)
        def output_attributes(self, *_a, **_k):
            return {}

    if cached:
        payload = output.prepare_cache_payload(scene, {
            "positions": np.ones((4, 3), np.float32),
            "surface_verts": verts, "surface_tris": tris})
    else:
        payload = output.prepare_live_payload(scene, Host())
    assert not payload.surface_visible and not payload.particles_visible
    assert payload.viewport_surface is surface_enabled
    assert payload.viewport_particles is not surface_enabled
    assert payload.retain_cache_geometry is cached
    if surface_enabled:
        assert len(payload.surface_verts) == 3
    else:
        assert len(payload.particles) == 4
    mutations = []
    _publish(output, payload, scene=scene, mutations=mutations)
    expected = [output.SURFACE_NAME] if surface_enabled or cached else []
    if cached or not surface_enabled:
        expected.append(output.PARTICLES_NAME)
    assert mutations == expected


@pytest.mark.parametrize("mode,failed", [("SURFACE", False), ("PREVIEW", True)])
def test_no_mesh_fallback_unhides_particles_without_enabling_render(monkeypatch, mode, failed):
    from fluidSim_warp_blender51.blender import liquid_display
    from fluidSim_warp_blender51.blender import output_publication as output
    from fluidSim_warp_blender51.blender.liquid_preview import handler
    scene, domain = _live_scene(mode == "PREVIEW")
    scene.fluidsim.liquid_display_mode = mode
    scene.fluidsim.show_particles = False
    monkeypatch.setattr(output, "_domain_settings", lambda _scene: domain)
    monkeypatch.setattr(handler, "failure", lambda: "shader failed" if failed else None)

    class Obj:
        hide_viewport = True
        hide_render = True
        def hide_set(self, hidden):
            self.hidden = hidden
        def hide_get(self):
            return getattr(self, "hidden", None)

    surface, particles = Obj(), Obj()
    monkeypatch.setattr(output, "output_object", lambda _scene, name:
                        surface if name == output.SURFACE_NAME else particles)
    liquid_display.apply_viewport(scene)
    assert surface.hidden
    assert not particles.hidden and not particles.hide_viewport
    assert particles.hide_render


@pytest.mark.parametrize("initial_mode", ["PREVIEW", "OFF"])
@pytest.mark.parametrize("has_mesh", [False, True])
def test_paused_cache_keeps_both_representations_for_mode_changes(monkeypatch, initial_mode, has_mesh):
    from fluidSim_warp_blender51.blender import liquid_display
    from fluidSim_warp_blender51.blender import output_publication as output
    scene, domain = _live_scene(initial_mode == "PREVIEW")
    scene.fluidsim.liquid_display_mode = initial_mode
    scene.fluidsim.show_particles = False
    domain.surface_enabled = True
    monkeypatch.setattr(output, "_domain_settings", lambda _scene: domain)
    verts = np.ones((3 if has_mesh else 0, 3), np.float32)
    payload = output.prepare_cache_payload(scene, {
        "positions": np.ones((4, 3), np.float32), "surface_verts": verts,
        "surface_tris": np.array([[0, 1, 2]], np.int32) if has_mesh else np.zeros((0, 3), np.int32)})
    assert not payload.surface_visible and not payload.particles_visible
    mutations = []
    _publish(output, payload, scene=scene, mutations=mutations)
    assert mutations == [output.SURFACE_NAME, output.PARTICLES_NAME]

    class Obj:
        hide_viewport = True
        hide_render = True
        def hide_set(self, hidden):
            self.hidden = hidden
        def hide_get(self):
            return getattr(self, "hidden", None)
    surface, particles = Obj(), Obj()
    surface.data = SimpleNamespace(vertices=payload.surface_verts)
    monkeypatch.setattr(output, "output_object", lambda _scene, name:
                        surface if name == output.SURFACE_NAME else particles)
    scene.fluidsim.liquid_display_mode = "SURFACE"
    liquid_display.apply_viewport(scene)  # metadata only, no solver/cache read
    assert surface.hidden is not has_mesh
    assert particles.hidden is has_mesh
    assert surface.hide_render and particles.hide_render


def test_renderer_fallback_refresh_is_deferred_deduplicated_and_removed(monkeypatch):
    from fluidSim_warp_blender51.blender import liquid_display, pending_settings
    handler, fake_bpy, _draw_handlers, _keymaps = _install_handler_bpy(monkeypatch)
    pending, refreshed = [], []
    fake_bpy.app.timers = SimpleNamespace(
        is_registered=lambda fn: fn in pending,
        register=lambda fn, **_kw: pending.append(fn),
        unregister=lambda fn: pending.remove(fn))
    scene = _scene(True)
    fake_bpy.data.scenes = [scene, _scene(False)]
    monkeypatch.setattr(liquid_display, "apply_viewport", lambda _scene: None)
    monkeypatch.setattr(pending_settings, "request_scene_update",
                        lambda scene, action: refreshed.append((scene, action)))
    liquid_display.show_surface_fallback(scene)
    liquid_display.show_surface_fallback(scene)
    assert pending == [liquid_display._refresh_fallback_outputs]
    assert not refreshed
    pending.pop()()
    assert refreshed == [(scene, pending_settings.SceneUpdate.OUTPUT)]
    liquid_display.show_surface_fallback(scene)
    handler.unregister()
    assert not pending
