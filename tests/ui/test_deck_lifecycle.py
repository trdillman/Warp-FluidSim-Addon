"""Liquid Deck lifecycle: one registry owns every handler, and file load decides visibility.

Spec trap 3 (docs/specs/4.0/SPEC-liquid-deck.md): draw handlers, keymap items and
gizmo groups must not leak across add-on reloads, and the load-post handler
re-syncs the Deck from the scene, never the reverse.
"""

from __future__ import annotations

from types import SimpleNamespace

from deck_doubles import (
    gizmo_group_count,
    install_deck_bpy,
    keymap_item_count,
    owned_app_handlers,
    reload_deck,
)


def _scene(stored, *, deck_shown=True):
    """A scene whose raw ``fluidsim`` group holds ``stored`` (the saved keys)."""
    raw = {"fluidsim": dict(stored)} if stored is not None else {}

    def bl_system_properties_get(do_create=False):
        return raw

    return SimpleNamespace(
        name="Scene",
        fluidsim=SimpleNamespace(deck_shown=deck_shown, deck_stage="SETUP"),
        bl_system_properties_get=bl_system_properties_get,
        raw=raw,
    )


def _load(bpy, deck, scene, filepath="C:/files/saved.blend"):
    bpy.data = SimpleNamespace(filepath=filepath, scenes=(scene,))
    bpy.context.scene = scene
    deck._on_deck_load_post(filepath)


def test_register_unregister_fifty_times_leaves_no_handlers_gizmo_groups_or_keymap_items(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    for _ in range(50):
        deck.register()
        assert bpy.types.SpaceView3D.handlers
        assert gizmo_group_count(bpy) >= 3
        assert keymap_item_count(bpy) >= 2
        assert owned_app_handlers(bpy)
        deck.unregister()
        assert bpy.types.SpaceView3D.handlers == []
        assert gizmo_group_count(bpy) == 0
        assert keymap_item_count(bpy) == 0
        assert owned_app_handlers(bpy) == []
        assert bpy.registered == []
        assert bpy.app.timers.registered == []
        assert all(value == 0 for value in deck.registry.REGISTRY.counts().values())
    deck.unregister()  # a second unregister is harmless


def test_unregister_keeps_walking_past_a_failure_and_reports_it_at_the_end(monkeypatch, capsys):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON",
                           refuse={"FLUIDSIM_OT_command_palette"})
    deck = reload_deck()
    deck.register()
    failures = deck.registry.REGISTRY.unregister_all()
    assert len(failures) == 1 and "FLUIDSIM_OT_command_palette" in failures[0]
    assert bpy.types.SpaceView3D.handlers == []
    assert keymap_item_count(bpy) == 0
    assert gizmo_group_count(bpy) == 0
    assert [cls.__name__ for cls in bpy.registered] == ["FLUIDSIM_OT_command_palette"]
    out = capsys.readouterr().out
    assert "FLUIDSIM_OT_command_palette" in out
    # The failed entry is forgotten: a later register starts clean.
    assert all(value == 0 for value in deck.registry.REGISTRY.counts().values())


def test_registry_is_shared_by_key_for_other_lanes(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    registry = deck.registry.REGISTRY

    def preview_draw():
        return None

    registry.add_draw_handler("liquid_preview", bpy.types.SpaceView3D,
                              preview_draw, (), "WINDOW", "POST_VIEW")
    assert registry.has_draw_handler("liquid_preview")
    registry.remove_draw_handler("liquid_preview")
    assert not registry.has_draw_handler("liquid_preview")
    assert bpy.types.SpaceView3D.handlers == []
    registry.remove_draw_handler("liquid_preview")  # already gone: harmless


def test_load_post_handler_is_registered_by_the_deck_module(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="AUTO")
    deck = reload_deck()
    deck.register()
    try:
        names = {(fn.__module__, fn.__name__) for fn in bpy.app.handlers.load_post}
        assert ("fluidSim_warp_blender51.blender.deck", "_on_deck_load_post") in names
    finally:
        deck.unregister()


def test_auto_leaves_the_deck_off_for_a_file_saved_by_3_2(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="AUTO")
    deck = reload_deck()
    scene = _scene({"quality": 1, "live": 0})  # 3.2 never wrote the scene stamp
    _load(bpy, deck, scene)
    assert scene.fluidsim.deck_shown is False
    assert not deck.deck_visible(bpy.context, scene)


def test_auto_shows_the_deck_for_a_file_saved_by_4_0(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="AUTO")
    deck = reload_deck()
    from fluidSim_warp_blender51.blender import file_compat

    scene = _scene({file_compat.SCENE_SCHEMA_KEY: file_compat.CURRENT_SCENE_SCHEMA})
    _load(bpy, deck, scene)
    assert scene.fluidsim.deck_shown is True
    assert deck.deck_visible(bpy.context, scene)


def test_auto_respects_a_4_0_file_saved_with_the_deck_hidden(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="AUTO")
    deck = reload_deck()
    from fluidSim_warp_blender51.blender import file_compat

    scene = _scene({file_compat.SCENE_SCHEMA_KEY: file_compat.CURRENT_SCENE_SCHEMA},
                   deck_shown=False)
    _load(bpy, deck, scene)
    assert scene.fluidsim.deck_shown is False


def test_auto_shows_the_deck_in_a_new_unsaved_file(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="AUTO")
    deck = reload_deck()
    scene = _scene(None)
    _load(bpy, deck, scene, filepath="")
    assert scene.fluidsim.deck_shown is True
    assert deck.deck_visible(bpy.context, scene)


def test_on_mode_keeps_the_scene_flag_for_an_old_file(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    scene = _scene({"quality": 1})
    _load(bpy, deck, scene)
    assert scene.fluidsim.deck_shown is True
    assert deck.deck_visible(bpy.context, scene)


def test_off_mode_hides_the_deck_whatever_the_scene_says(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="OFF")
    deck = reload_deck()
    from fluidSim_warp_blender51.blender import file_compat

    scene = _scene({file_compat.SCENE_SCHEMA_KEY: file_compat.CURRENT_SCENE_SCHEMA})
    _load(bpy, deck, scene)
    assert scene.fluidsim.deck_shown is True  # the scene flag is not rewritten
    assert not deck.deck_visible(bpy.context, scene)


def test_without_add_on_preferences_the_deck_is_off(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode=None)
    deck = reload_deck()
    scene = _scene(None)
    assert deck.deck_mode(bpy.context) == "OFF"
    assert not deck.deck_visible(bpy.context, scene)


def test_save_stamps_the_scene_as_saved_by_4_0(monkeypatch):
    install_deck_bpy(monkeypatch, deck_mode="AUTO")
    reload_deck()
    from fluidSim_warp_blender51.blender import file_compat

    scene = _scene(None)
    scene.library = None
    assert not file_compat.scene_saved_by_deck_release(scene)
    assert file_compat.stamp_scene_schema(scenes=(scene,)) == 1
    assert scene.raw["fluidsim"][file_compat.SCENE_SCHEMA_KEY] == file_compat.CURRENT_SCENE_SCHEMA
    assert file_compat.scene_saved_by_deck_release(scene)
    assert file_compat.stamp_scene_schema(scenes=(scene,)) == 0  # already current


class _Region:
    def __init__(self, kind, x, y, width, height):
        self.type, self.x, self.y, self.width, self.height = kind, x, y, width, height
        self.redraws = 0

    def as_pointer(self):
        return id(self)

    def tag_redraw(self):
        self.redraws += 1


class _Context(SimpleNamespace):
    def temp_override(self, **_kwargs):
        import contextlib

        return contextlib.nullcontext()


def _routing(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    modal, bar, controls = deck.modal, deck.bar, deck.controls
    main = _Region("WINDOW", 100, 0, 800, 600)
    sidebar = _Region("UI", 800, 0, 100, 600)
    area = SimpleNamespace(type="VIEW_3D", regions=(main, sidebar))
    window = SimpleNamespace(screen=SimpleNamespace(areas=(area,)))
    context = _Context(window=window)
    bpy.context = context
    button = bar.Item((10.0, 10.0, 40.0, 40.0), "button", controls.PLAY, "Play")
    layout = bar.BarLayout(card=(0.0, 0.0, 300.0, 50.0), items=(button,))
    monkeypatch.setattr(bar, "current_layout", lambda _ctx: layout)
    status = {"value": (True, "")}
    monkeypatch.setattr(controls, "control_status", lambda _ctx, _c: status["value"])
    ran = []
    monkeypatch.setattr(controls, "run_control", lambda _ctx, control, **_kwargs: ran.append(control))
    operator = modal.FLUIDSIM_OT_deck_modal()
    operator.generation = modal._GENERATION
    operator.status = status
    operator.reports = []
    operator.report = lambda kind, message: operator.reports.append((kind, message))
    return modal, operator, context, ran


def _event(kind, x, y, value="PRESS"):
    return SimpleNamespace(type=kind, value=value, mouse_x=x, mouse_y=y)


def test_deck_events_consume_clicks_only_over_the_bar_and_never_esc(monkeypatch):
    _modal, operator, context, ran = _routing(monkeypatch)
    # A click on the Play button (region-space 20,20 -> window 120,20) runs it.
    assert operator.modal(context, _event("LEFTMOUSE", 120, 20)) == {"RUNNING_MODAL"}
    assert [c.target for c in ran] == ["screen.animation_play"]
    # Elsewhere on the card: swallowed, nothing runs.
    assert operator.modal(context, _event("LEFTMOUSE", 350, 20)) == {"RUNNING_MODAL"}
    assert len(ran) == 1
    # Off the card, over a sidebar overlapping the region, or Esc: passed through.
    assert operator.modal(context, _event("LEFTMOUSE", 500, 300)) == {"PASS_THROUGH"}
    assert operator.modal(context, _event("LEFTMOUSE", 850, 20)) == {"PASS_THROUGH"}
    assert operator.modal(context, _event("ESC", 120, 20)) == {"PASS_THROUGH"}
    assert len(ran) == 1


def test_a_release_over_the_bar_is_kept_only_when_the_press_started_there(monkeypatch):
    _modal, operator, context, ran = _routing(monkeypatch)
    # A box select that starts in the viewport and ends over the bar keeps its release.
    assert operator.modal(context, _event("LEFTMOUSE", 500, 300)) == {"PASS_THROUGH"}
    assert operator.modal(context, _event("LEFTMOUSE", 120, 20, "RELEASE")) == {"PASS_THROUGH"}
    # A press on the bar owns its release.
    assert operator.modal(context, _event("LEFTMOUSE", 120, 20)) == {"RUNNING_MODAL"}
    assert operator.modal(context, _event("LEFTMOUSE", 120, 20, "RELEASE")) == {"RUNNING_MODAL"}
    assert len(ran) == 1


def test_file_load_brings_the_event_layer_timer_forward(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    from fluidSim_warp_blender51.blender.deck.registry import REGISTRY

    bpy.app.background = False
    intervals = []
    register = bpy.app.timers.register

    def recording(fn, first_interval=0.0, persistent=False):
        intervals.append(first_interval)
        register(fn, first_interval=first_interval, persistent=persistent)

    bpy.app.timers.register = recording
    deck.modal.start(REGISTRY)
    assert intervals == [0.1]
    _load(bpy, deck, _scene(None))
    # The next tick comes at once, not up to ENSURE_SECONDS after the file opens.
    assert intervals == [0.1, 0.0]
    assert len(bpy.app.timers.registered) == 1
    REGISTRY.unregister_all()


def test_a_refused_bar_click_runs_nothing_and_says_why(monkeypatch):
    _modal, operator, context, ran = _routing(monkeypatch)
    operator.status["value"] = (False, "A Production Bake is running")
    assert operator.modal(context, _event("LEFTMOUSE", 120, 20)) == {"RUNNING_MODAL"}
    assert ran == []
    assert operator.reports == [({"WARNING"}, "Play: A Production Bake is running")]


def test_a_modal_from_an_earlier_register_retires_itself(monkeypatch):
    modal, operator, context, ran = _routing(monkeypatch)
    modal.new_generation()
    assert operator.modal(context, _event("LEFTMOUSE", 120, 20)) == {"FINISHED", "PASS_THROUGH"}
    assert ran == []


def test_file_load_restarts_the_event_layer_even_if_blender_never_cancelled_it(monkeypatch):
    import contextlib

    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    modal = deck.modal
    window = SimpleNamespace(screen=None)
    started = []
    bpy.app.background = False
    bpy.context.window_manager.windows = (window,)
    bpy.context.window_manager.modal_handler_add = lambda _op: None
    bpy.context.temp_override = lambda **_kwargs: contextlib.nullcontext()
    bpy.ops.fluidsim.deck_modal = lambda *_args: started.append(window)

    operator = modal.FLUIDSIM_OT_deck_modal()
    context = SimpleNamespace(window=window, window_manager=bpy.context.window_manager)
    assert operator.invoke(context, _event("MOUSEMOVE", 0, 0)) == {"RUNNING_MODAL", "PASS_THROUGH"}
    modal._ensure_running()
    assert started == []  # this window already has its modal

    # The file load drops the modal without calling cancel(); the next tick restarts it.
    _load(bpy, deck, _scene(None))
    modal._ensure_running()
    assert started == [window]
    assert operator.modal(context, _event("MOUSEMOVE", 0, 0)) == {"FINISHED", "PASS_THROUGH"}


def test_deck_mode_preference_defaults_to_on():
    """A fresh install shows the Deck everywhere; AUTO hid it in every 3.x file."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
              / "preferences.py").read_text(encoding="utf-8")
    call = next(node.annotation for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Call)
                and getattr(node.target, "id", "") == "deck_mode")
    default = next(kw.value.value for kw in call.keywords if kw.arg == "default")
    assert default == "ON"
