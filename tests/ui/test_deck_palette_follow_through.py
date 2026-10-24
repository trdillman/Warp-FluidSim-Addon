"""Command palette follow-through: greyed rows, refusals, exact inline edits and write feedback.

The palette operator's ``invoke``/``modal`` and its draw callback are driven
against the recording ``bpy`` double, with fake ``fluidsim`` operators whose
``poll`` and result each test chooses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck
from settings_doubles import full_settings, split_settings


class _Area:
    type = "VIEW_3D"

    def tag_redraw(self):
        return None


def _event(kind, unicode="", value="PRESS"):
    return SimpleNamespace(type=kind, value=value, unicode=unicode, mouse_region_x=500,
                           mouse_region_y=400, ctrl=False, shift=False, alt=False, oskey=False)


class _FakeOp:
    """A ``bpy.ops.fluidsim.<name>`` stand-in: chosen poll, recorded calls, chosen result."""

    def __init__(self, bpy, idname, *, can_run=True, reason="", result=None, error=""):
        self.calls = []
        self.can_run = can_run
        self.result = result or {"FINISHED"}
        self.error = error
        identifier = "FLUIDSIM_OT_" + idname.split(".", 1)[1]
        self._rna = SimpleNamespace(identifier=identifier, name=idname, description="")
        fake = self

        class _Cls:
            @classmethod
            def poll(cls, _context):
                if reason:
                    cls.poll_message_set(reason)
                return fake.can_run

        setattr(bpy.types, identifier, _Cls)

    def get_rna_type(self):
        return self._rna

    def poll(self):
        return self.can_run

    def __call__(self, _call_context="INVOKE_DEFAULT", **properties):
        self.calls.append(properties)
        if self.error:
            raise RuntimeError(self.error)
        return self.result


@pytest.fixture
def deck(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    deck.bpy = bpy
    deck.undo_pushes = []
    bpy.ops.ed = SimpleNamespace(undo_push=lambda message="": deck.undo_pushes.append(message))
    deck.palette.bar._FLASHES.clear()
    monkeypatch.setattr(deck.palette, "_bake_hold_message", lambda: "", raising=False)
    return deck


def _fake_op(deck, idname, **kwargs):
    op = _FakeOp(deck.bpy, idname, **kwargs)
    setattr(deck.bpy.ops.fluidsim, idname.split(".", 1)[1], op)
    return op


def _scene(**values):
    fs = full_settings(resolution_divisions=64, flip_ratio=0.000123456,
                       viscosity_enabled=False, surface_tension_enabled=False,
                       product_preset="NONE", **values)
    return SimpleNamespace(fluidsim=split_settings(fs), objects=[])


def _open(deck, monkeypatch, entries, scene=None):
    scene = scene if scene is not None else _scene()
    monkeypatch.setattr(deck.palette, "build_index", lambda _scene: list(entries))
    context = SimpleNamespace(
        area=_Area(), region=SimpleNamespace(width=1000, height=800), scene=scene,
        window_manager=SimpleNamespace(modal_handler_add=lambda _op: None))
    deck.bpy.context.scene = scene
    operator = deck.palette.FLUIDSIM_OT_command_palette()
    operator.reports = []
    operator.report = lambda kinds, message: operator.reports.append((set(kinds), message))
    assert operator.invoke(context, _event("K")) == {"RUNNING_MODAL"}
    return operator, context


def _setting(deck, scene, name):
    return {e.target: e for e in deck.palette.setting_entries(scene)}[name]


# Defect 1: operator rows are greyed with their poll reason ------------------------

def test_operator_row_status_reports_the_poll_reason(deck):
    _fake_op(deck, "fluidsim.bake_start", can_run=False,
             reason="A production bake is already active")
    _fake_op(deck, "fluidsim.initialize")
    blocked = deck.palette.Entry("operator", "fluidsim.bake_start", "Bake")
    ready = deck.palette.Entry("operator", "fluidsim.initialize", "Initialize")
    context = SimpleNamespace(scene=_scene())
    assert deck.palette.row_status(context, blocked) == (
        False, "A production bake is already active")
    assert deck.palette.row_status(context, ready) == (True, "")


def test_enter_on_an_unavailable_operator_row_explains_and_keeps_the_palette_open(
        deck, monkeypatch):
    op = _fake_op(deck, "fluidsim.bake_start", can_run=False,
                  reason="A production bake is already active")
    entry = deck.palette.Entry("operator", "fluidsim.bake_start", "Bake")
    operator, context = _open(deck, monkeypatch, [entry])
    assert operator.modal(context, _event("RET")) == {"RUNNING_MODAL"}
    assert op.calls == []
    assert any("A production bake is already active" in m for _k, m in operator.reports)
    assert deck.registry.REGISTRY.has_draw_handler("palette")


def test_an_operator_that_refuses_after_enter_keeps_the_palette_open(deck, monkeypatch):
    _fake_op(deck, "fluidsim.export_vdb", error="Error: No baked cache to export")
    entry = deck.palette.Entry("operator", "fluidsim.export_vdb", "Export VDB")
    operator, context = _open(deck, monkeypatch, [entry])
    assert operator.modal(context, _event("RET")) == {"RUNNING_MODAL"}
    assert any("No baked cache" in m for _k, m in operator.reports)
    assert deck.registry.REGISTRY.has_draw_handler("palette")


def test_an_operator_that_runs_closes_the_palette(deck, monkeypatch):
    op = _fake_op(deck, "fluidsim.initialize")
    entry = deck.palette.Entry("operator", "fluidsim.initialize", "Initialize")
    operator, context = _open(deck, monkeypatch, [entry])
    assert operator.modal(context, _event("RET")) == {"FINISHED"}
    assert op.calls == [{}]
    assert not deck.registry.REGISTRY.has_draw_handler("palette")


def test_unavailable_rows_draw_dimmed_with_their_reason(deck, monkeypatch):
    _fake_op(deck, "fluidsim.bake_start", can_run=False, reason="Warp is not ready")
    _fake_op(deck, "fluidsim.initialize")
    entries = [deck.palette.Entry("operator", "fluidsim.initialize", "Initialize"),
               deck.palette.Entry("operator", "fluidsim.bake_start", "Bake")]
    operator, context = _open(deck, monkeypatch, entries)
    drawn = []
    monkeypatch.setattr(deck.palette, "_draw", SimpleNamespace(
        rounded_rect=lambda *a: None, text_width=lambda s, size: len(str(s)) * size * 0.5,
        text=lambda x, y, s, size, colour: drawn.append((str(s), tuple(colour)))))
    palette = deck.tokens.current_tokens(None)
    monkeypatch.setattr(deck.palette.tokens, "current_tokens", lambda _c: palette)
    monkeypatch.setattr(deck.palette.bar, "_ui_scale", lambda _c: 1.0)
    deck.bpy.context = SimpleNamespace(region=context.region, scene=context.scene)
    deck.palette._draw_palette(operator)
    colours = dict(drawn)
    dim = tuple(deck.tokens.rgba(palette.text_dim))
    assert colours["Initialize"] == tuple(deck.tokens.rgba(palette.text))
    assert colours["Bake"] == dim
    assert colours["Warp is not ready"] == dim


# Defect 6: demo rows whose .blend is missing are greyed ------------------------------

def test_demo_rows_with_a_missing_file_are_greyed_and_not_loaded(deck, monkeypatch):
    load = _fake_op(deck, "fluidsim.load_demo")
    from fluidSim_warp_blender51.blender import demo_gallery

    monkeypatch.setattr(demo_gallery, "available_demo_ids", lambda: ("dam_break",))
    entries = [deck.palette.Entry("demo", "crown_splash", "Crown Splash"),
               deck.palette.Entry("demo", "dam_break", "Dam Break")]
    operator, context = _open(deck, monkeypatch, entries)
    assert deck.palette.row_status(context, entries[0], operator.missing_demos) == (
        False, "(file missing)")
    assert deck.palette.row_status(context, entries[1], operator.missing_demos) == (True, "")
    assert operator.modal(context, _event("RET")) == {"RUNNING_MODAL"}
    assert load.calls == []
    operator.modal(context, _event("DOWN_ARROW"))
    assert operator.modal(context, _event("RET")) == {"FINISHED"}
    assert load.calls == [{"demo_id": "dam_break"}]


# Defect 3: a refused preset apply writes nothing --------------------------------------

def test_preset_is_not_written_when_apply_would_be_refused(deck, monkeypatch):
    apply = _fake_op(deck, "fluidsim.apply_product_preset", can_run=False,
                     reason="Production Bake owns this scene")
    entry = deck.palette.Entry("preset", "builtin:calm", "Calm Water")
    scene = _scene()
    operator, context = _open(deck, monkeypatch, [entry], scene)
    assert operator.modal(context, _event("RET")) == {"RUNNING_MODAL"}
    assert scene.fluidsim.product_preset == "NONE"
    assert apply.calls == []
    assert any("Production Bake owns this scene" in m for _k, m in operator.reports)


def test_preset_is_written_then_applied_when_apply_can_run(deck, monkeypatch):
    apply = _fake_op(deck, "fluidsim.apply_product_preset")
    entry = deck.palette.Entry("preset", "builtin:calm", "Calm Water")
    scene = _scene()
    operator, context = _open(deck, monkeypatch, [entry], scene)
    assert operator.modal(context, _event("RET")) == {"FINISHED"}
    assert scene.fluidsim.product_preset == "builtin:calm"
    assert apply.calls == [{}]


# Defect 2: inline edit starts from the exact value, selected ------------------------

def test_inline_edit_starts_from_the_exact_value_and_typing_replaces_it(deck, monkeypatch):
    scene = _scene()
    flip = _setting(deck, scene, "flip_ratio")
    operator, context = _open(deck, monkeypatch, [flip], scene)
    assert operator.modal(context, _event("RET")) == {"RUNNING_MODAL"}
    assert operator.editing is flip
    assert operator.edit_text == "0.000123456"  # not the .4g "0.0001235"
    assert operator.edit_selected
    operator.modal(context, _event("FIVE", unicode="5"))
    assert operator.edit_text == "5" and not operator.edit_selected
    operator.modal(context, _event("ZERO", unicode="0"))
    assert operator.edit_text == "50"


def test_backspace_edits_the_exact_value_instead_of_replacing_it(deck, monkeypatch):
    scene = _scene()
    flip = _setting(deck, scene, "flip_ratio")
    operator, context = _open(deck, monkeypatch, [flip], scene)
    operator.modal(context, _event("RET"))
    operator.modal(context, _event("BACK_SPACE"))
    assert operator.edit_text == "0.00012345" and not operator.edit_selected
    operator.modal(context, _event("SEVEN", unicode="7"))
    assert operator.edit_text == "0.000123457"


def test_enter_without_typing_writes_nothing(deck, monkeypatch):
    scene = _scene()
    flip = _setting(deck, scene, "flip_ratio")
    operator, context = _open(deck, monkeypatch, [flip], scene)
    operator.modal(context, _event("RET"))
    assert operator.modal(context, _event("RET")) == {"FINISHED"}
    assert scene.fluidsim.domain_object.fluidsim_domain.flip_ratio == 0.000123456
    assert deck.undo_pushes == []


def test_exact_value_text_round_trips_a_blender_float():
    from fluidSim_warp_blender51.blender.deck import palette

    assert palette.exact_value_text(0.1, "FLOAT") == "0.1"
    assert palette.exact_value_text(0.10000000149011612, "FLOAT") == "0.1"
    assert palette.exact_value_text(1234567.0, "FLOAT") == "1234567"
    assert palette.exact_value_text(96, "INT") == "96"


# Defects 4 and 5: write feedback follows the field's tier, and is undoable -----------

def test_live_tier_write_flashes_live_and_pushes_undo(deck):
    scene = _scene()
    viscosity = _setting(deck, scene, "viscosity_enabled")
    message = deck.palette.commit_inline(scene, viscosity, "")
    assert scene.fluidsim.domain_object.fluidsim_domain.viscosity_enabled is True
    assert deck.palette.bar._FLASHES  # the LIVE chip
    assert message == ""
    assert deck.undo_pushes == [f"Set {viscosity.label}"]


@pytest.mark.parametrize("name", ["surface_tension_enabled", "resolution_divisions"])
def test_reset_and_reinitialize_tier_writes_are_staged_not_live(deck, name):
    scene = _scene()
    entry = _setting(deck, scene, name)
    message = deck.palette.commit_inline(scene, entry, "80")
    assert not deck.palette.bar._FLASHES
    assert message and "LIVE" not in message
    assert len(deck.undo_pushes) == 1


def test_write_during_a_production_bake_is_held_not_live(deck, monkeypatch):
    hold = "A Production Bake is running; changes apply when it ends"
    monkeypatch.setattr(deck.palette, "_bake_hold_message", lambda: hold, raising=False)
    scene = _scene()
    viscosity = _setting(deck, scene, "viscosity_enabled")
    assert deck.palette.commit_inline(scene, viscosity, "") == hold
    assert not deck.palette.bar._FLASHES


def test_undo_push_is_skipped_quietly_without_an_undo_operator(deck):
    del deck.bpy.ops.ed
    scene = _scene()
    deck.palette.commit_inline(scene, _setting(deck, scene, "flip_ratio"), "0.5")
    assert scene.fluidsim.domain_object.fluidsim_domain.flip_ratio == pytest.approx(0.5)


def test_boolean_row_reports_the_staged_message_from_the_palette(deck, monkeypatch):
    scene = _scene()
    entry = _setting(deck, scene, "surface_tension_enabled")
    operator, context = _open(deck, monkeypatch, [entry], scene)
    assert operator.modal(context, _event("RET")) == {"FINISHED"}
    assert scene.fluidsim.domain_object.fluidsim_domain.surface_tension_enabled is True
    assert operator.reports and "LIVE" not in operator.reports[0][1]
