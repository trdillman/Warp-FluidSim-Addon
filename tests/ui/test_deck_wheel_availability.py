"""Wheel and bar availability: a control that cannot run says why instead of doing nothing.

Clicking a wheel slot whose operator's poll failed used to close the wheel and
run nothing, with no message, and every slot looked clickable. These checks pin
the fix: the status carries the operator's own poll message, a refused slot
keeps the wheel open and reports, hover tips explain each slot, and Preset and
Demo open the palette filtered to that list.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck

DOMAIN_REASON = "Select a mesh object to use as the FluidSim domain"


class _FakeOp:
    def __init__(self, identifier, ok, ran):
        self.identifier, self.ok, self.ran = identifier, ok, ran

    def get_rna_type(self):
        return SimpleNamespace(identifier=self.identifier)

    def poll(self):
        return self.ok

    def __call__(self, call_context, **properties):
        self.ran.append((self.identifier, call_context, properties))
        return {"FINISHED"}


@pytest.fixture
def deck(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    ran = []
    ready = {"use_active_as_domain": False}

    class FLUIDSIM_OT_use_active_as_domain(bpy.types.Operator):
        @classmethod
        def poll(cls, context):
            cls.poll_message_set(DOMAIN_REASON)
            return False

    ops = {}
    for name in ("setup_scene", "tag_liquid_emitter", "tag_collider", "use_active_as_domain",
                 "command_palette"):
        ops[name] = _FakeOp(f"FLUIDSIM_OT_{name}", ready.get(name, True), ran)
    bpy.ops = SimpleNamespace(fluidsim=SimpleNamespace(**ops))
    bpy.types.FLUIDSIM_OT_use_active_as_domain = FLUIDSIM_OT_use_active_as_domain
    bpy.context = SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(
        deck_stage="SETUP")), area=None, region=None)
    deck.ran = ran
    deck.bpy = bpy
    return deck


def _slot(deck, label):
    return next(c for c in deck.wheel.STAGE_SLOTS["SETUP"] if c.label == label)


def test_status_carries_the_operators_own_poll_message(deck):
    controls = deck.controls
    assert controls.control_status(deck.bpy.context, _slot(deck, "Domain")) == (
        False, DOMAIN_REASON)
    assert controls.control_status(deck.bpy.context, _slot(deck, "Emitter")) == (True, "")
    missing = controls.operator("fluidsim.not_registered", "Gone", "X", "Not here")
    assert controls.control_status(deck.bpy.context, missing)[0] is False


def test_status_explains_a_property_with_no_settings_group(deck, monkeypatch):
    controls = deck.controls
    monkeypatch.setattr(controls, "property_owner", lambda _ctx, _c: None)
    viscosity = deck.wheel.STAGE_SLOTS["SIMULATE"][4]
    ready, reason = controls.control_status(deck.bpy.context, viscosity)
    assert not ready and "domain" in reason.lower()


def _wheel(deck):
    operator = deck.wheel.FLUIDSIM_OT_radial_wheel()
    operator.handle = object()
    operator.reports = []
    operator.report = lambda kind, message: operator.reports.append((kind, message))
    return operator


def test_a_refused_slot_keeps_the_wheel_open_and_says_why(deck):
    wheel = _wheel(deck)
    index = deck.wheel.STAGE_SLOTS["SETUP"].index(_slot(deck, "Domain"))
    assert wheel._pick(deck.bpy.context, index) is None  # modal keeps running
    assert wheel.reports == [({"WARNING"}, f"Domain: {DOMAIN_REASON}")]
    assert deck.ran == []


def test_a_ready_slot_runs_and_closes_the_wheel(deck):
    wheel = _wheel(deck)
    index = deck.wheel.STAGE_SLOTS["SETUP"].index(_slot(deck, "Emitter"))
    assert wheel._pick(deck.bpy.context, index) == {"FINISHED"}
    assert [r[0] for r in deck.ran] == ["FLUIDSIM_OT_tag_liquid_emitter"]
    assert wheel.reports == []


def test_hover_tips_name_the_slot_and_the_reason_and_the_hub_names_the_next_stage(deck):
    slots = deck.wheel.STAGE_SLOTS["SETUP"]
    statuses = [deck.controls.control_status(deck.bpy.context, c) for c in slots]
    domain = slots.index(_slot(deck, "Domain"))
    title, body, note = deck.wheel.slot_tip("SETUP", domain, statuses)
    assert title == "Domain" and "domain" in body and note == f"Unavailable: {DOMAIN_REASON}"
    title, body, note = deck.wheel.slot_tip("SETUP", 1, statuses)
    assert title == "Emitter" and note == ""
    title, body, _ = deck.wheel.slot_tip("SETUP", "HUB", statuses)
    assert "Setup" in title and "Simulate" in body
    assert deck.wheel.slot_tip("SETUP", None, statuses) == ("", "", "")


def test_preset_and_demo_open_the_palette_filtered_to_that_list(deck):
    preset, demo = _slot(deck, "Preset"), _slot(deck, "Demo")
    assert (preset.target, dict(preset.properties)) == ("fluidsim.command_palette",
                                                       {"kinds": "preset"})
    assert (demo.target, dict(demo.properties)) == ("fluidsim.command_palette", {"kinds": "demo"})
    assert set(deck.palette.KIND_PROMPTS) == {"preset", "demo"}
    wheel = _wheel(deck)
    wheel._pick(deck.bpy.context, deck.wheel.STAGE_SLOTS["SETUP"].index(demo))
    assert deck.ran == [("FLUIDSIM_OT_command_palette", "INVOKE_DEFAULT", {"kinds": "demo"})]


def test_a_palette_opened_for_one_kind_lists_only_that_kind(deck, monkeypatch):
    palette = deck.palette
    entries = [palette.Entry("operator", "fluidsim.bake_start", "Bake"),
               palette.Entry("demo", "dam_break", "Dam Break"),
               palette.Entry("preset", "builtin:calm", "Calm Water")]
    monkeypatch.setattr(palette, "build_index", lambda _scene: list(entries))
    operator = palette.FLUIDSIM_OT_command_palette()
    operator.kinds = "demo"
    area = SimpleNamespace(tag_redraw=lambda: None)
    context = SimpleNamespace(scene=SimpleNamespace(), area=area, region=SimpleNamespace(),
                              window_manager=SimpleNamespace(modal_handler_add=lambda _op: None))
    assert operator.invoke(context, SimpleNamespace()) == {"RUNNING_MODAL"}
    assert [e.target for e in operator.results] == ["dam_break"]
    operator._close(context)
