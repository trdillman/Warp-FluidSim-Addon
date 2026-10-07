"""Contracts and unit regressions for the N-panel Demo Gallery."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from fluidSim_warp_blender51.blender import demo_gallery
from fluidSim_warp_blender51.blender.demo_gallery import (
    DEMO_CATALOG,
    FLUIDSIM_OT_load_demo,
    FLUIDSIM_PT_demo_gallery,
    is_scene_empty,
    resolve_demo_blend_path,
    resolve_demo_preview_path,
)

ROOT = Path(__file__).resolve().parents[1]
INIT_PY = ROOT / "__init__.py"


def test_demo_catalog_contains_only_the_four_canonical_demos():
    """Per DEMO_SCENE_CONTRACT.md, only 4 canonical demos exist."""
    canonical_keys = {"dam_break", "crown_splash", "paddle_stir", "chromatic_vortex"}
    assert set(DEMO_CATALOG.keys()) == canonical_keys

    for key, meta in DEMO_CATALOG.items():
        assert meta["title"], f"{key} missing title"
        assert meta["scene_title"], f"{key} missing scene_title"
        assert meta["blend_filename"].endswith(".blend"), f"{key} blend filename invalid"
        assert meta["subtitle"], f"{key} missing subtitle"
        assert meta["description"], f"{key} missing description"
        assert len(meta["tags"]) >= 2, f"{key} missing tags"
        assert meta["icon"], f"{key} missing fallback icon"


def test_resolve_demo_blend_paths_find_all_repo_demos():
    """All 4 canonical demo blend files must be found in the repository."""
    for key in DEMO_CATALOG:
        path = resolve_demo_blend_path(key)
        assert path is not None, f"Could not resolve blend path for {key}"
        assert path.is_file(), f"Resolved path is not a file: {path}"
        assert path.name == DEMO_CATALOG[key]["blend_filename"]


def test_resolve_demo_preview_paths():
    """Every declared preview ships in demos/; undeclared ones resolve to None."""
    for key, meta in DEMO_CATALOG.items():
        preview = resolve_demo_preview_path(key)
        if meta["preview_filename"]:
            assert preview is not None and preview.is_file(), key
            assert preview.name == meta["preview_filename"]
        else:
            assert preview is None, key
    assert resolve_demo_preview_path("not_a_demo") is None


def test_asset_search_prefers_earlier_directories(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "dam_break.blend").write_bytes(b"")
    monkeypatch.setattr(demo_gallery, "DEMO_SEARCH_DIRS", (first, second))
    assert resolve_demo_blend_path("dam_break") == second / "dam_break.blend"

    (first / "dam_break.blend").write_bytes(b"")
    assert resolve_demo_blend_path("dam_break") == first / "dam_break.blend"
    assert demo_gallery.available_demo_ids() == ("dam_break",)
    assert resolve_demo_blend_path("not_a_demo") is None


def test_is_scene_empty_truth_table(monkeypatch):
    """Verify is_scene_empty correctly classifies empty vs populated scenes."""
    # When active_domain_settings returns something, it is never empty
    monkeypatch.setattr(
        demo_gallery,
        "active_domain_settings",
        lambda scn: "fake_domain" if getattr(scn, "has_domain", False) else None,
    )

    # 1. None scene -> True
    assert is_scene_empty(None) is True

    # 2. Scene with domain -> False
    scene_with_domain = SimpleNamespace(has_domain=True, objects=[])
    assert is_scene_empty(scene_with_domain) is False

    # 3. Completely empty scene (0 objects) -> True
    empty_scene = SimpleNamespace(has_domain=False, objects=[])
    assert is_scene_empty(empty_scene) is True

    # 4. Default scene with Cube, Light, Camera -> True
    def make_obj(name, has_action=False, has_mod=False, role="NONE", custom_keys=None):
        obj = SimpleNamespace(
            name=name,
            animation_data=SimpleNamespace(action="act") if has_action else None,
            modifiers=[1] if has_mod else [],
            fluidsim=SimpleNamespace(role=role),
        )
        keys_dict = custom_keys or {}
        obj.keys = lambda: keys_dict.keys()
        return obj

    default_objects = [
        make_obj("Cube"),
        make_obj("Light"),
        make_obj("Camera"),
    ]
    scene_default = SimpleNamespace(has_domain=False, objects=default_objects)
    assert is_scene_empty(scene_default) is True

    # 5. Scene with user object (e.g. "Suzanne") -> False
    user_objects = [
        make_obj("Suzanne"),
        make_obj("Light"),
        make_obj("Camera"),
    ]
    scene_user = SimpleNamespace(has_domain=False, objects=user_objects)
    assert is_scene_empty(scene_user) is False

    # 6. Scene where Cube has animation -> False
    animated_objects = [
        make_obj("Cube", has_action=True),
        make_obj("Light"),
        make_obj("Camera"),
    ]
    scene_animated = SimpleNamespace(has_domain=False, objects=animated_objects)
    assert is_scene_empty(scene_animated) is False

    # 7. Scene where Cube has modifier -> False
    modded_objects = [
        make_obj("Cube", has_mod=True),
        make_obj("Light"),
        make_obj("Camera"),
    ]
    scene_modded = SimpleNamespace(has_domain=False, objects=modded_objects)
    assert is_scene_empty(scene_modded) is False

    # 8. Scene where Cube has FluidSim role -> False
    fluidsim_role_objects = [
        make_obj("Cube", role="COLLIDER"),
        make_obj("Light"),
        make_obj("Camera"),
    ]
    scene_fluidsim = SimpleNamespace(has_domain=False, objects=fluidsim_role_objects)
    assert is_scene_empty(scene_fluidsim) is False


def test_gallery_ui_metadata():
    """Verify panel and operator identifiers and sidebar registration properties."""
    assert FLUIDSIM_OT_load_demo.bl_idname == "fluidsim.load_demo"
    assert FLUIDSIM_PT_demo_gallery.bl_idname == "FLUIDSIM_PT_demo_gallery"
    assert FLUIDSIM_PT_demo_gallery.bl_space_type == "VIEW_3D"
    assert FLUIDSIM_PT_demo_gallery.bl_region_type == "UI"
    assert FLUIDSIM_PT_demo_gallery.bl_category == "FluidSim"
    assert FLUIDSIM_PT_demo_gallery.bl_order == 8
    # Collapsed by default so it never pushes the Setup stage off screen.
    assert "DEFAULT_CLOSED" in FLUIDSIM_PT_demo_gallery.bl_options


def test_root_registration_wires_demo_gallery():
    """Verify demo gallery is part of the add-on lifecycle in __init__.py."""
    source = INIT_PY.read_text(encoding="utf-8")
    assert "demo_gallery," in source or "from .blender import (\n        demo_gallery," in source
    assert '("demo gallery", demo_gallery.register, demo_gallery.unregister)' in source


def test_opened_demo_renders_next_to_the_artists_blend():
    # The shipped .blend files carry the build machine's absolute output path.
    for demo_id in DEMO_CATALOG:
        path = demo_gallery.demo_render_filepath(demo_id)
        assert path.startswith("//") and demo_id in path
    source = Path(demo_gallery.__file__).read_text(encoding="utf-8")
    assert "imported_scene.render.filepath = demo_render_filepath(self.demo_id)" in source


def _lifecycle_step_names(function_name: str) -> list[str]:
    tree = ast.parse(INIT_PY.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    steps = next(
        node.value for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "steps" for t in node.targets)
    )
    return [ast.literal_eval(item.elts[0]) for item in steps.elts]


def test_root_unregister_reverses_every_register_step():
    """A step registered but never unregistered breaks disable/re-enable and
    Reload Scripts with 'already registered' (caught by the installed smoke)."""
    registered = _lifecycle_step_names("register")
    # Teardown-only steps with no register() counterpart.
    teardown_only = {"native CUDA cache", "legacy effect migration handlers"}
    unregistered = [
        name for name in _lifecycle_step_names("unregister") if name not in teardown_only
    ]
    assert unregistered == list(reversed(registered))


def test_load_button_tooltip_describes_the_demo():
    for key, meta in DEMO_CATALOG.items():
        tip = FLUIDSIM_OT_load_demo.description(None, SimpleNamespace(demo_id=key))
        assert meta["description"] in tip
        assert meta["key_feature"] in tip
        assert all(tag in tip for tag in meta["tags"])
        assert "new scene" in tip
    unknown = FLUIDSIM_OT_load_demo.description(None, SimpleNamespace(demo_id="nope"))
    assert unknown == FLUIDSIM_OT_load_demo.bl_description


class _Recorder:
    """Minimal UILayout double that records labels and operator buttons."""

    def __init__(self, log=None):
        self.log = log if log is not None else []
        self.enabled = True
        self.active = True
        self.scale_y = 1.0

    def _child(self, *args, **kwargs):
        return _Recorder(self.log)

    row = column = box = _child

    def label(self, *, text="", **kwargs):
        self.log.append(("label", text, self.enabled, kwargs))

    def operator(self, idname, *, text="", **kwargs):
        props = SimpleNamespace()
        self.log.append(("operator", idname, text, props, kwargs))
        return props


def _draw_gallery(monkeypatch, available, pcoll=None):
    monkeypatch.setattr(demo_gallery, "available_demo_ids", lambda: tuple(available))
    monkeypatch.setattr(demo_gallery, "get_preview_collection", lambda: pcoll)
    layout = _Recorder()
    FLUIDSIM_PT_demo_gallery.draw(SimpleNamespace(layout=layout), SimpleNamespace())
    return layout.log


def test_gallery_draws_one_load_button_per_available_demo(monkeypatch):
    log = _draw_gallery(monkeypatch, DEMO_CATALOG)
    buttons = [entry for entry in log if entry[0] == "operator"]
    assert [b[3].demo_id for b in buttons] == list(DEMO_CATALOG)
    for _, idname, text, props, kwargs in buttons:
        assert idname == "fluidsim.load_demo"
        assert text == DEMO_CATALOG[props.demo_id]["title"]
        assert kwargs == {"icon": DEMO_CATALOG[props.demo_id]["icon"]}
    labels = [entry[1] for entry in log if entry[0] == "label"]
    for meta in DEMO_CATALOG.values():
        assert meta["subtitle"] in labels


def test_gallery_uses_loaded_thumbnails_as_button_icons(monkeypatch):
    pcoll = {"crown_splash": SimpleNamespace(icon_id=42)}
    log = _draw_gallery(monkeypatch, DEMO_CATALOG, pcoll=pcoll)
    icons = {e[3].demo_id: e[4] for e in log if e[0] == "operator"}
    assert icons["crown_splash"] == {"icon_value": 42}
    assert icons["dam_break"] == {"icon": DEMO_CATALOG["dam_break"]["icon"]}


def test_gallery_marks_a_missing_demo_file_instead_of_offering_it(monkeypatch):
    log = _draw_gallery(monkeypatch, ("dam_break", "crown_splash", "paddle_stir"))
    offered = {e[3].demo_id for e in log if e[0] == "operator"}
    assert offered == {"dam_break", "crown_splash", "paddle_stir"}
    missing = [e for e in log if e[0] == "label" and "file missing" in e[1]]
    assert len(missing) == 1
    assert missing[0][1].startswith(DEMO_CATALOG["chromatic_vortex"]["title"])
    assert missing[0][2] is False  # drawn disabled


def test_gallery_hides_when_no_demo_files_exist(monkeypatch):
    context = SimpleNamespace(scene=SimpleNamespace(fluidsim=object()))
    monkeypatch.setattr(demo_gallery, "available_demo_ids", lambda: ())
    assert FLUIDSIM_PT_demo_gallery.poll(context) is False
    monkeypatch.setattr(demo_gallery, "available_demo_ids", lambda: ("dam_break",))
    assert FLUIDSIM_PT_demo_gallery.poll(context) is True
    assert FLUIDSIM_PT_demo_gallery.poll(SimpleNamespace(scene=None)) is False
    no_addon = SimpleNamespace(scene=SimpleNamespace(fluidsim=None))
    assert FLUIDSIM_PT_demo_gallery.poll(no_addon) is False


def _run_operator(demo_id):
    reports = []
    op = SimpleNamespace(demo_id=demo_id, report=lambda kind, msg: reports.append((kind, msg)))
    result = FLUIDSIM_OT_load_demo.execute(op, SimpleNamespace(scene=None, window=None))
    return result, reports


def test_load_demo_rejects_unknown_ids():
    result, reports = _run_operator("not_a_demo")
    assert result == {"CANCELLED"}
    assert reports == [({"ERROR"}, "Unknown demo 'not_a_demo'")]


def test_load_demo_cancels_cleanly_when_the_file_is_missing(monkeypatch):
    monkeypatch.setattr(demo_gallery, "resolve_demo_blend_path", lambda demo_id: None)
    released = []
    monkeypatch.setattr(demo_gallery, "release_session", released.append)
    result, reports = _run_operator("dam_break")
    assert result == {"CANCELLED"}
    assert reports[0][0] == {"ERROR"} and "dam_break.blend" in reports[0][1]
    assert released == []  # nothing is torn down before the file is confirmed


def test_unique_scene_name_avoids_collisions(monkeypatch):
    own = object()
    scenes = {"Dam Break (Demo)": object(), "Dam Break (Demo) 2": object(), "Mine": own}
    monkeypatch.setattr(demo_gallery, "bpy", SimpleNamespace(data=SimpleNamespace(scenes=scenes)))
    assert demo_gallery._unique_scene_name("Crown Splash (Demo)") == "Crown Splash (Demo)"
    assert demo_gallery._unique_scene_name("Dam Break (Demo)") == "Dam Break (Demo) 3"
    assert demo_gallery._unique_scene_name("Mine", own) == "Mine"
