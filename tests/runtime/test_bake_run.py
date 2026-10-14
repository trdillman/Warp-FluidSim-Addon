"""One Production Cache Bake lifecycle, one bake registry, one restart entry list.

A bake used to be driven three ways -- ``cache_bake.run_synchronous``, the modal
operator and the command line -- each re-doing "reserve the cache path, register
ownership, protect a restart, settle the protection, release on any exit".
``BakeRun`` owns that sequence once; these tests pin its interface and keep the
callers on it.
"""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path
from types import SimpleNamespace

import pytest

_PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"


@pytest.fixture(autouse=True)
def _empty_bake_registry():
    """A failing test must not leave a lease that fails every later bake test."""
    from fluidSim_warp_blender51.blender import production_bake

    production_bake._ACTIVE_BAKES.clear()
    yield
    production_bake._ACTIVE_BAKES.clear()


class _Backend:
    """Smallest backend the BakeController accepts; records what it was asked."""

    def __init__(self, cache_dir: Path, *, fail_initialize: bool = False):
        self.cache_dir = cache_dir
        self.base_dir = cache_dir
        self.fail_initialize = fail_initialize
        self.events: list[str] = []

    def cache_path(self, _scene):
        return self.cache_dir

    def initialize(self, _scene, _start, _end, *, resume=False, restart=False):
        self.events.append("initialize")
        if self.fail_initialize:
            raise RuntimeError("admission rejected")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "fluidsim_cache.json").write_bytes(b"new-manifest")

    def step_and_commit(self, _scene, frame):
        self.events.append(f"frame:{frame}")

    def finish(self, _scene):
        self.events.append("finish")

    def cancel(self, _scene):
        self.events.append("cancel")

    def fail(self, _scene, _error):
        self.events.append("fail")


def _scene(token: int = 7):
    return SimpleNamespace(
        as_pointer=lambda: token,
        fluidsim=SimpleNamespace(
            bake_start=1, bake_end=2, checkpoint_interval=1, cache_compress=False),
    )


def _cache_with_old_manifest(root: Path) -> Path:
    root.mkdir()
    (root / "fluidsim_cache.json").write_bytes(b"old-manifest")
    return root


def _drain(run, limit=50):
    for _ in range(limit):
        if run.finished:
            return
        run.tick()
    raise AssertionError("bake did not reach a terminal state")


def _backups(cache_dir: Path):
    return tuple(cache_dir.glob(".fluidsim-restart-*"))


def test_a_bake_run_owns_the_cache_path_for_exactly_its_lifetime(tmp_path):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = tmp_path / "cache"
    scene = _scene()
    run = BakeRun(scene, backend=_Backend(cache))

    assert production_bake.cache_path_is_active(cache)
    assert production_bake.active_bake(scene) is run.controller
    assert run.controller._reserved_cache_path == cache.resolve()

    _drain(run)
    assert run.controller.status is production_bake.BakeStatus.COMPLETE
    run.close()

    assert not production_bake.cache_path_is_active(cache)
    assert production_bake.active_bake(scene) is None


def test_a_bake_run_defaults_to_the_scene_authored_range_and_backend(
        monkeypatch, tmp_path):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    made = {}

    class _Made(_Backend):
        def __init__(self, **kwargs):
            made.update(kwargs)
            super().__init__(tmp_path / "cache")

    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _Made)
    scene = _scene()
    scene.fluidsim.bake_start, scene.fluidsim.bake_end = 3, 5
    scene.fluidsim.checkpoint_interval, scene.fluidsim.cache_compress = 4, True

    run = BakeRun(scene)
    try:
        assert (run.controller.frame_start, run.controller.frame_end) == (3, 5)
        assert made == {"checkpoint_interval": 4, "compressed": True}
    finally:
        run.close()


def test_a_restart_keeps_the_old_cache_until_the_replacement_is_admitted(tmp_path):
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = _cache_with_old_manifest(tmp_path / "cache")
    run = BakeRun(_scene(), restart=True, backend=_Backend(cache))
    try:
        assert not (cache / "fluidsim_cache.json").exists()
        assert len(_backups(cache)) == 1

        run.tick()  # INITIALIZING -> RUNNING: the replacement is admitted

        assert (cache / "fluidsim_cache.json").read_bytes() == b"new-manifest"
        assert _backups(cache) == ()
    finally:
        run.close()


def test_a_rejected_restart_gets_the_old_cache_back(tmp_path):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = _cache_with_old_manifest(tmp_path / "cache")
    run = BakeRun(
        _scene(), restart=True, backend=_Backend(cache, fail_initialize=True))
    try:
        run.tick()

        assert run.controller.status is production_bake.BakeStatus.FAILED
        assert (cache / "fluidsim_cache.json").read_bytes() == b"old-manifest"
        assert _backups(cache) == ()
    finally:
        run.close()


def test_a_cancelled_restart_gets_the_old_cache_back(tmp_path):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = _cache_with_old_manifest(tmp_path / "cache")
    run = BakeRun(_scene(), restart=True, backend=_Backend(cache))
    try:
        run.request_cancel()
        run.tick()

        assert run.controller.status is production_bake.BakeStatus.CANCELLED
        assert (cache / "fluidsim_cache.json").read_bytes() == b"old-manifest"
    finally:
        run.close()


def test_close_restores_a_restart_that_was_never_admitted_and_is_idempotent(
        tmp_path):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = _cache_with_old_manifest(tmp_path / "cache")
    scene = _scene()
    run = BakeRun(scene, restart=True, backend=_Backend(cache))

    run.close()
    run.close()

    assert (cache / "fluidsim_cache.json").read_bytes() == b"old-manifest"
    assert _backups(cache) == ()
    assert production_bake.active_bake(scene) is None
    assert not production_bake.cache_path_is_active(cache)


def test_close_releases_ownership_even_when_the_restore_fails(
        monkeypatch, tmp_path):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache = _cache_with_old_manifest(tmp_path / "cache")
    scene = _scene()
    run = cache_bake.BakeRun(scene, restart=True, backend=_Backend(cache))

    def blocked(_self):
        raise OSError("restore blocked")

    monkeypatch.setattr(cache_bake._RestartCacheProtection, "restore", blocked)

    with pytest.raises(OSError, match="restore blocked"):
        run.close()

    assert production_bake.active_bake(scene) is None
    assert not production_bake.cache_path_is_active(cache)


class _Session:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.released: list[bool] = []

    def force_release(self, *, preserve_error):
        self.released.append(preserve_error)
        if self.fail:
            raise RuntimeError("device is gone")


def test_abandon_cancels_a_bake_whose_scene_was_removed(tmp_path):
    """No scene to tick against: free the solver, return the old cache, end CANCELLED."""
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = _cache_with_old_manifest(tmp_path / "cache")
    backend = _Backend(cache)
    backend.sess = _Session()
    scene = _scene()
    run = BakeRun(scene, restart=True, backend=backend)

    run.abandon("scene removed")
    run.close()

    controller = run.controller
    assert controller.status is production_bake.BakeStatus.CANCELLED
    assert controller.message == "scene removed"
    assert backend.sess.released == [True]
    assert (cache / "fluidsim_cache.json").read_bytes() == b"old-manifest"
    assert production_bake.active_bake(scene) is None
    assert backend.events == []  # nothing was ticked against the missing scene


def test_abandon_reports_cleanup_failures_as_failed_but_still_releases(
        monkeypatch, tmp_path):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache = _cache_with_old_manifest(tmp_path / "cache")
    backend = _Backend(cache)
    backend.sess = _Session(fail=True)
    scene = _scene()
    run = cache_bake.BakeRun(scene, restart=True, backend=backend)

    def blocked(_self):
        raise OSError("restore blocked")

    monkeypatch.setattr(cache_bake._RestartCacheProtection, "restore", blocked)

    run.abandon("scene removed")
    with pytest.raises(OSError, match="restore blocked"):
        run.close()

    controller = run.controller
    assert controller.status is production_bake.BakeStatus.FAILED
    assert "device is gone" in controller.error
    assert "restore blocked" in controller.error
    assert production_bake.active_bake(scene) is None


def test_a_failed_acquisition_leaves_no_ownership_behind(tmp_path):
    """A restart refused for a folder that is not a cache must not strand the lease."""
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    project = tmp_path / "project"
    (project / "frames").mkdir(parents=True)
    (project / "frames" / "render_0001.png").write_bytes(b"artist render")
    scene = _scene()

    with pytest.raises(ValueError, match="not a FluidSim cache"):
        BakeRun(scene, restart=True, backend=_Backend(project))

    assert production_bake.active_bake(scene) is None
    assert not production_bake.cache_path_is_active(project)
    assert (project / "frames" / "render_0001.png").read_bytes() == b"artist render"


def test_a_failed_acquisition_frees_the_solver_session_the_start_opened(tmp_path):
    """A started controller whose later setup step fails gives its session back too."""
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    class _Unreachable(_Backend):
        def cache_path(self, _scene):
            raise OSError("cache folder unreachable")

    backend = _Unreachable(tmp_path / "cache")
    backend.sess = _Session()
    scene = _scene()

    with pytest.raises(OSError, match="cache folder unreachable"):
        BakeRun(scene, backend=backend)

    assert backend.sess.released == [True]
    assert production_bake.active_bake(scene) is None


def test_a_scene_cannot_hold_two_bake_runs(tmp_path):
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    scene = _scene()
    first = BakeRun(scene, backend=_Backend(tmp_path / "a"))
    try:
        with pytest.raises(RuntimeError, match="already active"):
            BakeRun(scene, backend=_Backend(tmp_path / "b"))
    finally:
        first.close()


def test_the_registry_lists_every_active_bake_with_its_scene(tmp_path):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    scene_a, scene_b = _scene(11), _scene(22)
    run_a = BakeRun(scene_a, backend=_Backend(tmp_path / "a"))
    try:
        # One bake per process: the Solver Session is a singleton, so a second
        # bake in another scene is refused rather than listed beside the first.
        with pytest.raises(RuntimeError, match="only one bake"):
            BakeRun(scene_b, backend=_Backend(tmp_path / "b"))
        listed = production_bake.active_bakes()
        assert [(id(scene), id(controller)) for scene, controller in listed] == [
            (id(scene_a), id(run_a.controller))]
    finally:
        run_a.close()

    assert production_bake.active_bakes() == ()


def test_a_bake_is_released_by_controller_even_when_its_scene_is_gone(tmp_path):
    """A removed scene reports pointer 0, so its registry key cannot be recomputed."""
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    pointer = [31]
    scene = _scene()
    scene.as_pointer = lambda: pointer[0]
    run = BakeRun(scene, backend=_Backend(tmp_path / "cache"))
    pointer[0] = 0  # the scene was removed under the bake

    run.close()

    assert production_bake.active_bakes() == ()
    assert not production_bake.cache_path_is_active(tmp_path / "cache")


def test_restart_managed_entries_have_a_single_owner(tmp_path):
    from fluidSim_warp_blender51.blender import cache_manager, production_bake

    entries = cache_manager.RESTART_MANAGED_ENTRIES
    assert set(entries) == {"fluidsim_cache.json", "frames", "checkpoints", "logs"}

    cache = tmp_path / "cache"
    cache.mkdir()
    for name in entries:
        target = cache / name
        target.mkdir() if name != "fluidsim_cache.json" else target.write_bytes(b"x")
    (cache / "unrelated.txt").write_bytes(b"keep")

    production_bake.ProductionBakeBackend()._clear_for_restart(cache)

    assert sorted(path.name for path in cache.iterdir()) == ["unrelated.txt"]

    for module in ("cache_bake.py", "production_bake.py"):
        tree = ast.parse((_PACKAGE / "blender" / module).read_text(encoding="utf-8"))
        literals = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Tuple)
            and {"frames", "checkpoints", "logs"} <= {
                element.value for element in node.elts
                if isinstance(element, ast.Constant)}
        ]
        assert literals == [], f"{module} re-declares the restart entry list"


def test_the_modal_operator_has_no_private_bake_registry():
    """Bake ownership lives in production_bake; the operator holds no parallel copy."""
    tree = ast.parse(
        (_PACKAGE / "operator_domains" / "cache.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))

    assert not {
        "_ACTIVE_BAKE", "_ACTIVE_BAKE_SCENE",
        "_SceneRegistrationProxy", "_scene_registration_token",
    } & names


@pytest.mark.parametrize("relative", [
    ("operator_domains", "cache.py"),
    ("blender", "command_line.py"),
])
def test_bake_callers_drive_the_controller_through_bake_run(relative):
    """The operator and the command line share the lifecycle, not copies of it."""
    source = (_PACKAGE.joinpath(*relative)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    called |= {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}

    assert "BakeRun" in called
    assert not {
        "BakeController", "register_active_bake", "clear_active_bake",
        "_protect_restart_cache", "_settle_restart_cache",
    } & called


@pytest.mark.parametrize("status_name", ["IDLE", "PREFLIGHT"])
def test_a_run_that_has_not_started_is_not_finished(tmp_path, status_name):
    """``finished`` means terminal: a driver loop must not exit before the bake began."""
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    run = BakeRun(_scene(), backend=_Backend(tmp_path / "cache"))
    run.controller.status = production_bake.BakeStatus[status_name]

    assert not run.finished


@pytest.mark.parametrize("status_name", ["COMPLETE", "CANCELLED", "FAILED"])
def test_a_run_in_a_terminal_state_is_finished(tmp_path, status_name):
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    run = BakeRun(_scene(), backend=_Backend(tmp_path / "cache"))
    run.controller.status = production_bake.BakeStatus[status_name]

    assert run.finished


def test_a_driver_that_raises_mid_bake_still_releases_ownership(tmp_path):
    """The command line drives BakeRun like any other caller, so it gets the same release."""
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.blender.cache_bake import BakeRun

    cache = tmp_path / "cache"
    scene = _scene()
    run = BakeRun(scene, backend=_Backend(cache))
    assert production_bake.bake_is_active(scene)

    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt
        finally:
            run.close()

    assert not production_bake.bake_is_active(scene)
    assert not production_bake.cache_path_is_active(cache)
