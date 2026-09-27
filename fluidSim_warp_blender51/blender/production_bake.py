"""Transactional cache-v2 production bake controller and Blender backend.

The pure state machine performs at most one frame transaction per ``tick``.
Blender-specific scene/session work is confined to ``ProductionBakeBackend``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
import signal
from typing import Callable

from ..io.cache_schema import (
    CacheManifestV2,
    FrameEntry,
    ResumeDecision,
    load_manifest,
    manifest_path,
    verified_contiguous_frames,
    write_manifest_atomic,
    utc_now,
)
from ..io.checkpoint import (
    capture_solver,
    read_checkpoint,
    restore_solver,
    write_checkpoint_atomic,
)
from ..version import VERSION_STRING


class BakeStatus(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BakeResult:
    status: BakeStatus
    frame_start: int
    frame_end: int
    last_committed_frame: int | None
    cache_path: str
    message: str


class BakeController:
    """Unit-testable owner of production-bake transitions."""

    def __init__(self, *, backend, progress_callback: Callable[[dict], None] | None = None):
        self.backend = backend
        self.progress_callback = progress_callback
        self.status = BakeStatus.IDLE
        self.frame_start = 0
        self.frame_end = -1
        self.next_frame = 0
        self.last_committed_frame: int | None = None
        self.progress = 0.0
        self.message = ""
        self.error = ""
        self._resume = False
        self._restart = False
        self._cancel_requested = False
        self._reserved_cache_path: Path | None = None

    def reserve_cache_path(self, path: Path | str) -> None:
        """Reserve the resolved intended cache path before backend initialization."""
        self._reserved_cache_path = Path(path).resolve()

    def start(self, scene, start: int, end: int, *, restart: bool = False) -> None:
        if self.status in {
            BakeStatus.INITIALIZING,
            BakeStatus.RUNNING,
            BakeStatus.CANCELLING,
        }:
            raise RuntimeError("a production bake is already active")
        start = int(start)
        end = int(end)
        if end < start:
            raise ValueError(f"bake end frame {end} is before start frame {start}")
        self.frame_start = start
        self.frame_end = end
        self.next_frame = start
        self.last_committed_frame = None
        self.progress = 0.0
        self.message = "Initializing production bake"
        self.error = ""
        self._resume = False
        self._restart = bool(restart)
        self._cancel_requested = False
        self.status = BakeStatus.INITIALIZING

    def resume(self, scene, *, end: int | None = None) -> None:
        if self.status in {
            BakeStatus.INITIALIZING,
            BakeStatus.RUNNING,
            BakeStatus.CANCELLING,
        }:
            raise RuntimeError("a production bake is already active")
        decision = self.backend.resume_decision(scene)
        if not decision.allowed or decision.resume_frame is None:
            raise ValueError(f"{decision.code}: {decision.message}")
        manifest = self.backend.load_manifest(scene)
        if manifest is None:
            raise ValueError("NO_CACHE: no cache-v2 manifest exists")
        self.frame_start = manifest.frame_start
        self.frame_end = int(manifest.frame_end if end is None else end)
        if self.frame_end < decision.resume_frame:
            raise ValueError("resume end frame is before the next uncommitted frame")
        self.next_frame = decision.resume_frame
        self.last_committed_frame = decision.resume_frame - 1
        completed = max(0, self.next_frame - self.frame_start)
        total = self.frame_end - self.frame_start + 1
        self.progress = min(1.0, completed / max(total, 1))
        self.message = decision.message
        self.error = ""
        self._resume = True
        self._restart = False
        self._cancel_requested = False
        self.status = BakeStatus.INITIALIZING

    def request_cancel(self) -> None:
        if self.status in {BakeStatus.INITIALIZING, BakeStatus.RUNNING}:
            self._cancel_requested = True
            self.status = BakeStatus.CANCELLING
            self.message = "Cancellation requested; stopping at the committed boundary"

    def _emit(self, event: str, **values) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **values})

    def tick(self, scene) -> float | None:
        """Advance one transition/frame; return a suggested timer delay."""
        try:
            if self.status is BakeStatus.INITIALIZING:
                result = self.backend.initialize(
                    scene,
                    self.frame_start,
                    self.frame_end,
                    resume=self._resume,
                    restart=self._restart,
                )
                if self._resume and isinstance(result, int):
                    self.next_frame = result
                    self.last_committed_frame = result - 1
                self.status = BakeStatus.RUNNING
                self.message = f"Ready to commit frame {self.next_frame}"
                self._emit(
                    "bake_initialized",
                    frame_start=self.frame_start,
                    frame_end=self.frame_end,
                    resume=self._resume,
                )
                return 0.01

            if self.status is BakeStatus.CANCELLING or self._cancel_requested:
                self.backend.cancel(scene)
                self.status = BakeStatus.CANCELLED
                self.message = "Production bake cancelled at a safe frame boundary"
                self._emit(
                    "bake_cancelled", last_committed_frame=self.last_committed_frame
                )
                return None

            if self.status is not BakeStatus.RUNNING:
                return None

            if self.next_frame > self.frame_end:
                self.backend.finish(scene)
                self.status = BakeStatus.COMPLETE
                self.progress = 1.0
                self.message = "Production bake complete"
                self._emit(
                    "bake_complete", last_committed_frame=self.last_committed_frame
                )
                return None

            frame = self.next_frame
            self.backend.step_and_commit(scene, frame)
            self.last_committed_frame = frame
            self.next_frame = frame + 1
            total = self.frame_end - self.frame_start + 1
            completed = frame - self.frame_start + 1
            self.progress = min(1.0, completed / max(total, 1))
            self.message = f"Committed frame {frame}"
            self._emit(
                "frame_complete",
                frame=frame,
                progress=self.progress,
                last_committed_frame=frame,
            )
            return 0.01
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.message = f"Production bake failed: {exc}"
            try:
                self.backend.fail(scene, exc)
            except Exception as cleanup_exc:  # do not hide the original failure
                self.error += f"; cleanup failed: {cleanup_exc}"
            finally:
                # Keep the reserved cache path active through backend cleanup.
                # Mutators may run from cleanup callbacks and must still see
                # the bake lease until failure handling has fully finished.
                self.status = BakeStatus.FAILED
            self._emit("bake_failed", error=self.error)
            return None

    def result(self) -> BakeResult:
        return BakeResult(
            status=self.status,
            frame_start=self.frame_start,
            frame_end=self.frame_end,
            last_committed_frame=self.last_committed_frame,
            cache_path=str(getattr(self.backend, "base_dir", "")),
            message=self.message,
        )


_ACTIVE_BAKES: dict[int, BakeController] = {}
_ACTIVE_STATUSES = {
    BakeStatus.PREFLIGHT,
    BakeStatus.INITIALIZING,
    BakeStatus.RUNNING,
    BakeStatus.CANCELLING,
}


def _scene_key(scene) -> int:
    pointer = getattr(scene, "as_pointer", None)
    if callable(pointer):
        try:
            value = int(pointer())
            if value:
                return value
        except (ReferenceError, TypeError, ValueError):
            pass
    return id(scene)


def _active_cache_path(controller) -> Path | None:
    if getattr(controller, "status", None) not in _ACTIVE_STATUSES:
        return None
    reserved = getattr(controller, "_reserved_cache_path", None)
    if reserved is not None:
        return Path(reserved).resolve()
    backend = getattr(controller, "backend", None)
    base_dir = getattr(backend, "base_dir", None)
    return None if base_dir is None else Path(base_dir).resolve()


def register_active_bake(scene, controller: BakeController) -> None:
    """Give one controller authoritative bake ownership for ``scene``."""
    key = _scene_key(scene)
    existing = _ACTIVE_BAKES.get(key)
    if existing is not None and existing is not controller:
        raise RuntimeError("A Production Bake is already active for this scene")
    intended_path = _active_cache_path(controller)
    if intended_path is not None:
        for owner in tuple(_ACTIVE_BAKES.values()):
            if owner is not controller and _active_cache_path(owner) == intended_path:
                raise RuntimeError(
                    f"Production Bake cache path is already active: {intended_path}")
    _ACTIVE_BAKES[key] = controller


def active_bake(scene) -> BakeController | None:
    """Return the controller that currently owns ``scene``'s bake workflow."""
    return _ACTIVE_BAKES.get(_scene_key(scene))


def bake_is_active(scene) -> bool:
    controller = active_bake(scene)
    return bool(controller is not None and controller.status in {
        BakeStatus.PREFLIGHT,
        BakeStatus.INITIALIZING,
        BakeStatus.RUNNING,
        BakeStatus.CANCELLING,
    })


def clear_active_bake(scene, controller: BakeController | None = None) -> None:
    """Release bake ownership, optionally only for the expected controller."""
    key = _scene_key(scene)
    existing = _ACTIVE_BAKES.get(key)
    if existing is None or (controller is not None and existing is not controller):
        return
    _ACTIVE_BAKES.pop(key, None)


def cache_path_is_active(path: Path | str) -> bool:
    """Whether a registered running bake owns the resolved cache directory."""
    candidate = Path(path).resolve()
    return any(
        _active_cache_path(controller) == candidate
        for controller in tuple(_ACTIVE_BAKES.values())
    )


class ProductionBakeBackend:
    """Blender/session adapter for one-frame cache-v2 transactions."""

    def __init__(self, *, checkpoint_interval: int = 1, compressed: bool = False):
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.compressed = bool(compressed)
        self.base_dir = Path()
        self.manifest: CacheManifestV2 | None = None
        self.signatures: dict[str, str] = {}
        self.effect_signature = ""
        self.effect_inventory: tuple[dict[str, object], ...] = ()
        self.sess = None
        self._original_frame = None
        self._original_live = False
        self._original_session_payload = None
        self._original_session_signatures = None
        self._original_session_active = False
        self._initialized = False

    @staticmethod
    def _bpy():
        import bpy

        return bpy

    def cache_path(self, scene) -> Path:
        bpy = self._bpy()
        return Path(bpy.path.abspath(scene.fluidsim.cache_path)).resolve()

    def load_manifest(self, scene) -> CacheManifestV2 | None:
        return load_manifest(manifest_path(self.cache_path(scene)))

    @staticmethod
    def _compatible_solver_version(cached: str, current: str) -> bool:
        try:
            cached_parts = tuple(int(value) for value in cached.split("."))
            current_parts = tuple(int(value) for value in current.split("."))
        except ValueError:
            return False
        return cached_parts[:2] == current_parts[:2]

    def resume_decision(self, scene) -> ResumeDecision:
        from . import cache_identity

        base = self.cache_path(scene)
        try:
            manifest = load_manifest(manifest_path(base))
        except ValueError as exc:
            return ResumeDecision(False, None, "MANIFEST_INVALID", str(exc))
        if manifest is None:
            return ResumeDecision(
                False, None, "NO_CACHE", "No cache-v2 manifest exists to resume"
            )
        if not self._compatible_solver_version(
            manifest.solver_version, VERSION_STRING
        ):
            return ResumeDecision(
                False,
                None,
                "SOLVER_VERSION_MISMATCH",
                f"Cache solver {manifest.solver_version} is incompatible with {VERSION_STRING}",
            )
        from .effects import effect_identity
        try:
            current_effect_signature, current_effect_inventory = effect_identity(scene)
        except ValueError as exc:
            return ResumeDecision(False, None, "EFFECT_STACK_NOT_APPLIED", str(exc))
        if manifest.effect_signature != current_effect_signature:
            return ResumeDecision(
                False,
                None,
                "EFFECT_SIGNATURE_MISMATCH",
                "Current applied Custom Effects do not match the cache effect identity",
            )
        if tuple(manifest.effect_inventory) != tuple(current_effect_inventory):
            return ResumeDecision(
                False,
                None,
                "EFFECT_SIGNATURE_MISMATCH",
                "Current applied Custom Effects inventory does not match the cache",
            )
        signatures = cache_identity.signature_bundle(
            scene, self._bpy().context.evaluated_depsgraph_get()
        )
        for key, code in (
            ("scene_key", "SCENE_KEY_MISMATCH"),
            ("settings_signature", "SETTINGS_SIGNATURE_MISMATCH"),
            ("source_signature", "SOURCE_SIGNATURE_MISMATCH"),
        ):
            if getattr(manifest, key) != signatures[key]:
                return ResumeDecision(
                    False,
                    None,
                    code,
                    f"Current {key.replace('_', ' ')} does not match the cache",
                )
        try:
            verified_contiguous_frames(base, manifest)
            for frame in manifest.committed_frames:
                entry = manifest.frames[frame]
                checkpoint = read_checkpoint(
                    base / entry.checkpoint_file,
                    expected_sha256=entry.checkpoint_sha256,
                    expected_size=entry.checkpoint_size,
                    expected_frame=frame,
                    expected_settings_signature=manifest.settings_signature,
                    expected_source_signature=manifest.source_signature,
                    expected_effect_signature=manifest.effect_signature,
                )
                if checkpoint.frame != entry.solver_frame:
                    raise ValueError(
                        f"checkpoint frame {checkpoint.frame} does not match manifest"
                    )
                if abs(checkpoint.sim_time - entry.simulation_time) > 1.0e-12:
                    raise ValueError(
                        f"checkpoint time for frame {frame} does not match manifest"
                    )
        except ValueError as exc:
            return ResumeDecision(False, None, "CACHE_INTEGRITY_FAILURE", str(exc))
        last = manifest.last_committed_frame
        if last is None:
            return ResumeDecision(
                True,
                manifest.frame_start,
                "EMPTY_CACHE",
                f"Resume will start at frame {manifest.frame_start}",
            )
        if last >= manifest.frame_end:
            return ResumeDecision(
                False,
                None,
                "ALREADY_COMPLETE",
                f"Cache is already complete through frame {manifest.frame_end}",
            )
        return ResumeDecision(
            True,
            last + 1,
            "RESUME_READY",
            f"Resume from verified checkpoint {last}; next frame is {last + 1}",
        )

    def _snapshot_live_state(self, scene) -> None:
        from . import cache_identity, scene_bridge

        self.sess = scene_bridge.session()
        self._original_frame = int(scene.frame_current)
        self._original_live = bool(scene.fluidsim.live)
        self._original_session_active = bool(self.sess.owns_scene(scene))
        if self._original_session_active:
            signatures = cache_identity.signature_bundle(
                scene, self._bpy().context.evaluated_depsgraph_get()
            )
            from .effects import effect_identity
            original_effect_signature, original_effect_inventory = effect_identity(scene)
            self._original_session_payload = capture_solver(
                self.sess.solver,
                settings_signature=signatures["settings_signature"],
                source_signature=signatures["source_signature"],
                effect_signature=original_effect_signature,
                effect_inventory=original_effect_inventory,
            )
            self._original_session_signatures = signatures

    def _clear_for_restart(self, base: Path) -> None:
        from .cache_manager import require_restartable_cache_directory

        if not base.exists():
            return
        entries = ("fluidsim_cache.json", "frames", "checkpoints", "logs")
        require_restartable_cache_directory(base, entries)
        for name in entries:
            path = base / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    def initialize(
        self,
        scene,
        start: int,
        end: int,
        *,
        resume: bool = False,
        restart: bool = False,
    ) -> int | None:
        from . import cache_identity, playback, scene_bridge

        if self._initialized:
            raise RuntimeError("production bake backend is already initialized")
        self._snapshot_live_state(scene)
        scene.fluidsim.live = False
        self.base_dir = self.cache_path(scene)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        depsgraph = self._bpy().context.evaluated_depsgraph_get()
        self.signatures = cache_identity.signature_bundle(scene, depsgraph)
        from .effects import effect_identity
        self.effect_signature, self.effect_inventory = effect_identity(scene)

        if resume:
            decision = self.resume_decision(scene)
            if not decision.allowed or decision.resume_frame is None:
                raise ValueError(f"{decision.code}: {decision.message}")
            self.manifest = load_manifest(manifest_path(self.base_dir))
            if self.manifest is None:
                raise ValueError("cache-v2 manifest disappeared during resume")
            if self.manifest.effect_signature != self.effect_signature:
                raise ValueError("EFFECT_SIGNATURE_MISMATCH: cache effect identity changed")
        else:
            existing = load_manifest(manifest_path(self.base_dir))
            if existing is not None and not restart:
                raise ValueError(
                    "cache-v2 data already exists; use Resume or explicit Restart")
            if restart:
                self._clear_for_restart(self.base_dir)
            self.manifest = CacheManifestV2.new(
                solver_version=VERSION_STRING,
                scene_key=self.signatures["scene_key"],
                settings_signature=self.signatures["settings_signature"],
                source_signature=self.signatures["source_signature"],
                frame_start=int(start),
                frame_end=int(end),
                checkpoint_interval=self.checkpoint_interval,
                effect_signature=self.effect_signature,
                effect_inventory=self.effect_inventory,
            )
            write_manifest_atomic(manifest_path(self.base_dir), self.manifest)

        with playback.suppress_frame_handlers():
            self.sess.initialize(scene, depsgraph=depsgraph)
            self.sess.start_frame = int(start)
            self.sess.sync_scene(scene, depsgraph)
            if resume:
                last = self.manifest.last_committed_frame
                if last is not None:
                    entry = self.manifest.frames[last]
                    payload = read_checkpoint(
                        self.base_dir / entry.checkpoint_file,
                        expected_sha256=entry.checkpoint_sha256,
                        expected_size=entry.checkpoint_size,
                        expected_frame=last,
                        expected_settings_signature=self.manifest.settings_signature,
                        expected_source_signature=self.manifest.source_signature,
                        expected_effect_signature=self.manifest.effect_signature,
                    )
                    restore_solver(self.sess.solver, payload)
                next_frame = (
                    self.manifest.frame_start if last is None else last + 1
                )
            else:
                self.sess.solver.frame = int(start) - 1
                next_frame = int(start)
        self._initialized = True
        return next_frame

    def step_and_commit(self, scene, frame: int) -> FrameEntry:
        from . import playback, scene_bridge
        from ..io import cache
        from ..sim.frame_output import CacheFramePayload

        if not self._initialized or self.manifest is None or self.sess is None:
            raise RuntimeError("production bake backend is not initialized")
        frame = int(frame)
        expected = (
            self.manifest.frame_start
            if self.manifest.last_committed_frame is None
            else self.manifest.last_committed_frame + 1
        )
        if frame != expected:
            raise ValueError(f"frame transaction {frame} is not the next frame {expected}")

        with playback.suppress_frame_handlers():
            scene.frame_set(frame)
            self.sess.sync_scene(
                scene, self._bpy().context.evaluated_depsgraph_get())
            scene_bridge.step_frame(
                scene,
                self.sess,
                write_cache=False,
                refresh_pending=True,
                timeline_frame=frame,
                publish_outputs=False,
            )
        if int(self.sess.solver.frame) != frame:
            raise ValueError(
                f"solver advanced to frame {self.sess.solver.frame}, expected {frame}"
            )

        frame_payload = CacheFramePayload.capture(
            self.sess.solver,
            settings_signature=self.manifest.settings_signature,
        )
        frame_path, frame_digest, frame_size = cache.write_frame_v2_atomic(
            self.base_dir,
            frame_payload,
            frame,
            scene_key_value=self.manifest.scene_key,
            settings_signature=self.manifest.settings_signature,
            source_signature=self.manifest.source_signature,
            effect_signature=self.effect_signature,
            effect_inventory=self.effect_inventory,
            simulation_time=float(self.sess.solver.sim_time),
            compressed=self.compressed,
        )
        checkpoint = capture_solver(
            self.sess.solver,
            settings_signature=self.manifest.settings_signature,
            source_signature=self.manifest.source_signature,
            effect_signature=self.effect_signature,
            effect_inventory=self.effect_inventory,
        )
        if checkpoint.frame != frame:
            raise ValueError("captured checkpoint frame does not match frame output")
        checkpoint_path = (
            self.base_dir / "checkpoints" / f"checkpoint_{frame:06d}.npz"
        )
        checkpoint_digest = write_checkpoint_atomic(checkpoint_path, checkpoint)
        checkpoint_size = checkpoint_path.stat().st_size

        entry = FrameEntry(
            frame=frame,
            frame_file=frame_path.relative_to(self.base_dir).as_posix(),
            frame_sha256=frame_digest,
            frame_size=frame_size,
            checkpoint_file=checkpoint_path.relative_to(self.base_dir).as_posix(),
            checkpoint_sha256=checkpoint_digest,
            checkpoint_size=checkpoint_size,
            completed_at=utc_now(),
            solver_frame=checkpoint.frame,
            simulation_time=checkpoint.sim_time,
            settings_signature=self.manifest.settings_signature,
            source_signature=self.manifest.source_signature,
        )
        # The copied manifest is committed last. If this replacement fails,
        # both final artifacts remain uncommitted orphans and are ignored.
        candidate = self.manifest.with_committed_frame(entry)
        write_manifest_atomic(manifest_path(self.base_dir), candidate)
        self.manifest = candidate
        return entry

    def _restore_original_state(self, scene) -> None:
        from . import playback

        if self.sess is None:
            return
        restore_error = None
        try:
            if self._original_session_active and self._original_session_payload is not None:
                with playback.suppress_frame_handlers():
                    self.sess.initialize(
                        scene,
                        depsgraph=self._bpy().context.evaluated_depsgraph_get(),
                    )
                    self.sess.sync_scene(
                        scene, self._bpy().context.evaluated_depsgraph_get())
                    restore_solver(self.sess.solver, self._original_session_payload)
            else:
                self.sess.release(scene)
        except Exception as exc:  # restoration is part of the contract
            restore_error = exc
        finally:
            if self._original_frame is not None:
                try:
                    with playback.suppress_frame_handlers():
                        scene.frame_set(self._original_frame)
                except Exception as exc:
                    restore_error = restore_error or exc
            scene.fluidsim.live = self._original_live
            self._initialized = False
        if restore_error is not None:
            raise RuntimeError(f"could not restore pre-bake live state: {restore_error}")

    def finish(self, scene) -> None:
        self._restore_original_state(scene)

    def cancel(self, scene) -> None:
        self._restore_original_state(scene)

    def fail(self, scene, _error) -> None:
        if self._initialized:
            self._restore_original_state(scene)


def run_synchronous(
    scene,
    sess=None,
    start: int | None = None,
    end: int | None = None,
    *,
    resume: bool = False,
    restart: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> BakeResult:
    """Compatibility delegate; ``cache_bake`` owns sequential execution."""
    from .cache_bake import run_synchronous as run_cache_bake

    return run_cache_bake(
        scene, sess, start, end, resume=resume, restart=restart,
        progress_callback=progress_callback)


class SignalCancellation:
    """Request cancellation on SIGINT/SIGTERM without interrupting a transaction."""

    def __init__(self, controller: BakeController):
        self.controller = controller
        self.previous: dict[int, object] = {}

    def __enter__(self):
        def request(_signum, _frame):
            self.controller.request_cancel()

        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        for signum, handler in self.previous.items():
            # getsignal() returns None when the previous handler was installed
            # outside Python (Blender does this); it cannot be reinstalled, and
            # passing None raises TypeError, so fall back to the default.
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)
        return False
