"""Transactional cache-v3 production bake controller and Blender backend.

The pure state machine performs at most one frame transaction per ``tick``.
Blender-specific scene/session work is confined to ``ProductionBakeBackend``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import platform as _platform
import re
import shutil
import signal
from typing import Callable

from ..io.cache_schema import (
    CacheManifestV3,
    FrameEntry,
    LegacyCacheFormatError,
    ResumeDecision,
    first_changed_source_frame,
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
from ..sim.enums import ParticleSampleType
from ..version import VERSION_STRING
from .scene_access import scene_identity


class BakeStatus(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ResumeRefused(ValueError):
    """A resume was not allowed; ``decision`` says why (and, for a changed
    animation, which frame a re-bake would start from)."""

    def __init__(self, decision: ResumeDecision):
        super().__init__(f"{decision.code}: {decision.message}")
        self.decision = decision


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
        self._rebake_from: int | None = None
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
        self._rebake_from = None
        self._cancel_requested = False
        self.status = BakeStatus.INITIALIZING

    def resume(
        self, scene, *, end: int | None = None, rebake_from: int | None = None
    ) -> None:
        """Continue a cache, or with ``rebake_from`` replace it from that frame."""
        if self.status in {
            BakeStatus.INITIALIZING,
            BakeStatus.RUNNING,
            BakeStatus.CANCELLING,
        }:
            raise RuntimeError("a production bake is already active")
        if rebake_from is None:
            decision = self.backend.resume_decision(scene)
        else:
            decision = self.backend.resume_decision(scene, rebake_from=rebake_from)
        if not decision.allowed or decision.resume_frame is None:
            raise ResumeRefused(decision)
        manifest = self.backend.load_manifest(scene)
        if manifest is None:
            raise ValueError("NO_CACHE: no cache manifest exists")
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
        self._rebake_from = None if rebake_from is None else int(rebake_from)
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
                # Only a re-bake passes the extra argument, so a backend that
                # predates it keeps working for every other bake.
                extra = (
                    {} if self._rebake_from is None
                    else {"rebake_from": self._rebake_from}
                )
                result = self.backend.initialize(
                    scene,
                    self.frame_start,
                    self.frame_end,
                    resume=self._resume,
                    restart=self._restart,
                    **extra,
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


# Bakes holding ownership: scene key -> (scene, controller). The scene is kept so
# a holder that has only the controller (the cancel operator) can still reach the
# scene it bakes, and so a bake can be released by controller after its scene was
# removed (a removed scene's pointer reads 0, so its key cannot be recomputed).
_ACTIVE_BAKES: dict[int, tuple[object, BakeController]] = {}
_ACTIVE_STATUSES = {
    BakeStatus.PREFLIGHT,
    BakeStatus.INITIALIZING,
    BakeStatus.RUNNING,
    BakeStatus.CANCELLING,
}


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
    key = scene_identity(scene)
    existing = _ACTIVE_BAKES.get(key)
    if existing is not None and existing[1] is not controller:
        raise RuntimeError("A Production Bake is already active for this scene")
    intended_path = _active_cache_path(controller)
    if intended_path is not None:
        for _owner_scene, owner in tuple(_ACTIVE_BAKES.values()):
            if owner is not controller and _active_cache_path(owner) == intended_path:
                raise RuntimeError(
                    f"Production Bake cache path is already active: {intended_path}")
    _ACTIVE_BAKES[key] = (scene, controller)


def active_bake(scene) -> BakeController | None:
    """Return the controller that currently owns ``scene``'s bake workflow."""
    held = _ACTIVE_BAKES.get(scene_identity(scene))
    return None if held is None else held[1]


def bake_is_active(scene) -> bool:
    controller = active_bake(scene)
    return bool(controller is not None and controller.status in _ACTIVE_STATUSES)


def active_bakes() -> tuple[tuple[object, BakeController], ...]:
    """Every registered bake still running, as ``(scene, controller)`` pairs."""
    return tuple(
        (scene, controller)
        for scene, controller in tuple(_ACTIVE_BAKES.values())
        if controller.status in _ACTIVE_STATUSES
    )


def clear_active_bake(scene, controller: BakeController | None = None) -> None:
    """Release bake ownership, optionally only for the expected controller."""
    key = scene_identity(scene)
    existing = _ACTIVE_BAKES.get(key)
    if existing is None or (controller is not None and existing[1] is not controller):
        return
    _ACTIVE_BAKES.pop(key, None)


def release_active_bake(controller: BakeController) -> None:
    """Release ``controller``'s ownership; needs no live scene to find it."""
    for key, (_scene, held) in tuple(_ACTIVE_BAKES.items()):
        if held is controller:
            del _ACTIVE_BAKES[key]


def cache_path_is_active(path: Path | str) -> bool:
    """Whether a registered running bake owns the resolved cache directory."""
    candidate = Path(path).resolve()
    return any(
        _active_cache_path(controller) == candidate
        for _scene, controller in tuple(_ACTIVE_BAKES.values())
    )


def _platform_record(solver) -> dict[str, str]:
    """Describe where the cache was baked; informational, never in a signature."""
    from ..runtime.warp_runtime import probe_runtime

    probe = probe_runtime()
    alias = str(getattr(solver, "device", "") or "")
    devices = {device.alias: device for device in probe.cuda_devices}
    device = devices.get(alias) or next(iter(devices.values()), None)
    driver = int(probe.cuda_driver_version or 0)
    return {
        "os": _platform.platform() or "unknown",
        "gpu_name": (device.name if device is not None else "") or "unknown",
        # CUDA driver API version (12090 -> "12.9"), as Warp reports it.
        "driver": f"{driver // 1000}.{(driver % 1000) // 10}" if driver else "unknown",
        "warp": probe.warp_version or "unknown",
    }


def _liquid_sdf_arrays(solver, enabled: bool):
    """The frame's ``liquid_sdf_*`` arrays: the solver's persistent phi when on."""
    if not enabled:
        return None
    import numpy as np

    band = getattr(solver, "narrow_band", None)
    if band is None:
        raise RuntimeError("narrow band is enabled but the solver has no band state")
    coords, phi = band.leaf_phi()
    if getattr(solver.stats, "narrow_band_reseed_overflow", False):
        print(
            f"[fluidsim] frame {int(solver.frame)}: narrow-band reseed hit the "
            "particle capacity; the band is under-seeded", flush=True)
    return {
        "liquid_sdf_sparse": phi,
        "liquid_sdf_leaf_coords": coords,
        "liquid_sdf_origin": np.asarray(tuple(solver.grid.origin), dtype=np.float32),
        "liquid_sdf_voxel_size": np.array(
            float(solver.grid.voxel_size), dtype=np.float32),
    }


def _narrow_band_record(settings) -> dict[str, object]:
    """The manifest's ``narrow_band`` block, read from the solver settings."""
    enabled = bool(getattr(settings, "narrow_band_enabled", False))
    return {
        "enabled": enabled,
        "band_voxels": (
            float(getattr(settings, "narrow_band_voxels", 0.0))
            if enabled else 0.0),
    }


def _grid_spec_record(quality) -> dict[str, object]:
    """The grid the solver resolved, as the manifest records it.

    ``resolution_divisions`` is the cell count along the longest domain axis
    (``max(unpadded_voxel_dimensions)``); the Resolution Divisions work (lane G)
    replaces this derivation with the authored value.
    """
    return {
        "resolution_divisions": int(max(quality.unpadded_voxel_dimensions)),
        "voxel_size": float(quality.voxel_size),
        "domain_size": [float(value) for value in quality.domain_size],
        "leaf_dim": 8,
    }


_FRAME_FILE_RE = re.compile(r"^(?:frame|checkpoint)_(\d+)\.npz")


class ProductionBakeBackend:
    """Blender/session adapter for one-frame cache-v3 transactions."""

    def __init__(self, *, checkpoint_interval: int = 1, compressed: bool = False):
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.compressed = bool(compressed)
        self.base_dir = Path()
        self.manifest: CacheManifestV3 | None = None
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
        # True from the pre-bake snapshot until it is restored, so a start that
        # is refused before the solver is touched still puts the live flag back.
        self._snapshot_open = False

    @staticmethod
    def _bpy():
        import bpy

        return bpy

    def cache_path(self, scene) -> Path:
        bpy = self._bpy()
        return Path(bpy.path.abspath(scene.fluidsim.cache_path)).resolve()

    def load_manifest(self, scene) -> CacheManifestV3 | None:
        return load_manifest(manifest_path(self.cache_path(scene)))

    @staticmethod
    def _compatible_solver_version(cached: str, current: str) -> bool:
        try:
            cached_parts = tuple(int(value) for value in cached.split("."))
            current_parts = tuple(int(value) for value in current.split("."))
        except ValueError:
            return False
        return cached_parts[:2] == current_parts[:2]

    def resume_decision(
        self, scene, *, rebake_from: int | None = None
    ) -> ResumeDecision:
        """Whether the cache can continue, or with ``rebake_from`` restart there.

        A plain resume continues after the last committed frame and is refused
        with ``SOURCE_ANIMATION_CHANGED`` when any committed frame was baked
        from a different animation of a source or collider; the decision then
        names the first such frame. ``rebake_from=N`` keeps the frames below N
        (which must still match the animation) and bakes N onward.
        """
        base = self.cache_path(scene)
        try:
            manifest = load_manifest(manifest_path(base))
        except LegacyCacheFormatError as exc:
            return ResumeDecision(False, None, "CACHE_FORMAT_2X", str(exc))
        except ValueError as exc:
            return ResumeDecision(False, None, "MANIFEST_INVALID", str(exc))
        if manifest is None:
            return ResumeDecision(
                False, None, "NO_CACHE", "No cache manifest exists to resume"
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
        signatures = self._signatures_at(scene, manifest.frame_start)
        if manifest.settings_signature != signatures["settings_signature"]:
            # A cache baked before owner decision D1 hashed the device into
            # its identity; judge it against that form of the same scene.
            from . import cache_identity

            for legacy in cache_identity.legacy_signature_bundles(scene, signatures):
                if legacy["settings_signature"] == manifest.settings_signature:
                    signatures = legacy
                    break
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
        target = None if rebake_from is None else int(rebake_from)
        if target is not None:
            last_committed = manifest.last_committed_frame
            newest = (
                manifest.frame_start if last_committed is None
                else last_committed + 1
            )
            if not manifest.frame_start <= target <= newest:
                return ResumeDecision(
                    False,
                    None,
                    "REBAKE_FRAME_INVALID",
                    f"Cannot re-bake from frame {target}: this cache can be "
                    f"re-baked from frames {manifest.frame_start} to {newest}",
                )
            # Everything below is judged on the frames that survive.
            manifest = manifest.truncated_before(target)
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
        changed = self._first_changed_frame(scene, manifest)
        if changed is not None:
            if target is not None:
                message = (
                    f"Cannot re-bake from frame {target}: the animation of a "
                    f"source or collider already differs at frame {changed}"
                )
            else:
                message = (
                    "The animation of a source or collider changed since this "
                    f"cache was baked; the first changed frame is {changed}. "
                    f"Re-bake from frame {changed}, or restart the bake"
                )
            return ResumeDecision(
                False, None, "SOURCE_ANIMATION_CHANGED", message,
                first_changed_frame=changed)
        last = manifest.last_committed_frame
        if last is None:
            return ResumeDecision(
                True,
                manifest.frame_start,
                "EMPTY_CACHE" if target is None else "REBAKE_READY",
                f"Resume will start at frame {manifest.frame_start}",
            )
        if last >= manifest.frame_end:
            return ResumeDecision(
                False,
                None,
                "ALREADY_COMPLETE",
                f"Cache is already complete through frame {manifest.frame_end}",
            )
        if target is not None:
            return ResumeDecision(
                True,
                last + 1,
                "REBAKE_READY",
                f"Keep frames {manifest.frame_start} to {last} and re-bake "
                f"from frame {last + 1}",
            )
        return ResumeDecision(
            True,
            last + 1,
            "RESUME_READY",
            f"Resume from verified checkpoint {last}; next frame is {last + 1}",
        )

    def _first_changed_frame(self, scene, manifest: CacheManifestV3) -> int | None:
        """First committed frame whose sources are no longer what was baked.

        Walks the committed frames forward on the scene, so it leaves the
        playhead where it found it. The scene's pose at a frame is not
        independent of the frames evaluated before it for rigid-body or other
        simulated sources, which is why this is a forward walk from the start.
        """
        if not manifest.frames:
            return None
        from . import playback

        original = int(scene.frame_current)

        def current(frame: int) -> str:
            if int(scene.frame_current) != frame:
                scene.frame_set(frame)
            return self._stamp_now(scene, frame)

        with playback.suppress_frame_handlers():
            try:
                return first_changed_source_frame(manifest, current)
            finally:
                if int(scene.frame_current) != original:
                    scene.frame_set(original)

    def _stamp_now(self, scene, frame: int) -> str:
        """Stamp of the sources as the scene stands, which must be at ``frame``."""
        from . import cache_identity

        return cache_identity.source_frame_stamp(
            scene, self._bpy().context.evaluated_depsgraph_get(), frame)

    def _signatures_at(self, scene, frame: int) -> dict[str, str]:
        """Scene identity evaluated at the bake's reference frame.

        The source signature hashes transforms and evaluated geometry, so an
        animated collider or emitter gives a different identity on every
        frame. Pinning it to the bake start frame makes the stamp written at
        initialize and the one checked on resume agree wherever the playhead
        happens to be.
        """
        from . import cache_identity, playback

        bpy = self._bpy()
        current = int(scene.frame_current)
        with playback.suppress_frame_handlers():
            try:
                if current != int(frame):
                    scene.frame_set(int(frame))
                return cache_identity.signature_bundle(
                    scene, bpy.context.evaluated_depsgraph_get())
            finally:
                if int(scene.frame_current) != current:
                    scene.frame_set(current)

    def _snapshot_live_state(self, scene) -> None:
        from . import cache_identity, scene_bridge

        self.sess = scene_bridge.session()
        self._original_frame = int(scene.frame_current)
        self._original_live = bool(scene.fluidsim.live)
        self._original_session_active = bool(self.sess.owns_scene(scene))
        self._snapshot_open = True
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
        from .cache_manager import (
            RESTART_MANAGED_ENTRIES,
            require_restartable_cache_directory,
        )

        if not base.exists():
            return
        require_restartable_cache_directory(base)
        for name in RESTART_MANAGED_ENTRIES:
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
        rebake_from: int | None = None,
    ) -> int | None:
        from . import playback, scene_bridge

        if self._initialized:
            raise RuntimeError("production bake backend is already initialized")
        if rebake_from is not None and not resume:
            raise ValueError("a re-bake continues an existing cache; pass resume=True")
        self._snapshot_live_state(scene)
        scene.fluidsim.live = False
        self.base_dir = self.cache_path(scene)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.signatures = self._signatures_at(scene, int(start))
        depsgraph = self._bpy().context.evaluated_depsgraph_get()
        from .effects import effect_identity
        self.effect_signature, self.effect_inventory = effect_identity(scene)

        stale_from = None
        if resume:
            decision = (
                self.resume_decision(scene) if rebake_from is None
                else self.resume_decision(scene, rebake_from=rebake_from)
            )
            if not decision.allowed or decision.resume_frame is None:
                raise ResumeRefused(decision)
            self.manifest = load_manifest(manifest_path(self.base_dir))
            if self.manifest is None:
                raise ValueError("cache manifest disappeared during resume")
            if self.manifest.effect_signature != self.effect_signature:
                raise ValueError("EFFECT_SIGNATURE_MISMATCH: cache effect identity changed")
            if rebake_from is not None:
                # In memory only: the truncated manifest is committed below,
                # once the solver is restored, so a failure before that point
                # leaves the cache exactly as it was.
                stale_from = int(rebake_from)
                self.manifest = self.manifest.truncated_before(stale_from)
        else:
            try:
                existing = load_manifest(manifest_path(self.base_dir))
            except LegacyCacheFormatError:
                # A 2.x cache cannot be continued, only replaced.
                if not restart:
                    raise
                existing = None
            if existing is not None and not restart:
                raise ValueError(
                    "cache data already exists; use Resume or explicit Restart")
            if restart:
                self._clear_for_restart(self.base_dir)

        with playback.suppress_frame_handlers():
            self.sess.initialize(scene, depsgraph=depsgraph)
            self.sess.start_frame = int(start)
            self.sess.sync_scene(scene, depsgraph)
            if not resume:
                self.manifest = self._new_manifest(int(start), int(end))
                write_manifest_atomic(manifest_path(self.base_dir), self.manifest)
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
                else:
                    # Nothing to restore from (a re-bake from the first frame,
                    # or a cache with no committed frame): start from rest.
                    self.sess.solver.frame = self.manifest.frame_start - 1
                next_frame = (
                    self.manifest.frame_start if last is None else last + 1
                )
            else:
                self.sess.solver.frame = int(start) - 1
                next_frame = int(start)
        if stale_from is not None:
            self._discard_stale_frames(stale_from)
        self._initialized = True
        return next_frame

    def _new_manifest(self, start: int, end: int) -> CacheManifestV3:
        from .sources import SOURCE_SCHEMA_VERSION

        return CacheManifestV3.new(
            solver_version=VERSION_STRING,
            scene_key=self.signatures["scene_key"],
            settings_signature=self.signatures["settings_signature"],
            source_signature=self.signatures["source_signature"],
            frame_start=start,
            frame_end=end,
            checkpoint_interval=self.checkpoint_interval,
            effect_signature=self.effect_signature,
            effect_inventory=self.effect_inventory,
            source_schema_version=SOURCE_SCHEMA_VERSION,
            transfer_mode=ParticleSampleType(
                int(self.sess.solver.settings.sample_type)).name,
            grid_spec=_grid_spec_record(self.sess.solver.quality),
            narrow_band=_narrow_band_record(self.sess.solver.settings),
            platform=_platform_record(self.sess.solver),
        )

    def _discard_stale_frames(self, first_stale: int) -> None:
        """Commit the truncated manifest, then delete the frames it dropped.

        The manifest goes first: a crash in between leaves unreferenced files,
        which every reader ignores, never a manifest naming deleted files.
        """
        write_manifest_atomic(manifest_path(self.base_dir), self.manifest)
        for directory in ("frames", "checkpoints"):
            folder = self.base_dir / directory
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                match = _FRAME_FILE_RE.match(path.name)
                if match is not None and int(match.group(1)) >= first_stale:
                    path.unlink()

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
            # The stamp describes the sources this frame is simulated against:
            # taken after frame_set and before the step.
            source_frame_stamp = self._stamp_now(scene, frame)
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
        frame_path, frame_digest, frame_size = cache.write_frame_v3_atomic(
            self.base_dir,
            frame_payload,
            frame,
            scene_key_value=self.manifest.scene_key,
            settings_signature=self.manifest.settings_signature,
            source_signature=self.manifest.source_signature,
            source_frame_stamp=source_frame_stamp,
            effect_signature=self.effect_signature,
            effect_inventory=self.effect_inventory,
            simulation_time=float(self.sess.solver.sim_time),
            narrow_band_enabled=bool(self.manifest.narrow_band["enabled"]),
            liquid_sdf=_liquid_sdf_arrays(
                self.sess.solver, bool(self.manifest.narrow_band["enabled"])),
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
            source_frame_stamp=source_frame_stamp,
            whitewater_count=len(frame_payload.whitewater),
            particle_count=len(frame_payload.positions),
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
            self._snapshot_open = False
        if restore_error is not None:
            raise RuntimeError(f"could not restore pre-bake live state: {restore_error}")

    def finish(self, scene) -> None:
        self._restore_original_state(scene)

    def cancel(self, scene) -> None:
        self._restore_original_state(scene)

    def fail(self, scene, _error) -> None:
        if self._initialized:
            self._restore_original_state(scene)
        elif self._snapshot_open:
            # initialize() raised after snapshotting (a refused resume, a 2.x
            # cache, existing data): it had already switched live playback off.
            self._snapshot_open = False
            scene.fluidsim.live = self._original_live


def run_synchronous(
    scene,
    sess=None,
    start: int | None = None,
    end: int | None = None,
    *,
    resume: bool = False,
    restart: bool = False,
    rebake_from: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> BakeResult:
    """Compatibility delegate; ``cache_bake`` owns sequential execution."""
    from .cache_bake import run_synchronous as run_cache_bake

    return run_cache_bake(
        scene, sess, start, end, resume=resume, restart=restart,
        rebake_from=rebake_from, progress_callback=progress_callback)


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
