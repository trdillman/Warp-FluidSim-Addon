"""Inactive, main-thread producer for the opt-in FSPT device surface route.

The caller decides cadence and mandatory host demand before prepare(). It
commits a prepared key only after scene-input synchronization. The ready
offer is metadata. A renderer still needs the pool's checked read lease.
"""

from dataclasses import dataclass
from itertools import count
from math import isfinite
from operator import index
from threading import current_thread, main_thread

import warp as wp

from ..fspt.surface_pool import SurfaceKey
from ..sim.constants import LEAF_DIM
from ..sim.device_surface import DeviceSurfacePacket, DeviceSurfaceUnavailable

# Match SurfaceExtractor._sparse_arrays: one set of (LEAF_DIM + 1)^3
# candidate cubes per active leaf of that exact extractor's grid. The pinned
# Warp marching-cubes table emits at most twelve edge vertices and five faces
# per cube. Sheet Detail changes the input SDF; Waves displaces positions
# without creating vertices or faces. Neither widens this topology bound.
_CANDIDATES_PER_LEAF = (LEAF_DIM + 1) ** 3
_SPARSE_BOUND_WARP_VERSION = "1.19.0"
_MAX_INDEX = (1 << 31) - 1


@dataclass(frozen=True)
class LiveSurfaceSource:
    """Completed solver facts from the existing publication owner."""

    scene_token: int
    owner_token: tuple[int | None, str]
    epoch: int
    source_timeline_frame: int
    state_revision: int
    solver_frame: int
    simulation_time: float
    state_ready_at: float | None = None


@dataclass(frozen=True)
class DeviceSurfaceOffer:
    status: str
    reason: str = ""
    key: SurfaceKey | None = None
    vertex_count: int | None = None
    triangle_count: int | None = None
    source: LiveSurfaceSource | None = None


def _require_main_thread():
    if current_thread() is not main_thread():
        raise RuntimeError("FSPT device publication must use the main thread")


class DeviceSurfaceBridge:
    """One device producer. Own selection, never reader-owned retirement."""

    def __init__(self, budget):
        _require_main_thread()
        self.pool = budget.pool
        self._budget = budget
        self._demands = {}
        self._generations = count(1)
        self._source = self._key = self._committed = None
        self._failure = ""
        self._closing = self._closed = False

    def subscribe(self, scene_token, *, required_channels=()):
        """Record active view demand and return its main-thread withdrawal."""
        _require_main_thread()
        if self._closing:
            raise RuntimeError("FSPT device publication is closing")
        token = object()
        self._demands.setdefault(scene_token, {})[token] = frozenset(required_channels)

        def withdraw():
            _require_main_thread()
            demands = self._demands.get(scene_token)
            if demands is None or demands.pop(token, None) is None:
                return
            if not demands:
                del self._demands[scene_token]
                self.invalidate(scene_token)

        return withdraw

    def _requested_channels(self, scene_token):
        demands = self._demands.get(scene_token)
        return frozenset().union(*demands.values()) if demands else None

    def _blocked(self, scene_token, channels):
        if self._closing:
            return DeviceSurfaceOffer("UNAVAILABLE", "FSPT device publication is closing")
        if channels is None:
            return DeviceSurfaceOffer("NO_DEMAND")
        if self._failure or self.pool.poisoned:
            self.invalidate()
            return DeviceSurfaceOffer("FAILED", self._failure or "surface pool is quarantined")
        if not channels <= DeviceSurfacePacket.channels:
            self.invalidate(scene_token)
            return DeviceSurfaceOffer("UNSUPPORTED", "required surface channels need evaluated mesh")
        return None

    def _fallback(self, scene_token, status, reason):
        self.invalidate(scene_token)
        return DeviceSurfaceOffer(status, reason)

    def _quarantine(self, error):
        self._failure = f"{type(error).__name__}: {error}"
        self.invalidate()
        return DeviceSurfaceOffer("FAILED", self._failure)

    def prepare(self, host, source, *, producer_stream, host_required, unsupported_reason=""):
        """Reserve and enqueue one copy. Leave ordinary host publication to the caller."""
        _require_main_thread()
        scene_token = source.scene_token
        channels = self._requested_channels(scene_token)
        blocked = self._blocked(scene_token, channels)
        if blocked is not None:
            return blocked
        if unsupported_reason:
            return self._fallback(scene_token, "UNSUPPORTED", unsupported_reason)
        if wp.__version__ != _SPARSE_BOUND_WARP_VERSION:
            return self._fallback(scene_token, "UNSUPPORTED", "requires the verified Warp 1.19.0 sparse surface route")
        try:
            solver = host.solver
            if (solver is None or not solver.owns_host_frame(host)
                    or host.frame != source.solver_frame or solver.frame != source.solver_frame
                    or solver.liquid_state_revision != source.state_revision
                    or solver.sim_time != source.simulation_time):
                return self._fallback(scene_token, "UNAVAILABLE", "stale host frame or solver source")
            ready_at = source.state_ready_at
            if ready_at is not None and (type(ready_at) not in (int, float) or not isfinite(ready_at)):
                return self._fallback(scene_token, "UNAVAILABLE", "invalid solver completion clock")
            # FluidSolver keeps an alias string. The matching pool device
            # owns the CUDA context and graph-capture state.
            device = self.pool.device
            if solver.device != device or getattr(device, "is_capturing", False):
                return self._fallback(scene_token, "UNSUPPORTED", "requires the producer CUDA device outside graph capture")
            pipeline = solver.surface
            try:
                stamp = pipeline.device_surface_stamp()
            except DeviceSurfaceUnavailable as exc:
                return self._fallback(scene_token, "UNAVAILABLE", str(exc))
            if stamp.backend != "sparse":
                return self._fallback(scene_token, "UNSUPPORTED", f"{stamp.backend} surface requires evaluated mesh")
            if self._source == source and self._key.build_stamp == stamp:
                return DeviceSurfaceOffer("PREPARED", key=self._key, source=source)

            cache = pipeline.mesh_cache
            if cache is not None and pipeline.mesh_key == stamp.build_key:
                # Waves may have already materialized geometry as part of
                # solver stepping, even without a native Blender output.
                if host_required or stamp.waves_enabled:
                    return self._fallback(scene_token, "UNSUPPORTED", "mandatory host surface was already materialized")
                nv, nt = len(cache[0]), len(cache[1])
            else:
                leaves = index(pipeline.render_extractor.grid.active_leaf_count)
                candidates = leaves * _CANDIDATES_PER_LEAF
                nv, nt = candidates * 12, candidates * 5
            if min(nv, nt) < 0 or nv > _MAX_INDEX or nt > _MAX_INDEX // 3:
                return self._fallback(scene_token, "UNSUPPORTED", "surface is outside signed 32-bit geometry limits")

            key = SurfaceKey(scene_token, source.owner_token, source.epoch, "LIVE", "GPU_FINAL",
                             next(self._generations), stamp, source.source_timeline_frame,
                             source.state_revision, source.solver_frame, source.simulation_time)
            if not self.pool.select(key):
                return self._fallback(scene_token, "UNAVAILABLE", "device surface selection was rejected")
            self._key, self._source, self._committed = key, None, None
            self.pool.poll()

            def capture(destination, expected_stamp):
                return host.copy_final_device_surface(
                    destination, expected_stamp=expected_stamp, required_channels=channels,
                    materialize_host=host_required)

            if not self._budget.submit_surface(key, nv, nt, capture, producer_stream=producer_stream):
                return self._fallback(scene_token, "DEFERRED", "surface reservation denied by pool or shared geometry budget")
            self._source = source
            return DeviceSurfaceOffer("PREPARED", key=key, source=source)
        except Exception as exc:
            # The checked pool/destination owns any uncertain CUDA resources.
            # Keep the optional error out of the normal mandatory host path.
            return self._quarantine(exc)

    def commit(self, key):
        """Acknowledge normal scene-input synchronization for the exact offer."""
        _require_main_thread()
        if (key is None or self._closing or key != self._key
                or self._blocked(key.scene_token, self._requested_channels(key.scene_token)) is not None):
            return False
        self._committed = key
        return True

    def poll(self, scene_token):
        """Check copy readiness with active demand and expose committed counts."""
        _require_main_thread()
        blocked = self._blocked(scene_token, self._requested_channels(scene_token))
        if blocked is not None:
            return blocked
        key = self._key
        if key is None or key.scene_token != scene_token:
            return DeviceSurfaceOffer("UNAVAILABLE", "no selected device surface")
        try:
            self.pool.poll()
            if self._committed != key:
                return DeviceSurfaceOffer("PENDING", "scene inputs have not committed")
            counts = self._budget.surface_ready(key)
            if counts is None:
                return DeviceSurfaceOffer("PENDING", "device copy is not ready")
            nv, nt = counts
            return DeviceSurfaceOffer("READY", key=key, vertex_count=nv, triangle_count=nt, source=self._source)
        except Exception as exc:
            return self._quarantine(exc)

    def invalidate(self, scene_token=None):
        """Retire eligibility immediately, leaving open reads with the pool."""
        _require_main_thread()
        if self._key is not None and (scene_token is None or self._key.scene_token == scene_token):
            self.pool.invalidate()
            self._key = self._source = self._committed = None

    def close(self):
        """Withdraw all demand, then drain owned submissions after readers finish."""
        _require_main_thread()
        if self._closed:
            return
        self._closing = True
        self._demands.clear()
        self.invalidate()
        self.pool.close()
        self._closed = True
