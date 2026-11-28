"""Development-only device geometry intents and shared explicit-array accounting.

A request owns immutable host data and source identity, not GPU read rights.
The admitted backend must acquire the referenced pool on its preparation stream.
The byte budget covers pool destinations plus explicit aggregate geometry/status
arrays, including retained duplicates. BVH/internal allocator, shading, image and
solver memory require the separate observed combined-VRAM qualification.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from operator import index
from threading import current_thread

import numpy as np

from .contracts import DeviceSceneUnavailable, Material, SceneSnapshot, _owned, _rgb
from .surface_pool import SurfaceKey


class GeometryBudgetExceeded(DeviceSceneUnavailable):
    """Optional preparation cannot fit its explicit geometry-array reservation."""


@dataclass(frozen=True, eq=False)
class DeviceGeometryReference:
    pool: object = field(repr=False)
    key: SurfaceKey
    vertex_count: int
    triangle_count: int
    material_index: int

    def __post_init__(self):
        if not isinstance(self.key, SurfaceKey):
            raise ValueError('device geometry requires a validated SurfaceKey')
        for name in ('vertex_count', 'triangle_count', 'material_index'):
            object.__setattr__(self, name, index(getattr(self, name)))
        if (min(self.vertex_count, self.triangle_count, self.material_index) < 0
                or (self.triangle_count and not self.vertex_count)
                or self.vertex_count > np.iinfo(np.int32).max
                or self.triangle_count > np.iinfo(np.int32).max // 3):
            raise ValueError('invalid device geometry counts or material')

    @property
    def identity(self):
        key, stamp = self.key, self.key.build_stamp
        # The opaque producer owner is a process-local identity. Never deepcopy
        # it with asdict(), or stringify a newly copied object's address.
        return (key.scene_token, key.owner_token, key.epoch, key.source, key.channel,
                key.generation, key.source_timeline_frame, key.state_revision,
                key.solver_frame, key.simulation_time, id(stamp.owner), stamp.build_key,
                stamp.backend, stamp.revision, stamp.solver_frame, stamp.solver_time,
                stamp.waves_enabled, stamp.wave_frame, stamp.wave_revision,
                stamp.wave_settings, self.vertex_count, self.triangle_count)


@dataclass(frozen=True)
class PartLayout:
    vertex_offset: int
    index_offset: int
    vertex_count: int
    triangle_count: int


@dataclass(frozen=True, eq=False)
class DeviceSceneRequest:
    parts: tuple
    materials: tuple[Material, ...]
    world: tuple[float, float, float]
    frame: int
    publications: tuple = ()
    layout: tuple[PartLayout, ...] = field(init=False)
    vertex_count: int = field(init=False)
    triangle_count: int = field(init=False)
    geometry_digest: str = field(init=False)
    shading_digest: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self):
        parts, materials = tuple(self.parts), tuple(self.materials)
        if sum(isinstance(p, DeviceGeometryReference) for p in parts) != 1:
            raise ValueError('device scene requires exactly one GPU reference')
        if not materials or any(not isinstance(m, Material) for m in materials):
            raise ValueError('scene requires validated materials')
        world, frame = _rgb(self.world, 'world'), index(self.frame)
        publications = tuple(self.publications)
        if any(not is_dataclass(p) or not p.__dataclass_params__.frozen for p in publications):
            raise ValueError('surface publications must be immutable records')
        layout, shading_parts, identities = [], [], []
        nv = nt = 0
        for part in parts:
            if isinstance(part, DeviceGeometryReference):
                pv, pt = part.vertex_count, part.triangle_count
                if part.material_index >= len(materials):
                    raise ValueError('device material outside scene materials')
                shading_parts.append(('GPU_FINAL', pt, part.material_index))
                identities.append(('GPU_FINAL', part.identity))
            elif isinstance(part, SceneSnapshot):
                pv, pt = len(part.vertices), len(part.triangles)
                if part.materials != materials:
                    raise ValueError('host part must use the scene material table')
                shading_parts.append(('HOST', hashlib.sha256(part.material_indices.tobytes()).hexdigest()))
                identities.append(('HOST', part.geometry_digest))
            else:
                raise TypeError('expected immutable host part or device geometry reference')
            if nv + pv > np.iinfo(np.int32).max or nt + pt > np.iinfo(np.int32).max // 3:
                raise ValueError('aggregate geometry exceeds int32 offsets')
            layout.append(PartLayout(nv, 3 * nt, pv, pt))
            nv, nt = nv + pv, nt + pt
        geometry = hashlib.sha256(json.dumps(identities, allow_nan=False).encode())
        shading = hashlib.sha256(json.dumps(shading_parts, allow_nan=False).encode())
        shading.update(json.dumps([(m.kind, m.color, m.emission, m.ior) for m in materials],
                                  allow_nan=False).encode())
        digest = hashlib.sha256(geometry.digest() + shading.digest())
        digest.update(json.dumps([world, frame, [asdict(p) for p in publications]],
                                 allow_nan=False).encode())
        for name, value in (('parts', parts), ('materials', materials), ('world', world),
                            ('frame', frame), ('publications', publications),
                            ('layout', tuple(layout)), ('vertex_count', nv), ('triangle_count', nt),
                            ('geometry_digest', 'gpu-identity:' + geometry.hexdigest()),
                            ('shading_digest', shading.hexdigest()), ('digest', digest.hexdigest())):
            object.__setattr__(self, name, value)

    @property
    def material_indices(self):
        # Materialize only after backend admission; requests with large device
        # counts must remain small immutable metadata while capture is denied.
        assignments = [np.full(p.triangle_count, p.material_index, np.int32)
                       if isinstance(p, DeviceGeometryReference) else p.material_indices
                       for p in self.parts]
        return _owned(np.concatenate(assignments), np.int32, (0,), 'material indices')

    @property
    def reference(self):
        return next(p for p in self.parts if isinstance(p, DeviceGeometryReference))


class _ArrayReservation:
    def __init__(self, budget, size, generation):
        self.budget, self.size, self.generation = budget, size, generation

    def release(self):
        if self.budget is not None:
            self.budget._check_thread()
            self.budget._reservations.remove(self)
            self.budget = None


class GeometryArrayBudget:
    """One shared ledger for all consumers of one device's SurfacePool.

    All new producer submissions must use submit_surface() while this ledger is
    attached. A direct SurfacePool.submit() bypasses combined array admission.
    """
    def __init__(self, pool, *, byte_limit):
        if getattr(pool, '_geometry_array_budget', None) is not None:
            raise ValueError('surface pool already has a shared geometry budget')
        self.pool, self.byte_limit = pool, index(byte_limit)
        if self.byte_limit < 0:
            raise ValueError('negative geometry byte limit')
        if pool.reserved_bytes > self.byte_limit:
            raise GeometryBudgetExceeded('existing pool already exceeds combined array budget')
        self._thread = current_thread()
        self._reservations = set()
        self._retiring = []
        pool._geometry_array_budget = self
        self.peak_bytes = self.reserved_bytes
        self.denials = 0
        self.peak_generations = self.generation_count

    def _check_thread(self):
        if current_thread() is not self._thread:
            raise RuntimeError('use geometry admission on its publication thread')

    @property
    def reserved_bytes(self):
        return self.pool.reserved_bytes + sum(r.size for r in self._reservations)

    def _generation_count(self, extra_generation=None, extra_surface=None):
        generations = {r.generation for r in self._reservations}
        if extra_generation is not None:
            generations.add(extra_generation)
        sources = {slot.key for slot in getattr(self.pool, '_slots', ())}
        if extra_surface is not None:
            sources.add(extra_surface)
        # A packet and its aggregate are components of one generation. Distinct
        # static geometry captures of the same fluid packet still count apart.
        represented = {generation[0] for generation in generations}
        return len(generations) + len(sources - represented)

    @property
    def generation_count(self):
        return self._generation_count()

    def check_reservation(self, size, *, generation=None):
        self._check_thread()
        size = index(size)
        if size < 0:
            raise ValueError('negative geometry reservation')
        if generation is None:
            generation = (object(), 'unscoped geometry allocation')
        elif (type(generation) is not tuple or len(generation) != 2
              or not isinstance(generation[0], SurfaceKey) or type(generation[1]) is not str):
            raise ValueError('invalid aggregate generation identity')
        if self._generation_count(extra_generation=generation) > 2:
            self.denials += 1
            raise GeometryBudgetExceeded('two geometry generations are already retained')
        if self.reserved_bytes + size > self.byte_limit:
            self.denials += 1
            raise GeometryBudgetExceeded('aggregate geometry exceeds shared array budget')
        return size, generation

    def reserve(self, size, *, generation=None):
        size, generation = self.check_reservation(size, generation=generation)
        claim = _ArrayReservation(self, size, generation)
        self._reservations.add(claim)
        self.peak_generations = max(self.peak_generations, self.generation_count)
        self.peak_bytes = max(self.peak_bytes, self.reserved_bytes)
        return claim

    def defer_release(self, reservation, lease, release_owners):
        """Keep bytes/generation charged until the pool checks the last read.

        release_owners is a synchronous ownership operation, not a CUDA drain.
        The pool calls collect_retired only after its checked fence or close.
        """
        self._check_thread()
        if reservation.budget is not self or reservation not in self._reservations:
            raise ValueError('geometry reservation belongs to another ledger')
        if any(item[0] is reservation for item in self._retiring):
            raise ValueError('geometry reservation is already retiring')
        self._retiring.append((reservation, lease, release_owners))

    def collect_retired(self):
        self._check_thread()
        for item in tuple(self._retiring):
            reservation, lease, release_owners = item
            if lease.retired:
                release_owners()
                reservation.release()
                self._retiring.remove(item)

    def surface_ready(self, key):
        """Return active counts only; no lease, poll, extraction or GPU access.

        Readiness offers may retire before backend admission. The consumer must
        still acquire/revalidate the selected reference on its own stream.
        """
        self._check_thread()
        pool = self.pool
        pool._check_thread()
        if pool._closing or pool._closed or pool.poisoned or key != pool._key:
            return None
        for slot in pool._slots:
            if slot.key == key and slot.ready and slot.failure is None:
                return (slot.packet.vertex_count, slot.packet.triangle_count)
        return None

    def submit_surface(self, key, vertex_capacity, triangle_capacity, capture, *, producer_stream):
        self._check_thread()
        nv, nt = index(vertex_capacity), index(triangle_capacity)
        if min(nv, nt) < 0:
            raise ValueError('negative surface capacity')
        if (self._generation_count(extra_surface=key) > 2
                or self.reserved_bytes + 12 * (nv + nt) > self.byte_limit):
            self.denials += 1
            return False
        try:
            return self.pool.submit(key, nv, nt, capture, producer_stream=producer_stream)
        finally:
            self.peak_bytes = max(self.peak_bytes, self.reserved_bytes)
            self.peak_generations = max(self.peak_generations, self.generation_count)
