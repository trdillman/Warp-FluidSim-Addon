"""Opt-in GPU_FINAL consumer, intentionally not connected to Blender publication.

Each accepted aggregate retains its SurfacePool lease for its entire generation.
A replacement also retains its predecessor until checked job retirement. Thus
packet slots account for accepted, candidate and retired aggregate generations.
Every device scene uses the reference fresh single BVH and original face order.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import current_thread

import numpy as np

from .async_backend import AsyncCudaBackend
from .device_scene import DeviceGeometryReference, DeviceSceneRequest, DeviceSceneUnavailable


@dataclass(eq=False)
class _Generation:
    lease: object
    reservation: object
    arrays: object = None
    mesh: object = None
    validation_host: object = None

    def release_owners(self):
        self.arrays = self.mesh = self.validation_host = None


class DeviceAsyncCudaBackend(AsyncCudaBackend):
    """Development-only consumer. Callers supply one shared pool/array budget."""

    def __init__(self, device='cuda:0', *, geometry_budget):
        super().__init__(device)
        self._initialize_device_resources(geometry_budget)
        if self.wp.get_device(geometry_budget.pool.device) != self.device:
            raise ValueError('geometry budget and renderer device differ')

    def _initialize_device_resources(self, budget):
        self.geometry_budget = budget
        self._publication_thread = current_thread()
        self._device_current = None
        self._device_retired = []
        self.statistics.update(device_geometry_copies=0, device_geometry_copy_bytes=0,
                               geometry_array_reserved_bytes=0,
                               geometry_array_peak_reserved_bytes=budget.peak_bytes,
                               geometry_digest_is_identity=1)

    def _submit(self, scene, camera, width, height, samples, max_bounces, seed, final):
        if self.closed:
            raise RuntimeError('FSPT backend is closed')
        if self.pending:
            return False
        if current_thread() is not self._publication_thread:
            raise RuntimeError('GPU scene preparation requires its publication thread')
        self._release_retired_geometry()
        if not isinstance(scene, DeviceSceneRequest):
            raise TypeError('development GPU consumer requires DeviceSceneRequest')
        try:
            reference = scene.reference
            if reference.pool is not self.geometry_budget.pool:
                raise DeviceSceneUnavailable('device scene references a different geometry pool')
            if self.geometry_digest != scene.geometry_digest:
                # A capacity/stale-intent refusal queued no work and must never
                # enter the base abort/drain path behind an older reader fence.
                self.geometry_budget.check_reservation(
                    12 * (scene.vertex_count + scene.triangle_count) + 4,
                    generation=(reference.key, scene.geometry_digest))
                ready = self.geometry_budget.surface_ready(reference.key)
                if ready != (reference.vertex_count, reference.triangle_count):
                    raise DeviceSceneUnavailable('captured GPU surface retired or changed; recapture')
        except DeviceSceneUnavailable:
            # An earlier busy attempt may already have queued this owner. A
            # now-unrenderable request must not remain ahead of healthy peers.
            self.arbiter.cancel(self.owner)
            raise
        return super()._submit(scene, camera, width, height, samples, max_bounces, seed, final)

    def _upload_shading(self, scene, arrays):
        wp = self.wp
        kinds = {'diffuse': 0, 'glass': 1, 'emission': 2}
        arrays['materials'] = self._upload(scene.material_indices, int, np.int32)
        arrays['kinds'] = self._upload([kinds[m.kind] for m in scene.materials], int, np.int32)
        arrays['colors'] = self._upload([m.color for m in scene.materials], wp.vec3, np.float32)
        arrays['emissions'] = self._upload([m.emission for m in scene.materials], wp.vec3, np.float32)
        arrays['iors'] = self._upload([m.ior for m in scene.materials], float, np.float32)

    def _host_copy(self, destination, values, dtype, offset):
        if not len(values):
            return
        host = self._retain(np.ascontiguousarray(values))
        view = self._retain(self.wp.array(host, dtype=dtype, device='cpu', copy=False))
        self.wp.copy(destination, view, dest_offset=offset, count=len(values))

    def _prepare_device_geometry(self, scene):
        wp = self.wp
        reference = scene.reference
        if reference.pool is not self.geometry_budget.pool:
            raise DeviceSceneUnavailable('device scene references a different geometry pool')
        if wp.get_device(reference.pool.device) != self.device:
            raise DeviceSceneUnavailable('device scene references a different CUDA device')
        # Geometry, rebased indices, and bounded validation status are all
        # charged before any aggregate allocation. Native combined memory also
        # measures opaque BVH, shading, history, allocator and solver storage.
        reservation = self.geometry_budget.reserve(
            12 * (scene.vertex_count + scene.triangle_count) + 4,
            generation=(reference.key, scene.geometry_digest))
        lease = None
        try:
            lease = reference.pool.acquire(reference.key, self._prepare_stream)
            if lease is None:
                raise DeviceSceneUnavailable('captured GPU surface retired or is not ready; recapture')
        except BaseException:
            reservation.release()
            raise
        generation = _Generation(lease, reservation)
        lease.retain(generation)
        # Before the first allocation/copy, both the job and pool reader own the
        # generation. Failed preparation keeps this record until a checked drain.
        self._device_retired.append(generation)
        packet = lease.packet
        if (packet.device != self.device or packet.stamp != reference.key.build_stamp
                or packet.vertex_count != reference.vertex_count
                or packet.triangle_count != reference.triangle_count
                or packet.coordinate_space != 'WORLD'
                or not {'positions', 'triangle_indices'} <= packet.channels):
            raise DeviceSceneUnavailable('GPU packet does not match captured geometry metadata')
        for array, dtype, stride in ((packet.positions, wp.vec3, 12), (packet.indices, wp.int32, 4)):
            if (array.dtype != dtype or array.ndim != 1 or array.device != self.device
                    or not array.is_contiguous or array.strides != (stride,)):
                raise DeviceSceneUnavailable('GPU packet has incompatible array layout')
        points = self._retain(wp.empty(scene.vertex_count, dtype=wp.vec3, device=self.device))
        indices = self._retain(wp.empty(3 * scene.triangle_count, dtype=wp.int32, device=self.device))
        status = self._retain(wp.zeros(1, dtype=wp.int32, device=self.device))
        generation.arrays = {'points': points, 'indices': indices, 'validation': status}
        generation.validation_host = self._retain(wp.empty(1, dtype=wp.int32, device='cpu',
                                                          pinned=self.device.is_cuda))
        from .device_scene_kernels import copy_indices, copy_positions
        for part, layout in zip(scene.parts, scene.layout, strict=True):
            if isinstance(part, DeviceGeometryReference):
                if part.vertex_count:
                    wp.launch(copy_positions, dim=part.vertex_count,
                              inputs=[packet.positions, points, layout.vertex_offset, status],
                              device=self.device)
                if part.triangle_count:
                    wp.launch(copy_indices, dim=3 * part.triangle_count,
                              inputs=[packet.indices, indices, layout.index_offset,
                                      layout.vertex_offset, part.vertex_count, status], device=self.device)
            else:
                self._host_copy(points, part.vertices, wp.vec3, layout.vertex_offset)
                self._host_copy(indices, part.triangles.ravel() + layout.vertex_offset,
                                wp.int32, layout.index_offset)
        generation.mesh = self._build_mesh(points, indices) if scene.triangle_count else None
        return generation

    def _set_scene(self, scene):
        if not isinstance(scene, DeviceSceneRequest):
            raise TypeError('development GPU consumer requires DeviceSceneRequest')
        changed = self.geometry_digest != scene.geometry_digest
        shading = self.shading_digest != scene.shading_digest
        if changed:
            candidate = self._prepare_device_geometry(scene)
            arrays = candidate.arrays
            # A new scene owns its full material binding even if it happens to
            # equal the old table. No mutable assignment array is aliased.
            self._upload_shading(scene, arrays)
            if self._device_current is not None:
                self._device_retired.append(self._device_current)
            self._device_retired.remove(candidate)
            self._device_current = candidate
            self.arrays, self.mesh = arrays, candidate.mesh
            self.statistics['geometry_uploads'] += 1
            self.statistics['mesh_builds'] += int(candidate.mesh is not None)
            self.statistics['geometry_array_allocations'] += 3
            self.statistics['device_geometry_copies'] += 1
            self.statistics['device_geometry_copy_bytes'] += 12 * (
                scene.reference.vertex_count + scene.reference.triangle_count)
            self.statistics['shading_uploads'] += 1
        elif shading:
            arrays = dict(self.arrays)
            self._retain(self.arrays)
            self._upload_shading(scene, arrays)
            self.arrays = self._device_current.arrays = arrays
            self.statistics['shading_uploads'] += 1
        self.scene_digest = scene.digest
        self.geometry_digest, self.shading_digest = scene.geometry_digest, scene.shading_digest
        self.statistics['geometry_array_reserved_bytes'] = self.geometry_budget.reserved_bytes
        self.statistics['geometry_array_peak_reserved_bytes'] = self.geometry_budget.peak_bytes

    def _queue_geometry_validation(self, stream):
        generation = self._device_current
        self.wp.copy(generation.validation_host, generation.arrays['validation'], count=1, stream=stream)

    def _validate_completed_geometry(self):
        if int(self._device_current.validation_host.numpy()[0]):
            raise RuntimeError('FSPT rejected invalid device geometry before presentation')

    def _release_retired_geometry(self):
        # Called only after checked job completion/owned-stream drain. The
        # ledger still owns/charges these aggregates until pool.poll/close proves
        # the new last-read fence retired. Open accepted leases survive refinement.
        if current_thread() is not self._publication_thread:
            # Final suspension may collect a checked job on Blender's render
            # worker. Keep the records, leases and byte claims strongly owned;
            # only the next publication-thread pump may mutate the pool/ledger.
            return False
        while self._device_retired:
            generation = self._device_retired[0]
            generation.lease.finish()
            self.geometry_budget.defer_release(generation.reservation, generation.lease,
                                               generation.release_owners)
            self._device_retired.pop(0)
        self.statistics['geometry_array_reserved_bytes'] = self.geometry_budget.reserved_bytes
        return True

    def retire_completed_geometry(self):
        """Main-thread, nonblocking cleanup after a checked worker collection."""
        if current_thread() is not self._publication_thread:
            raise RuntimeError('geometry retirement requires its publication thread')
        if not self._state_lock.acquire(blocking=False):
            return False
        try:
            if self.pending:
                return False
            return self._release_retired_geometry()
        finally:
            self._state_lock.release()

    def _retire(self):
        try:
            self._release_retired_geometry()
        except BaseException as exc:
            self._job.failure = exc
            self.owner.quarantined = self
            raise
        super()._retire()

    def _forget_scene(self):
        current = getattr(self, '_device_current', None)
        if current is not None:
            self._device_retired.append(current)
            self._device_current = None
        super()._forget_scene()

    def close(self):
        with self._state_lock:
            if not self.closed:
                super().close()
            self._release_retired_geometry()
