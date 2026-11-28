"""Lazy CUDA-only reference backend; owns every array and completed image."""
from __future__ import annotations

from threading import RLock

import numpy as np

from . import BACKEND_ID, TRANSPORT_ID
from .contracts import AccumulationState, RenderFrame, frame_key


class CudaBackend:
    # A CUDA synchronization error can outlive the view that submitted work.
    # Pin buffers and partial preparation frames until explicit retry/close
    # proves that prior device readers retired.
    _uncertain = {}

    def __new__(cls, *args, **kwargs):
        backend = super().__new__(cls)
        backend._sync_lock = RLock()
        backend._pending_resources = []
        return backend

    def __init__(self, device="cuda:0"):
        if not str(device).startswith("cuda:"):
            raise ValueError("FSPT requires an explicit CUDA device; CPU fallback is forbidden")
        from ..runtime.warp_runtime import _import_warp, probe_runtime, runtime_status_for_device

        state = probe_runtime()
        ready, diagnostic = runtime_status_for_device(str(device), probe=state)
        if not ready:
            raise RuntimeError(diagnostic)
        self.wp = _import_warp()
        self.device = self.wp.get_device(str(device))
        if not self.device.is_cuda:
            raise RuntimeError("FSPT workload must execute on CUDA")
        from .kernels import trace_paths

        self.kernel = trace_paths
        self._initialize_resources()

    def _initialize_resources(self):
        self.history = AccumulationState()
        self.scene_digest = None
        self.geometry_digest = None
        self.shading_digest = None
        self.statistics = {"geometry_uploads": 0, "mesh_builds": 0,
                           "geometry_array_allocations": 0, "position_upload_bytes": 0,
                           "index_upload_bytes": 0, "position_upload_ranges": 0, "index_upload_ranges": 0,
                           "shading_uploads": 0, "image_allocations": 0,
                           "dispatches": 0, "readbacks": 0}
        self.mesh = None
        self.arrays = None
        self.accumulation = None
        self.output = None
        self.closed = False

    def _retain(self, owner):
        # An admitted submission owns each new buffer from allocation through
        # reader/copy retirement, even if a later allocation or drain fails.
        self._pending_resources.append(owner)
        return owner

    def _retain_dispatch(self, *transfer_owners):
        # A mesh ID in the launch arguments is only a scalar. Pin the actual
        # BVH, all array arguments and transfer destinations for every dispatch,
        # including when scene preparation reused every accepted allocation.
        self._pending_resources.extend((self.mesh, self.arrays, self.accumulation, self.output, *transfer_owners))

    def _upload(self, values, dtype, host_dtype):
        host = self._retain(np.ascontiguousarray(values, dtype=host_dtype))
        return self._retain(self.wp.array(host, dtype=dtype, device=self.device))

    def _build_mesh(self, points, indices):
        return self._retain(self.wp.Mesh(points=points, indices=indices))

    def _update_buffer(self, arrays, label, host, dtype, layout, old_layout, counts):
        storage_key = label + "_storage"
        capacity_key = label + "_capacity"
        store = arrays.get(storage_key)
        capacity = arrays.get(capacity_key, 0)
        if not len(host):
            return None
        if store is None or len(host) > capacity or not layout:
            # Legacy callers without spans retain their one-shot upload path.
            store = self._upload(host, dtype, host.dtype)
            arrays[storage_key], arrays[capacity_key] = store, len(host)
            counts["geometry_array_allocations"] += 1
            counts[label + "_upload_bytes"] += host.nbytes
            counts[label + "_upload_ranges"] += 1
            return store
        for key, span in layout.items():
            start, count, _digest = span
            if not count or old_layout.get(key) == span:
                continue
            stop = start + count
            source = self._retain(host[start:stop])
            host_view = self._retain(self.wp.array(source, dtype=dtype, device="cpu", copy=False))
            self.wp.copy(store, host_view, dest_offset=start, count=count)
            counts[label + "_upload_bytes"] += source.nbytes
            counts[label + "_upload_ranges"] += 1
        if len(host) == capacity:
            return store
        return self._retain(store[:len(host)])

    def _set_scene(self, scene):
        wp = self.wp
        arrays = dict(self.arrays or {})
        shading_changed = self.shading_digest != scene.shading_digest
        if shading_changed:
            kinds = {"diffuse": 0, "glass": 1, "emission": 2}
            arrays["materials"] = self._upload(scene.material_indices, int, np.int32)
            arrays["kinds"] = self._upload([kinds[m.kind] for m in scene.materials], int, np.int32)
            arrays["colors"] = self._upload([m.color for m in scene.materials], wp.vec3, np.float32)
            arrays["emissions"] = self._upload([m.emission for m in scene.materials], wp.vec3, np.float32)
            arrays["iors"] = self._upload([m.ior for m in scene.materials], float, np.float32)
        geometry_changed = self.geometry_digest != scene.geometry_digest
        layout = {a.key: a.position for a in scene.assets} if scene.asset_spans else {}
        index_layout = {a.key: a.index for a in scene.assets} if scene.asset_spans else {}
        mesh = self.mesh
        counts = dict.fromkeys(("geometry_array_allocations", "position_upload_bytes",
                                "index_upload_bytes", "position_upload_ranges", "index_upload_ranges"), 0)
        if geometry_changed:
            # The accepted mesh aliases its inputs. From the first attempted
            # write onward its old identity cannot be restored by reassigning
            # the old mesh handle. Readers have already retired on this lane.
            self._retain(self.arrays)
            self._retain(self.mesh)
            self.scene_digest = self.geometry_digest = None
            try:
                points = self._update_buffer(arrays, "position", scene.vertices, wp.vec3,
                                             layout, arrays.get("position_layout", {}), counts)
                indices = self._update_buffer(arrays, "index", scene.triangles.ravel(), int,
                                              index_layout, arrays.get("index_layout", {}), counts)
                # Fresh construction preserves the flattened traversal's tie
                # behavior, including ties between faces of the same object.
                mesh = self._build_mesh(points, indices) if len(scene.triangles) else None
            except BaseException:
                self._forget_scene()
                raise
            arrays["points"], arrays["indices"] = points, indices
        arrays["position_layout"], arrays["index_layout"] = layout, index_layout
        self.arrays, self.mesh, self.scene_digest = arrays, mesh, scene.digest
        self.geometry_digest, self.shading_digest = scene.geometry_digest, scene.shading_digest
        self.statistics["geometry_uploads"] += int(geometry_changed)
        self.statistics["mesh_builds"] += int(geometry_changed and mesh is not None)
        self.statistics["shading_uploads"] += int(shading_changed)
        for key, count in counts.items():
            self.statistics[key] = self.statistics.get(key, 0) + count

    def render(self, scene, camera, width, height, *, samples=1, max_bounces=6, seed=0):
        with self._sync_lock:
            return self._render(scene, camera, width, height, samples, max_bounces, seed)

    def _render(self, scene, camera, width, height, samples, max_bounces, seed):
        if self.closed:
            raise RuntimeError("FSPT backend is closed")
        if not 1 <= int(samples) <= 4096:
            raise ValueError("samples per dispatch must be between 1 and 4096")
        key = frame_key(scene, camera, width, height, max_bounces, seed)
        if width * height > 16_777_216:
            raise ValueError("FSPT P1 image exceeds its explicit 16M-pixel allocation limit")
        wp = self.wp
        if self in self._uncertain:
            wp.synchronize_device(self.device)
            del self._uncertain[self]
            self._pending_resources.clear()
        try:
            if self.scene_digest != scene.digest:
                self._set_scene(scene)
            if self.history.key != key:
                accumulation = self._retain(wp.zeros(width * height, dtype=wp.vec3, device=self.device))
                output = self._retain(wp.empty(width * height, dtype=wp.vec4, device=self.device))
                self.accumulation, self.output = accumulation, output
                self.statistics["image_allocations"] += 1
                self.history.begin(key)
            self._retain_dispatch()
            arrays = self.arrays
            inverse = wp.mat44(*camera.inverse_view_projection.ravel().tolist())
            wp.launch(self.kernel, dim=width * height, inputs=[
                self.mesh.id if self.mesh else wp.uint64(0), len(scene.triangles),
                arrays["materials"], arrays["kinds"], arrays["colors"], arrays["emissions"], arrays["iors"],
                wp.vec3(*scene.world), inverse, width, height, self.history.samples,
                int(samples), int(max_bounces), int(seed), self.accumulation, self.output,
            ], device=self.device)
            self.statistics["dispatches"] += 1
            wp.synchronize_device(self.device)
            pixels = self.output.numpy().reshape(height, width, 4)
            self.statistics["readbacks"] += 1
            if not np.isfinite(pixels).all():
                raise RuntimeError("FSPT CUDA output contains nonfinite values")
        except BaseException as exc:
            try:
                wp.synchronize_device(self.device)
            except BaseException:
                self._uncertain[self] = (exc, self.arrays, self.accumulation, self.output)
            else:
                self._pending_resources.clear()
            if self.scene_digest == scene.digest:
                # An asynchronous upload/build error can first surface here.
                self._forget_scene()
            self.history = AccumulationState()
            self.accumulation = self.output = None
            raise
        self._pending_resources.clear()
        self.history.advance(int(samples))
        return RenderFrame(pixels, key, self.history.samples, str(self.output.device),
                           BACKEND_ID, TRANSPORT_ID, scene.frame, scene.publications,
                           scene.geometry_digest, tuple(self.statistics.items()))

    def _forget_scene(self):
        self.arrays = self.mesh = None
        self.scene_digest = self.geometry_digest = self.shading_digest = None

    def close(self):
        with self._sync_lock:
            if self.closed and self not in self._uncertain:
                return
            try:
                self.wp.synchronize_device(self.device)
                self._uncertain.pop(self, None)
                self._pending_resources.clear()
            finally:
                self._forget_scene()
                self.accumulation = self.output = None
                self.history = AccumulationState()
                self.closed = True
