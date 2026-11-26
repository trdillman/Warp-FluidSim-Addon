"""Lazy CUDA-only reference backend; owns every array and completed image."""
from __future__ import annotations

import numpy as np

from . import BACKEND_ID, TRANSPORT_ID
from .contracts import AccumulationState, RenderFrame, frame_key


class CudaBackend:
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
        self.history = AccumulationState()
        self.scene_digest = None
        self.geometry_digest = None
        self.shading_digest = None
        self.statistics = {"geometry_uploads": 0, "mesh_builds": 0,
                           "shading_uploads": 0, "image_allocations": 0,
                           "dispatches": 0, "readbacks": 0}
        self.mesh = None
        self.arrays = None
        self.accumulation = None
        self.output = None
        self.closed = False

    def _set_scene(self, scene):
        wp = self.wp
        geometry_changed = self.geometry_digest != scene.geometry_digest
        shading_changed = self.shading_digest != scene.shading_digest
        arrays = dict(self.arrays or {})
        mesh = self.mesh
        if geometry_changed:
            arrays["points"] = wp.array(scene.vertices, dtype=wp.vec3, device=self.device)
            arrays["indices"] = wp.array(scene.triangles.ravel(), dtype=int, device=self.device)
            mesh = wp.Mesh(points=arrays["points"], indices=arrays["indices"]) if len(scene.triangles) else None
        if shading_changed:
            arrays.update(
                materials=wp.array(scene.material_indices, dtype=int, device=self.device),
                kinds=wp.array([{"diffuse": 0, "glass": 1, "emission": 2}[m.kind]
                                for m in scene.materials], dtype=int, device=self.device),
                colors=wp.array([m.color for m in scene.materials], dtype=wp.vec3, device=self.device),
                emissions=wp.array([m.emission for m in scene.materials], dtype=wp.vec3, device=self.device),
                iors=wp.array([m.ior for m in scene.materials], dtype=float, device=self.device),
            )
        # Every required replacement succeeds before accepted resources change.
        self.arrays, self.mesh, self.scene_digest = arrays, mesh, scene.digest
        self.geometry_digest, self.shading_digest = scene.geometry_digest, scene.shading_digest
        self.statistics["geometry_uploads"] += int(geometry_changed)
        self.statistics["mesh_builds"] += int(geometry_changed and mesh is not None)
        self.statistics["shading_uploads"] += int(shading_changed)

    def render(self, scene, camera, width, height, *, samples=1, max_bounces=6, seed=0):
        if self.closed:
            raise RuntimeError("FSPT backend is closed")
        if not 1 <= int(samples) <= 4096:
            raise ValueError("samples per dispatch must be between 1 and 4096")
        key = frame_key(scene, camera, width, height, max_bounces, seed)
        if width * height > 16_777_216:
            raise ValueError("FSPT P1 image exceeds its explicit 16M-pixel allocation limit")
        wp = self.wp
        if self.scene_digest != scene.digest:
            self._set_scene(scene)
        if self.history.key != key:
            accumulation = wp.zeros(width * height, dtype=wp.vec3, device=self.device)
            output = wp.empty(width * height, dtype=wp.vec4, device=self.device)
            self.accumulation, self.output = accumulation, output
            self.statistics["image_allocations"] += 1
            self.history.begin(key)
        arrays = self.arrays
        inverse = wp.mat44(*camera.inverse_view_projection.ravel().tolist())
        try:
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
        except Exception:
            # A failed dispatch/readback may have changed only part of an image.
            # Never reuse that buffer or its previous sample count on retry.
            self.history = AccumulationState()
            self.accumulation = self.output = None
            raise
        self.history.advance(int(samples))
        return RenderFrame(pixels, key, self.history.samples, str(self.output.device),
                           BACKEND_ID, TRANSPORT_ID, scene.frame, scene.publications,
                           scene.geometry_digest, tuple(self.statistics.items()))

    def close(self):
        if self.closed:
            return
        try:
            self.wp.synchronize_device(self.device)
        finally:
            self.mesh = self.arrays = self.accumulation = self.output = None
            self.scene_digest = None
            self.geometry_digest = self.shading_digest = None
            self.history = AccumulationState()
            self.closed = True
