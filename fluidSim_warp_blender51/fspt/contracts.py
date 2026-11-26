"""Owned scene snapshots and image identity, independent of bpy and Warp."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

MAX_BOUNCES = 64


def _rgb(value, name, *, bounded=False):
    result = tuple(float(x) for x in value)
    if len(result) != 3 or not all(math.isfinite(x) and x >= 0 for x in result):
        raise ValueError(f"{name} must contain three finite nonnegative values")
    if bounded and max(result) > 1:
        raise ValueError(f"{name} reflectance must not exceed one")
    return result


@dataclass(frozen=True)
class Material:
    kind: str = "diffuse"
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ior: float = 1.45

    def __post_init__(self):
        if self.kind not in {"diffuse", "glass", "emission"}:
            raise ValueError(f"unsupported material kind: {self.kind}")
        object.__setattr__(self, "color", _rgb(self.color, "color", bounded=True))
        object.__setattr__(self, "emission", _rgb(self.emission, "emission"))
        if not math.isfinite(self.ior) or self.ior <= 0:
            raise ValueError("IOR must be finite and positive")


def _owned(value, dtype, shape, name):
    array = np.asarray(value)
    if array.size == 0:
        array = np.empty(shape, dtype=dtype)
    if np.issubdtype(dtype, np.integer) and array.size:
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain finite integer indices")
        limit = np.iinfo(dtype)
        if array.min() < limit.min or array.max() > limit.max:
            raise ValueError(f"{name} indices exceed supported range")
    array = np.array(array, dtype=dtype, order="C", copy=True)
    if len(array.shape) != len(shape) or array.shape[1:] != shape[1:]:
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    # A bytes owner makes write protection irreversible, unlike setflags(False)
    # on an owning ndarray, which callers can turn writable again.
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class SceneSnapshot:
    vertices: np.ndarray
    triangles: np.ndarray
    material_indices: np.ndarray
    materials: tuple[Material, ...]
    world: tuple[float, float, float]
    frame: int
    digest: str = field(init=False)

    def __post_init__(self):
        vertices = _owned(self.vertices, np.float32, (0, 3), "vertices")
        triangles = _owned(self.triangles, np.int32, (0, 3), "triangles")
        indices = _owned(self.material_indices, np.int32, (0,), "material indices")
        materials = tuple(self.materials)
        world = _rgb(self.world, "world")
        if not materials or not all(isinstance(m, Material) for m in materials):
            raise ValueError("scene requires validated materials")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError("triangle index outside vertices")
        if len(indices) != len(triangles) or (indices.size and
                (indices.min() < 0 or indices.max() >= len(materials))):
            raise ValueError("material index outside materials or wrong count")
        for name, value in (("vertices", vertices), ("triangles", triangles),
                            ("material_indices", indices), ("materials", materials), ("world", world)):
            object.__setattr__(self, name, value)
        digest = hashlib.sha256()
        for array in (vertices, triangles, indices):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(json.dumps([[(m.kind, m.color, m.emission, m.ior) for m in materials],
                                  world, int(self.frame)], allow_nan=False).encode())
        object.__setattr__(self, "digest", digest.hexdigest())


@dataclass(frozen=True, eq=False)
class CameraSnapshot:
    inverse_view_projection: np.ndarray
    digest: str = field(init=False)

    def __post_init__(self):
        matrix = _owned(self.inverse_view_projection, np.float32, (4, 4), "camera matrix")
        if matrix.shape != (4, 4) or abs(float(np.linalg.det(matrix))) < 1e-20:
            raise ValueError("camera matrix must be invertible and 4 by 4")
        # Invertibility alone does not ensure finite near/far unprojection.
        # Extreme clip ratios can round the far plane to homogeneous w=0.
        corners = np.array([(x, y, z, 1) for x in (-1, 1) for y in (-1, 1)
                            for z in (-1, 1)], dtype=np.float64)
        projected = corners @ matrix.astype(np.float64).T
        weights = projected[:, 3]
        if np.any(np.abs(weights) < 1e-12) or weights.min() * weights.max() < 0:
            raise ValueError("camera clip projection is not finite; reduce the near/far ratio")
        endpoints = projected[:, :3] / weights[:, None]
        if (not np.isfinite(endpoints).all()
                or np.max(np.linalg.norm(endpoints, axis=1)) > np.sqrt(np.finfo(np.float32).max) / 4):
            raise ValueError("camera clip projection exceeds float32 ray bounds")
        object.__setattr__(self, "inverse_view_projection", matrix)
        object.__setattr__(self, "digest", hashlib.sha256(matrix.tobytes()).hexdigest())


def frame_key(scene, camera, width, height, max_bounces, seed):
    if not (1 <= width <= 16384 and 1 <= height <= 16384 and 1 <= max_bounces <= MAX_BOUNCES):
        raise ValueError("invalid render dimensions or bounce limit")
    return (scene.digest, camera.digest, int(width), int(height), int(max_bounces), int(seed))


@dataclass
class AccumulationState:
    key: tuple | None = None
    samples: int = 0

    def begin(self, key):
        changed = key != self.key
        if changed:
            self.key, self.samples = key, 0
        return changed

    def advance(self, samples):
        if samples < 1 or self.key is None:
            raise ValueError("cannot accumulate without an active image identity")
        self.samples += int(samples)


def require_current(expected, received):
    if expected != received:
        raise RuntimeError("FSPT refused a stale image identity")


@dataclass(frozen=True)
class RenderFrame:
    pixels: np.ndarray
    key: tuple
    samples: int
    device: str
    backend: str
    transport: str
