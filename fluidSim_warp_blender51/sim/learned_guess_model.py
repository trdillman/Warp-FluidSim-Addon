"""Learned pressure guess: weights schema and the numpy reference model.

Pure numpy on purpose: the Warp inference (``learned_guess.py``), the PyTorch
training tool (``tools/learned_guess/train.py``) and the tests all share these
definitions, so the three implementations cannot drift apart. No Warp, no
torch, no bpy here.

Model (``learned_guess_v1``), a small Fourier neural operator on the coarse
grid that has one cell per sparse-grid leaf, padded up to powers of two:

* input, 4 channels per coarse cell: the leaf's summed pressure right-hand
  side, its fluid fraction, its solid fraction and the previous solve's
  leaf-mean pressure (all as described in ``coarse_features``);
* the right-hand-side and previous-pressure channels are divided by one
  per-sample scale ``s`` (the RMS of the right-hand-side channel) and the
  output is multiplied by ``s``, so the guess scales linearly with the
  problem, as the true solution does;
* lift (1x1) -> ``layers`` x [truncated spectral convolution + 1x1, GELU
  (tanh form) except after the last layer] -> 1x1 -> GELU -> 1x1 = one
  pressure value per coarse cell.

The spectral convolution keeps ``2*modes-1`` signed frequencies on x and y
and ``modes`` non-negative ones on z (the real-input half spectrum). A
frequency the padded axis cannot represent without aliasing
(``|k| > (n-1)//2``) is dropped by zeroing its basis row, so one weights file
serves every domain size. Array layout is ``[channel, z, y, x]`` (x fastest),
matching the flat coarse cell index ``x + nx*(y + ny*z)``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

SCHEMA = "learned_guess_v1"
IN_CHANNELS = 4
MAX_WEIGHTS_BYTES = 5 * 1024 * 1024
_GELU_C = math.sqrt(2.0 / math.pi)


class WeightsError(ValueError):
    """The weights cannot be used; ``str(exc)`` is the user-facing reason."""


@dataclass(frozen=True)
class Model:
    width: int
    modes: int
    layers: int
    proj_hidden: int
    lift_w: np.ndarray        # (C, IN)
    lift_b: np.ndarray        # (C,)
    spec_re: tuple            # per layer (C_in, C_out, Kz, Ky, Kx)
    spec_im: tuple
    pw_w: tuple               # per layer (C_out, C_in)
    pw_b: tuple               # per layer (C_out,)
    proj1_w: np.ndarray       # (P, C)
    proj1_b: np.ndarray       # (P,)
    proj2_w: np.ndarray       # (P,)
    proj2_b: float

    @property
    def kxy(self):
        return 2 * self.modes - 1

    @property
    def kz(self):
        return self.modes


def next_pow2(n):
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def padded_dims(leaves_dim):
    """Coarse grid shape (x, y, z) for a sparse grid's ``leaves_dim``."""
    return tuple(next_pow2(int(v)) for v in leaves_dim)


def dft_basis(n, modes, half=False):
    """Forward DFT rows ``(cos, sin)`` of shape (K, n) for one axis.

    ``half=False``: signed frequencies ``-(modes-1) .. modes-1``;
    ``half=True``: ``0 .. modes-1``. Rows a length-``n`` axis cannot resolve
    without aliasing are zero. Also returns the integer frequencies.
    """
    if half:
        freqs = np.arange(modes)
    else:
        freqs = np.arange(2 * modes - 1) - (modes - 1)
    x = np.arange(n)
    theta = 2.0 * np.pi * np.outer(freqs, x) / float(n)
    keep = (np.abs(freqs) <= (n - 1) // 2)[:, None]
    cos = np.where(keep, np.cos(theta), 0.0)
    sin = np.where(keep, np.sin(theta), 0.0)
    return cos, sin, freqs


def half_weights(modes):
    """Inverse weight of each kept z frequency: 1 for k=0, 2 otherwise."""
    w = np.full(modes, 2.0)
    w[0] = 1.0
    return w


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x * x * x)))


# -- weights I/O ---------------------------------------------------------------

def random_weights(*, width, modes, layers, proj_hidden, seed=0, scale=0.5):
    """A deterministic, untrained weights dict (test fixtures, tooling smoke)."""
    rng = np.random.default_rng(seed)
    kxy, kz = 2 * modes - 1, modes

    def rnd(*shape, fan_in):
        return (rng.standard_normal(shape) * scale / math.sqrt(fan_in)).astype(np.float32)

    w = {
        "schema": np.array(SCHEMA),
        "width": np.array(width, np.int32),
        "modes": np.array(modes, np.int32),
        "layers": np.array(layers, np.int32),
        "proj_hidden": np.array(proj_hidden, np.int32),
        "lift_w": rnd(width, IN_CHANNELS, fan_in=IN_CHANNELS),
        "lift_b": rnd(width, fan_in=IN_CHANNELS),
        "proj1_w": rnd(proj_hidden, width, fan_in=width),
        "proj1_b": rnd(proj_hidden, fan_in=width),
        "proj2_w": rnd(proj_hidden, fan_in=proj_hidden),
        "proj2_b": rnd(1, fan_in=proj_hidden),
    }
    for layer in range(layers):
        spec_scale = width * kz * kxy * kxy
        w[f"spec_re_{layer}"] = rnd(width, width, kz, kxy, kxy, fan_in=spec_scale)
        w[f"spec_im_{layer}"] = rnd(width, width, kz, kxy, kxy, fan_in=spec_scale)
        w[f"pw_w_{layer}"] = rnd(width, width, fan_in=width)
        w[f"pw_b_{layer}"] = rnd(width, fan_in=width)
    return w


def save_weights(path, weights):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **weights)


def load_weights(path):
    """Load and validate a weights file; raise :class:`WeightsError` with a reason."""
    if not path or not os.path.isfile(path):
        raise WeightsError("weights missing")
    if os.path.getsize(path) > MAX_WEIGHTS_BYTES:
        raise WeightsError("invalid weights (file larger than 5 MB)")
    try:
        with np.load(path, allow_pickle=False) as data:
            weights = {key: np.array(data[key]) for key in data.files}
    except Exception as exc:  # noqa: BLE001 - any parse failure is one reason
        raise WeightsError(f"weights unreadable ({type(exc).__name__})") from None
    return validate_weights(weights)


def _scalar_int(weights, key):
    try:
        return int(np.asarray(weights[key]).reshape(()))
    except Exception:  # noqa: BLE001
        raise WeightsError(f"invalid weights (bad '{key}')") from None


def validate_weights(weights):
    """Check schema, shapes and finiteness; return a float32 :class:`Model`."""
    try:
        schema = str(np.asarray(weights["schema"]).reshape(()))
    except Exception:  # noqa: BLE001
        raise WeightsError("weights schema mismatch (no schema)") from None
    if schema != SCHEMA:
        raise WeightsError(f"weights schema mismatch ({schema}, need {SCHEMA})")
    width = _scalar_int(weights, "width")
    modes = _scalar_int(weights, "modes")
    layers = _scalar_int(weights, "layers")
    hidden = _scalar_int(weights, "proj_hidden")
    if min(width, modes, layers, hidden) < 1:
        raise WeightsError("invalid weights (non-positive size)")
    kxy, kz = 2 * modes - 1, modes
    expected = {
        "lift_w": (width, IN_CHANNELS), "lift_b": (width,),
        "proj1_w": (hidden, width), "proj1_b": (hidden,),
        "proj2_w": (hidden,), "proj2_b": (1,),
    }
    for layer in range(layers):
        expected[f"spec_re_{layer}"] = (width, width, kz, kxy, kxy)
        expected[f"spec_im_{layer}"] = (width, width, kz, kxy, kxy)
        expected[f"pw_w_{layer}"] = (width, width)
        expected[f"pw_b_{layer}"] = (width,)
    arrays = {}
    for key, shape in expected.items():
        if key not in weights:
            raise WeightsError(f"invalid weights (missing '{key}')")
        value = np.asarray(weights[key])
        if tuple(value.shape) != shape:
            raise WeightsError(
                f"invalid weights ('{key}' is {tuple(value.shape)}, need {shape})")
        if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
            raise WeightsError(f"invalid weights ('{key}' is not finite floats)")
        arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
    return Model(
        width=width, modes=modes, layers=layers, proj_hidden=hidden,
        lift_w=arrays["lift_w"], lift_b=arrays["lift_b"],
        spec_re=tuple(arrays[f"spec_re_{i}"] for i in range(layers)),
        spec_im=tuple(arrays[f"spec_im_{i}"] for i in range(layers)),
        pw_w=tuple(arrays[f"pw_w_{i}"] for i in range(layers)),
        pw_b=tuple(arrays[f"pw_b_{i}"] for i in range(layers)),
        proj1_w=arrays["proj1_w"], proj1_b=arrays["proj1_b"],
        proj2_w=arrays["proj2_w"], proj2_b=float(arrays["proj2_b"][0]),
    )


# -- features ----------------------------------------------------------------

def input_scale(features):
    """Per-sample scale ``s``: RMS of the right-hand-side channel (0 if none)."""
    rhs = np.asarray(features[0], dtype=np.float64)
    return float(math.sqrt(float(np.mean(rhs * rhs))))


def normalise(features):
    """Return (network input, output scale) for raw coarse features."""
    s = input_scale(features)
    out = np.array(features, dtype=np.float32, copy=True)
    inv = 1.0 / s if s > 0.0 else 1.0
    out[0] *= inv
    out[3] *= inv
    return out, s


# -- reference forward ---------------------------------------------------------

def spectral_numpy(h, re, im, modes):
    """Truncated spectral convolution of h (C, Z, Y, X) with complex weights."""
    _, nz, ny, nx = h.shape
    cx, sx, _ = dft_basis(nx, modes)
    cy, sy, _ = dft_basis(ny, modes)
    cz, sz, _ = dft_basis(nz, modes, half=True)
    ex, ey, ez = cx - 1j * sx, cy - 1j * sy, cz - 1j * sz
    f = np.einsum("czyx,az,by,dx->cabd", h.astype(np.float64), ez, ey, ex)
    w = re.astype(np.float64) + 1j * im.astype(np.float64)
    g = np.einsum("cabd,coabd->oabd", f, w)
    g = g * half_weights(modes)[None, :, None, None]
    out = np.einsum("oabd,az,by,dx->ozyx", g, np.conj(ez), np.conj(ey), np.conj(ex))
    return out.real / float(nx * ny * nz)


def forward_numpy(model, features):
    """Reference forward on *normalised* features (IN, Z, Y, X) -> (Z, Y, X)."""
    h = np.einsum("ci,izyx->czyx", model.lift_w.astype(np.float64),
                  features.astype(np.float64)) + model.lift_b[:, None, None, None]
    for layer in range(model.layers):
        spec = spectral_numpy(h, model.spec_re[layer], model.spec_im[layer], model.modes)
        h = (spec + np.einsum("oc,czyx->ozyx", model.pw_w[layer].astype(np.float64), h)
             + model.pw_b[layer][:, None, None, None])
        if layer + 1 < model.layers:
            h = gelu(h)
    t = gelu(np.einsum("pc,czyx->pzyx", model.proj1_w.astype(np.float64), h)
             + model.proj1_b[:, None, None, None])
    q = np.einsum("p,pzyx->zyx", model.proj2_w.astype(np.float64), t) + model.proj2_b
    return q.astype(np.float32)


def predict_numpy(model, raw_features):
    """Raw features -> coarse pressure guess (applies the scale both ways)."""
    feats, s = normalise(raw_features)
    if s <= 0.0:
        return np.zeros(raw_features.shape[1:], np.float32)
    return (forward_numpy(model, feats) * np.float32(s)).astype(np.float32)
