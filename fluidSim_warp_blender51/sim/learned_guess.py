"""Learned pressure initial guess: Warp-native inference (4.0 lane N).

Gated and off by default (``pressure_learned_guess``; spec
``docs/specs/4.0/SPEC-learned-pressure-guess.md``). When it is on and a valid
``learned_guess_v1`` weights file is present, each PCG pressure solve starts
from a coarse guess (one value per sparse-grid leaf) predicted by the small
Fourier neural operator defined in ``learned_guess_model.py``, instead of
from zero. The PCG still runs to the same tolerance, so the guess can only
change how many iterations a solve takes, never what it converges to.

Any problem (no weights file, a schema mismatch, an unreadable or invalid
file, ``FORCE_OFF``) is decided ONCE, at construction: the guess stays
inactive, the solve runs the untouched zero-guess path and :attr:`reason`
says why. Nothing is logged or warned per frame.

Inference contract:

* fixed shapes: every buffer is sized at construction from the grid's leaf
  dimensions (padded to powers of two) and the weights, so a step allocates
  nothing and never reads back to the host;
* deterministic: every kernel is a gather with a fixed summation order, no
  atomics, so a bake stays bitwise repeatable;
* graph-capturable: on CUDA the whole inference (and the post-solve
  restriction that feeds the next guess) is recorded once per buffer set as a
  CUDA graph and replayed.

No bpy imports allowed here.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np
import warp as wp

from . import learned_guess_model as lgm
from .constants import LEAF_VOLUME, VOXEL_COLLISION, VOXEL_FLUID

WEIGHTS_FILENAME = "learned_guess_v1.npz"
#: Shipped weights location. No trained weights ship in 4.0 (the ship gate
#: was not met, see docs/RELEASE_4_0_STATUS.md), so this file is absent and
#: the guess reports "weights missing" if someone turns the hidden setting on.
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "learned_guess_weights", WEIGHTS_FILENAME)
#: Test/diagnostic switch: keep the guess inactive even with valid weights.
FORCE_OFF = False
JACOBI_REASON = "needs the PCG pressure solver"
_MAX_GRAPHS = 4

_GELU_C = 0.7978845608028654


@wp.func
def _gelu(x: float):
    return 0.5 * x * (1.0 + wp.tanh(_GELU_C * (x + 0.044715 * x * x * x)))


@wp.func
def _is_unknown(vt: wp.int32):
    return (vt & VOXEL_FLUID) != 0 and (vt & VOXEL_COLLISION) == 0


@wp.func
def _cell_slot(cell: int, pdim: wp.vec3i, leaves_dim: wp.vec3i,
               leaf_table: wp.array(dtype=wp.int32), num_leaves: int):
    """Leaf slot of a coarse cell; -1 when empty, -2 when outside the domain."""
    cx = cell % pdim[0]
    cy = (cell // pdim[0]) % pdim[1]
    cz = cell // (pdim[0] * pdim[1])
    if cx >= leaves_dim[0] or cy >= leaves_dim[1] or cz >= leaves_dim[2]:
        return -2
    slot = leaf_table[cx + leaves_dim[0] * (cy + leaves_dim[1] * cz)]
    if slot < 0 or slot >= num_leaves:
        return -1
    return slot


@wp.kernel
def k_lg_features(
    rhs: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    pdim: wp.vec3i,
    ncell: int,
    prev_q: wp.array(dtype=float),
    feats: wp.array(dtype=float),
):
    """One thread per coarse cell; the leaf's 512 voxels in fixed order."""
    cell = wp.tid()
    slot = _cell_slot(cell, pdim, leaves_dim, leaf_table, num_leaves[0])
    total = wp.float32(0.0)
    fluid = wp.float32(0.0)
    solid = wp.float32(0.0)
    prev = wp.float32(0.0)
    if slot == -2:
        solid = 1.0          # padding outside the domain reads as solid wall
    elif slot >= 0:
        base = slot * LEAF_VOLUME
        nf = wp.int32(0)
        ns = wp.int32(0)
        for local in range(LEAF_VOLUME):
            vt = voxel_type[base + local]
            if _is_unknown(vt):
                total += rhs[base + local]
                nf += 1
            if (vt & VOXEL_COLLISION) != 0:
                ns += 1
        fluid = float(nf) / float(LEAF_VOLUME)
        solid = float(ns) / float(LEAF_VOLUME)
        prev = prev_q[cell]
    feats[cell] = total
    feats[ncell + cell] = fluid
    feats[2 * ncell + cell] = solid
    feats[3 * ncell + cell] = prev


@wp.kernel
def k_lg_scale(feats: wp.array(dtype=float), ncell: int, scale: wp.array(dtype=float)):
    """Single thread, fixed order: scale = [1/s (or 1), s] with s = RMS(rhs)."""
    acc = wp.float32(0.0)
    for cell in range(ncell):
        v = feats[cell]
        acc += v * v
    s = wp.sqrt(acc / float(ncell))
    if s > 0.0:
        scale[0] = 1.0 / s
    else:
        scale[0] = 1.0
    scale[1] = s


@wp.kernel
def k_lg_lift(feats: wp.array(dtype=float), scale: wp.array(dtype=float),
              w: wp.array(dtype=float), b: wp.array(dtype=float),
              in_ch: int, ncell: int, h: wp.array(dtype=float)):
    tid = wp.tid()
    c = tid // ncell
    cell = tid - c * ncell
    acc = b[c]
    for i in range(in_ch):
        f = feats[i * ncell + cell]
        if i == 0 or i == 3:
            f = f * scale[0]
        acc += w[c * in_ch + i] * f
    h[tid] = acc


@wp.kernel
def k_lg_axis(in_re: wp.array(dtype=float), in_im: wp.array(dtype=float), use_im: int,
              cos_t: wp.array(dtype=float), sin_t: wp.array(dtype=float), sgn: float,
              n_sum: int, n_out: int, n_in: int,
              out_re: wp.array(dtype=float), out_im: wp.array(dtype=float)):
    """out[o,k,i] = sum_j in[o,j,i] * (cos[k,j] + sgn*i*sin[k,j]), fixed j order."""
    tid = wp.tid()
    o = tid // (n_out * n_in)
    rem = tid - o * n_out * n_in
    k = rem // n_in
    i = rem - k * n_in
    acc_re = wp.float32(0.0)
    acc_im = wp.float32(0.0)
    for j in range(n_sum):
        src = (o * n_sum + j) * n_in + i
        a = in_re[src]
        bb = wp.float32(0.0)
        if use_im != 0:
            bb = in_im[src]
        c = cos_t[k * n_sum + j]
        s = sgn * sin_t[k * n_sum + j]
        acc_re += a * c - bb * s
        acc_im += a * s + bb * c
    out_re[tid] = acc_re
    out_im[tid] = acc_im


@wp.kernel
def k_lg_mix(f_re: wp.array(dtype=float), f_im: wp.array(dtype=float),
             w_re: wp.array(dtype=float), w_im: wp.array(dtype=float),
             width: int, nk: int,
             g_re: wp.array(dtype=float), g_im: wp.array(dtype=float)):
    """G[o,k] = sum_c F[c,k] * W[c,o,k] (complex), fixed c order."""
    tid = wp.tid()
    o = tid // nk
    k = tid - o * nk
    acc_re = wp.float32(0.0)
    acc_im = wp.float32(0.0)
    for c in range(width):
        a = f_re[c * nk + k]
        bb = f_im[c * nk + k]
        widx = (c * width + o) * nk + k
        wr = w_re[widx]
        wi = w_im[widx]
        acc_re += a * wr - bb * wi
        acc_im += a * wi + bb * wr
    g_re[tid] = acc_re
    g_im[tid] = acc_im


@wp.kernel
def k_lg_pointwise(h: wp.array(dtype=float), spec: wp.array(dtype=float),
                   w: wp.array(dtype=float), b: wp.array(dtype=float),
                   width: int, ncell: int, activate: int, out: wp.array(dtype=float)):
    tid = wp.tid()
    o = tid // ncell
    cell = tid - o * ncell
    acc = spec[tid] + b[o]
    for c in range(width):
        acc += w[o * width + c] * h[c * ncell + cell]
    if activate != 0:
        acc = _gelu(acc)
    out[tid] = acc


@wp.kernel
def k_lg_project(h: wp.array(dtype=float), w1: wp.array(dtype=float), b1: wp.array(dtype=float),
                 w2: wp.array(dtype=float), b2: float, width: int, hidden: int, ncell: int,
                 scale: wp.array(dtype=float), q: wp.array(dtype=float)):
    cell = wp.tid()
    acc = b2
    for p in range(hidden):
        t = b1[p]
        for c in range(width):
            t += w1[p * width + c] * h[c * ncell + cell]
        acc += w2[p] * _gelu(t)
    q[cell] = acc * scale[1]


@wp.kernel
def k_lg_inject(q: wp.array(dtype=float), voxel_type: wp.array(dtype=wp.int32),
                leaf_coords: wp.array(dtype=wp.vec3i), num_leaves: wp.array(dtype=wp.int32),
                pdim: wp.vec3i, x0: wp.array(dtype=float)):
    """Piecewise-constant prolongation of the coarse guess onto the unknowns."""
    tid = wp.tid()
    slot = tid // LEAF_VOLUME
    if slot >= num_leaves[0]:
        x0[tid] = 0.0
        return
    if not _is_unknown(voxel_type[tid]):
        x0[tid] = 0.0
        return
    lc = leaf_coords[slot]
    x0[tid] = q[lc[0] + pdim[0] * (lc[1] + pdim[1] * lc[2])]


@wp.kernel
def k_lg_restrict(pressure: wp.array(dtype=float), voxel_type: wp.array(dtype=wp.int32),
                  leaf_table: wp.array(dtype=wp.int32), leaves_dim: wp.vec3i,
                  num_leaves: wp.array(dtype=wp.int32), pdim: wp.vec3i,
                  prev_q: wp.array(dtype=float)):
    """Leaf mean of the solved pressure over the unknowns (next guess's input)."""
    cell = wp.tid()
    slot = _cell_slot(cell, pdim, leaves_dim, leaf_table, num_leaves[0])
    if slot < 0:
        prev_q[cell] = 0.0
        return
    base = slot * LEAF_VOLUME
    total = wp.float32(0.0)
    n = wp.int32(0)
    for local in range(LEAF_VOLUME):
        if _is_unknown(voxel_type[base + local]):
            total += pressure[base + local]
            n += 1
    if n > 0:
        prev_q[cell] = total / float(n)
    else:
        prev_q[cell] = 0.0


def _arr(values, device):
    return wp.array(np.ascontiguousarray(values, dtype=np.float32).ravel(),
                    dtype=float, device=device)


class _Network:
    """Fixed-shape device buffers + launches for one model on one coarse grid."""

    def __init__(self, model, pdim, device):
        self.model = model
        self.device = device
        px, py, pz = (int(v) for v in pdim)
        self.pdim = (px, py, pz)
        self.ncell = px * py * pz
        c, m = model.width, model.modes
        kxy, kz = model.kxy, model.kz
        self.kx, self.ky, self.kz = kxy, kxy, kz
        n = self.ncell
        cx, sx, _ = lgm.dft_basis(px, m)
        cy, sy, _ = lgm.dft_basis(py, m)
        cz, sz, _ = lgm.dft_basis(pz, m, half=True)
        wz = lgm.half_weights(m)
        self.fwd = [(_arr(cx, device), _arr(sx, device)),
                    (_arr(cy, device), _arr(sy, device)),
                    (_arr(cz, device), _arr(sz, device))]
        self.inv = [(_arr(cz.T * wz[None, :], device), _arr(sz.T * wz[None, :], device)),
                    (_arr(cy.T, device), _arr(sy.T, device)),
                    (_arr(cx.T / float(n), device), _arr(sx.T / float(n), device))]
        self.lift_w = _arr(model.lift_w, device)
        self.lift_b = _arr(model.lift_b, device)
        self.spec = [(_arr(model.spec_re[i], device), _arr(model.spec_im[i], device))
                     for i in range(model.layers)]
        self.pw = [(_arr(model.pw_w[i], device), _arr(model.pw_b[i], device))
                   for i in range(model.layers)]
        self.proj1_w = _arr(model.proj1_w, device)
        self.proj1_b = _arr(model.proj1_b, device)
        self.proj2_w = _arr(model.proj2_w, device)
        zeros = lambda size: wp.zeros(max(1, size), dtype=float, device=device)  # noqa: E731
        self.h = zeros(c * n)
        self.h2 = zeros(c * n)
        self.spec_out = zeros(c * n)
        self.ta = (zeros(c * pz * py * kxy), zeros(c * pz * py * kxy))
        self.tb = (zeros(c * pz * kxy * kxy), zeros(c * pz * kxy * kxy))
        self.tf = (zeros(c * kz * kxy * kxy), zeros(c * kz * kxy * kxy))
        self.tg = (zeros(c * kz * kxy * kxy), zeros(c * kz * kxy * kxy))
        self.q = zeros(n)

    def _axis(self, src, use_im, table, sgn, outer, n_sum, n_out, n_in, dst):
        wp.launch(k_lg_axis, dim=outer * n_out * n_in,
                  inputs=[src[0], src[1], use_im, table[0], table[1], sgn, n_sum, n_out, n_in,
                          dst[0], dst[1]], device=self.device)

    def run(self, feats, scale):
        """feats (IN*ncell), scale [1/s, s] (device) -> self.q (ncell)."""
        model, dev, n = self.model, self.device, self.ncell
        c = model.width
        px, py, pz = self.pdim
        kx, ky, kz = self.kx, self.ky, self.kz
        wp.launch(k_lg_lift, dim=c * n,
                  inputs=[feats, scale, self.lift_w, self.lift_b, lgm.IN_CHANNELS, n, self.h],
                  device=dev)
        h, h2 = self.h, self.h2
        for layer in range(model.layers):
            real = (h, h)
            self._axis(real, 0, self.fwd[0], -1.0, c * pz * py, px, kx, 1, self.ta)
            self._axis(self.ta, 1, self.fwd[1], -1.0, c * pz, py, ky, kx, self.tb)
            self._axis(self.tb, 1, self.fwd[2], -1.0, c, pz, kz, ky * kx, self.tf)
            nk = kz * ky * kx
            wp.launch(k_lg_mix, dim=c * nk,
                      inputs=[self.tf[0], self.tf[1], self.spec[layer][0], self.spec[layer][1],
                              c, nk, self.tg[0], self.tg[1]], device=dev)
            self._axis(self.tg, 1, self.inv[0], 1.0, c, kz, pz, ky * kx, self.tb)
            self._axis(self.tb, 1, self.inv[1], 1.0, c * pz, ky, py, kx, self.ta)
            # The imaginary half of the last inverse pass is discarded: it is
            # written into h2, which the pointwise step overwrites next.
            self._axis(self.ta, 1, self.inv[2], 1.0, c * pz * py, kx, px, 1,
                       (self.spec_out, h2))
            activate = 1 if layer + 1 < model.layers else 0
            wp.launch(k_lg_pointwise, dim=c * n,
                      inputs=[h, self.spec_out, self.pw[layer][0], self.pw[layer][1],
                              c, n, activate, h2], device=dev)
            h, h2 = h2, h
        wp.launch(k_lg_project, dim=n,
                  inputs=[h, self.proj1_w, self.proj1_b, self.proj2_w, float(model.proj2_b),
                          c, model.proj_hidden, n, scale, self.q], device=dev)
        return self.q


def forward_warp(model, features, device=None):
    """Run the network on *normalised* features (IN, Z, Y, X); test helper."""
    feats = np.asarray(features, dtype=np.float32)
    nz, ny, nx = feats.shape[1:]
    net = _Network(model, (nx, ny, nz), device)
    scale = wp.array(np.array([1.0, 1.0], np.float32), dtype=float, device=device)
    q = net.run(_arr(feats, device), scale)
    return q.numpy()[: nx * ny * nz].reshape(nz, ny, nx)


class LearnedGuess:
    """Per-solver learned guess. Inactive (with a reason) unless it can run."""

    def __init__(self, grid, device, weights_path=None):
        self.grid = grid
        self.device = device
        self.weights_path = DEFAULT_WEIGHTS_PATH if weights_path is None else weights_path
        self.model = None
        self.reason = ""
        self.use_graph = True
        self.graph_launches = 0
        self._graphs = OrderedDict()
        self._net = None
        if FORCE_OFF:
            self.reason = "forced off"
            return
        try:
            self.model = lgm.load_weights(self.weights_path)
        except lgm.WeightsError as exc:
            self.reason = str(exc)
            return
        leaves = grid.leaves_dim
        self.leaves_dim = wp.vec3i(int(leaves[0]), int(leaves[1]), int(leaves[2]))
        pdim = lgm.padded_dims((int(leaves[0]), int(leaves[1]), int(leaves[2])))
        self.pdim = wp.vec3i(*pdim)
        self._net = _Network(self.model, pdim, device)
        n = self._net.ncell
        self.feats = wp.zeros(lgm.IN_CHANNELS * n, dtype=float, device=device)
        self.prev_q = wp.zeros(n, dtype=float, device=device)
        self.scale = wp.zeros(2, dtype=float, device=device)

    @property
    def active(self):
        return self._net is not None

    @property
    def message(self):
        return "on" if self.active else f"off ({self.reason})"

    def reset(self):
        """Forget the previous solve (the prev-pressure input goes to zero)."""
        if self.active:
            self.prev_q.zero_()

    def release(self):
        self._graphs.clear()

    # -- step ------------------------------------------------------------------

    def _infer(self, ps, voxel_type):
        g = self.grid
        n = self._net.ncell
        wp.launch(k_lg_features, dim=n,
                  inputs=[ps.rhs, voxel_type, g.leaf_table, self.leaves_dim, g.num_leaves,
                          self.pdim, n, self.prev_q, self.feats], device=self.device)
        wp.launch(k_lg_scale, dim=1, inputs=[self.feats, n, self.scale], device=self.device)
        q = self._net.run(self.feats, self.scale)
        wp.launch(k_lg_inject, dim=g.voxel_capacity,
                  inputs=[q, voxel_type, g.leaf_coords, g.num_leaves, self.pdim, ps.x0],
                  device=self.device)

    def _restrict(self, ps, voxel_type):
        g = self.grid
        wp.launch(k_lg_restrict, dim=self._net.ncell,
                  inputs=[ps.pressure, voxel_type, g.leaf_table, self.leaves_dim,
                          g.num_leaves, self.pdim, self.prev_q], device=self.device)

    def _run(self, tag, fn, ps, voxel_type, key_arrays):
        if not (self.use_graph and wp.get_device(self.device).is_cuda):
            fn(ps, voxel_type)
            return
        key = (tag,) + tuple(int(a.ptr) for a in key_arrays)
        graph = self._graphs.get(key)
        if graph is None:
            wp.load_module(sys.modules[__name__], device=self.device)
            with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                fn(ps, voxel_type)
            graph = capture.graph
            self._graphs[key] = graph
            while len(self._graphs) > _MAX_GRAPHS:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        wp.capture_launch(graph)
        self.graph_launches += 1

    def fill(self, ps, voxel_type):
        """Write the guess into ``ps.x0``; False (x0 untouched) when inactive."""
        if not self.active:
            return False
        self._run("infer", self._infer, ps, voxel_type,
                  (ps.rhs, voxel_type, ps.x0))
        return True

    def observe(self, ps, voxel_type):
        """After a solve: keep the leaf means of the pressure for the next guess."""
        if not self.active:
            return
        self._run("restrict", self._restrict, ps, voxel_type,
                  (ps.pressure, voxel_type))
