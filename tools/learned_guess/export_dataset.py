"""Export (features, target) pairs for the learned pressure guess.

For every pressure solve of every scene this records exactly what the
shipped inference sees -- the four coarse input channels built by
``learned_guess.k_lg_features`` -- and the training target: the leaf mean of
the converged pressure over the fluid unknowns (``k_lg_restrict``). The
solve itself is untouched: the recorder's ``fill`` returns False, so the
zero-guess path runs bit for bit, and the "previous pressure" channel is
the previous solve's target, exactly as at inference time.

Output, one compressed ``.npz`` per scene (datasets stay out of git)::

    features  (S, 4, Z, Y, X) float32   raw (un-normalised) features
    target    (S, Z, Y, X)    float32   leaf-mean solved pressure
    iterations (S,) int32               PCG iterations of the zero-guess solve
    frame     (S,) int32
    seed      ()   int32

Usage::

    python tools/learned_guess/export_dataset.py --out DIR [--seeds 0:200] [--frames 48]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.learned_guess import scenes  # noqa: E402


class FeatureRecorder:
    """Stands in for ``LearnedGuess`` on the solver; records, never guesses."""

    active = True
    reason = ""
    message = "recording"

    def __init__(self, grid, device):
        import warp as wp

        from fluidSim_warp_blender51.sim import learned_guess as lg
        from fluidSim_warp_blender51.sim import learned_guess_model as lgm

        self._wp, self._lg = wp, lg
        self.grid = grid
        self.device = device
        leaves = tuple(int(v) for v in grid.leaves_dim)
        self.leaves_dim = wp.vec3i(*leaves)
        self.dims = lgm.padded_dims(leaves)
        self.pdim = wp.vec3i(*self.dims)
        self.ncell = self.dims[0] * self.dims[1] * self.dims[2]
        self.feats = wp.zeros(lgm.IN_CHANNELS * self.ncell, dtype=float, device=device)
        self.prev_q = wp.zeros(self.ncell, dtype=float, device=device)
        self.frame = 0
        self.rows = []
        self._pending = None

    def reset(self):
        self.prev_q.zero_()

    def release(self):
        pass

    def _shape(self, flat, channels=None):
        nx, ny, nz = self.dims
        shape = (nz, ny, nx) if channels is None else (channels, nz, ny, nx)
        return flat.reshape(shape)

    def fill(self, ps, voxel_type):
        wp, lg, g = self._wp, self._lg, self.grid
        wp.launch(lg.k_lg_features, dim=self.ncell,
                  inputs=[ps.rhs, voxel_type, g.leaf_table, self.leaves_dim, g.num_leaves,
                          self.pdim, self.ncell, self.prev_q, self.feats], device=self.device)
        self._pending = self._shape(self.feats.numpy().copy(), 4)
        return False          # zero guess: the solve is the production one

    def observe(self, ps, voxel_type):
        wp, lg, g = self._wp, self._lg, self.grid
        wp.launch(lg.k_lg_restrict, dim=self.ncell,
                  inputs=[ps.pressure, voxel_type, g.leaf_table, self.leaves_dim,
                          g.num_leaves, self.pdim, self.prev_q], device=self.device)
        if self._pending is not None:
            self.rows.append((self._pending, self._shape(self.prev_q.numpy().copy()),
                              int(ps.last_iterations), self.frame))
        self._pending = None


def record_scene(spec, device, frames):
    """Run ``spec`` for ``frames`` frames; return the dataset arrays."""
    solver = scenes.build_solver(spec, device)
    try:
        recorder = FeatureRecorder(solver.grid, device)
        solver.pressure_solver.learned_guess = recorder
        for frame in range(1, frames + 1):
            recorder.frame = frame
            solver.step_frame()
        rows = recorder.rows
    finally:
        solver.pressure_solver.learned_guess = None
        solver.release()
    if not rows:
        raise RuntimeError(f"scene {spec.seed} made no pressure solves")
    return {
        "features": np.stack([r[0] for r in rows]).astype(np.float32),
        "target": np.stack([r[1] for r in rows]).astype(np.float32),
        "iterations": np.array([r[2] for r in rows], np.int32),
        "frame": np.array([r[3] for r in rows], np.int32),
        "seed": np.array(spec.seed, np.int32),
    }


def parse_seeds(text):
    lo, _, hi = text.partition(":")
    return range(int(lo), int(hi)) if hi else range(int(lo), int(lo) + 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="0:100", help="start:stop (train range by default)")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if any(seed in scenes.HELDOUT_SEEDS for seed in seeds):
        parser.error("held-out seeds are for evaluate.py only")
    import warp as wp

    wp.init()
    os.makedirs(args.out, exist_ok=True)
    for seed in seeds:
        path = os.path.join(args.out, f"scene_{seed:05d}.npz")
        if os.path.exists(path):
            continue
        data = record_scene(scenes.make_scene(seed), args.device, args.frames)
        np.savez_compressed(path, **data)
        print(f"seed {seed}: {len(data['iterations'])} solves -> {path}", flush=True)


if __name__ == "__main__":
    main()
