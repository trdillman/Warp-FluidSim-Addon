"""Train the learned pressure guess (PyTorch; developer machine only).

Torch never ships in the extension: this script trains the exact network
defined in ``fluidSim_warp_blender51/sim/learned_guess_model.py`` (same
features, scaling, DFT bases and tanh-GELU), so the exported weights run
unchanged in the Warp inference. ``model_forward`` is checked against the
numpy reference in the tests.

Loss: mean squared error of the normalised prediction against the
normalised target (leaf-mean pressure / s) over coarse cells that hold
fluid. Samples with s = 0 (no right-hand side) are skipped; the shipped
inference returns a zero guess for them anyway.

Usage::

    python tools/learned_guess/train.py --data DIR --out CHECKPOINT.pt
        [--width 16 --modes 4 --layers 3 --hidden 32 --epochs 30 --seed 0]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fluidSim_warp_blender51.sim import learned_guess_model as lgm  # noqa: E402

_HEADER = ("schema", "width", "modes", "layers", "proj_hidden")


def _torch():
    import torch

    return torch


def init_state(config, seed=0):
    """Initial parameters (numpy, weights-file keys) from ``lgm.random_weights``."""
    w = lgm.random_weights(width=config["width"], modes=config["modes"],
                           layers=config["layers"], proj_hidden=config["proj_hidden"],
                           seed=seed, scale=1.0)
    return {k: v for k, v in w.items() if k not in _HEADER}


class Bases:
    """Complex DFT bases per padded shape, built from ``lgm.dft_basis``."""

    def __init__(self, modes):
        self.modes = modes
        self.cache = {}

    def get(self, shape):
        torch = _torch()
        if shape not in self.cache:
            nz, ny, nx = shape
            out = []
            for n, half in ((nz, True), (ny, False), (nx, False)):
                cos, sin, _ = lgm.dft_basis(n, self.modes, half=half)
                out.append(torch.tensor(cos - 1j * sin, dtype=torch.complex64))
            hw = torch.tensor(lgm.half_weights(self.modes), dtype=torch.float32)
            self.cache[shape] = (*out, hw)
        return self.cache[shape]


def gelu(x):
    torch = _torch()
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def model_forward(params, config, x, bases):
    """Batched forward on normalised features x (B, 4, Z, Y, X) -> (B, Z, Y, X)."""
    torch = _torch()
    shape = tuple(int(v) for v in x.shape[2:])
    ez, ey, ex, hw = bases.get(shape)
    n = float(shape[0] * shape[1] * shape[2])
    h = (torch.einsum("ci,bizyx->bczyx", params["lift_w"], x)
         + params["lift_b"][None, :, None, None, None])
    for layer in range(config["layers"]):
        f = torch.einsum("bczyx,kz,ly,mx->bcklm", h.to(torch.complex64), ez, ey, ex)
        w = torch.complex(params[f"spec_re_{layer}"], params[f"spec_im_{layer}"])
        g = torch.einsum("bcklm,coklm->boklm", f, w) * hw[None, None, :, None, None]
        spec = torch.einsum("boklm,kz,ly,mx->bozyx", g,
                            ez.conj(), ey.conj(), ex.conj()).real / n
        h = (spec + torch.einsum("oc,bczyx->bozyx", params[f"pw_w_{layer}"], h)
             + params[f"pw_b_{layer}"][None, :, None, None, None])
        if layer + 1 < config["layers"]:
            h = gelu(h)
    t = gelu(torch.einsum("pc,bczyx->bpzyx", params["proj1_w"], h)
             + params["proj1_b"][None, :, None, None, None])
    return torch.einsum("p,bpzyx->bzyx", params["proj2_w"], t) + params["proj2_b"][0]


def load_samples(data_dir):
    """Normalised (x, y, mask) samples grouped by padded shape."""
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "scene_*.npz"))):
        with np.load(path) as data:
            for feats, target in zip(data["features"], data["target"]):
                x, s = lgm.normalise(feats)
                if s <= 0.0:
                    continue
                mask = (feats[1] > 0.0).astype(np.float32)
                groups[tuple(x.shape[1:])].append(
                    (x, (target / s).astype(np.float32), mask))
    return groups


def train(groups, config, *, epochs=30, lr=2e-3, batch=16, seed=0, log=print):
    """Adam + cosine schedule; returns the trained parameters as numpy."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = {k: torch.nn.Parameter(torch.tensor(v))
              for k, v in init_state(config, seed).items()}
    opt = torch.optim.Adam(params.values(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    bases = Bases(config["modes"])
    batches = [(shape, i) for shape, samples in groups.items()
               for i in range(0, len(samples), batch)]
    for epoch in range(epochs):
        total, count = 0.0, 0
        for index in rng.permutation(len(batches)):
            shape, i = batches[index]
            chunk = groups[shape][i:i + batch]
            x = torch.tensor(np.stack([c[0] for c in chunk]))
            y = torch.tensor(np.stack([c[1] for c in chunk]))
            m = torch.tensor(np.stack([c[2] for c in chunk]))
            pred = model_forward(params, config, x, bases)
            loss = ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(chunk)
            count += len(chunk)
        sched.step()
        log(f"epoch {epoch + 1}/{epochs} loss {total / max(1, count):.6g}")
    return {k: v.detach().numpy().copy() for k, v in params.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch = _torch()
    config = {"width": args.width, "modes": args.modes, "layers": args.layers,
              "proj_hidden": args.hidden}
    groups = load_samples(args.data)
    print(f"{sum(len(v) for v in groups.values())} samples in {len(groups)} shapes",
          flush=True)
    state = train(groups, config, epochs=args.epochs, lr=args.lr, batch=args.batch,
                  seed=args.seed, log=lambda line: print(line, flush=True))
    torch.save({"config": config, "state": state}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
