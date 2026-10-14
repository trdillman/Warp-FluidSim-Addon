"""Export a ``train.py`` checkpoint to a ``learned_guess_v1`` weights file.

The checkpoint holds the training config and every parameter as a numpy
array under its weights-file key (``lift_w``, ``spec_re_0``, ...), so the
mapping is a straight copy plus the scalar header. The result is validated
with the add-on's own ``validate_weights`` and refused if it is larger than
the 5 MB ship limit, then written with ``save_weights``.

Usage::

    python tools/learned_guess/export_weights.py CHECKPOINT.pt OUT.npz

Shipping a file means copying it to
``fluidSim_warp_blender51/sim/learned_guess_weights/learned_guess_v1.npz`` --
only after ``evaluate.py`` reports the ship gate as met.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fluidSim_warp_blender51.sim import learned_guess_model as lgm  # noqa: E402

CONFIG_KEYS = ("width", "modes", "layers", "proj_hidden")


def weights_from_state(config, state):
    """Weights dict (npz keys) from a config and numpy parameters; validated."""
    weights = {"schema": np.array(lgm.SCHEMA)}
    for key in CONFIG_KEYS:
        weights[key] = np.array(int(config[key]), np.int32)
    for key, value in state.items():
        weights[key] = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if "proj2_b" in weights:
        weights["proj2_b"] = weights["proj2_b"].reshape(1)
    lgm.validate_weights(weights)
    return weights


def compressed_size(weights):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **weights)
    return buffer.tell()


def export(config, state, out_path):
    """Validate, size-check and write; returns the file size in bytes."""
    weights = weights_from_state(config, state)
    size = compressed_size(weights)
    if size > lgm.MAX_WEIGHTS_BYTES:
        raise lgm.WeightsError(
            f"invalid weights ({size} bytes, over the {lgm.MAX_WEIGHTS_BYTES}-byte limit)")
    lgm.save_weights(out_path, weights)
    lgm.load_weights(out_path)          # round-trip through the shipped loader
    return os.path.getsize(out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key: np.asarray(value) for key, value in payload["state"].items()}
    size = export(payload["config"], state, args.out)
    print(f"wrote {args.out} ({size} bytes)")


if __name__ == "__main__":
    main()
