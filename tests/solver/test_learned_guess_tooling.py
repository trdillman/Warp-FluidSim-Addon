"""tools/learned_guess: scenes, dataset export, training model, export, evaluate.

The training script needs PyTorch, which is not in Blender's Python; its
parity test skips without torch. Everything else (scene generator, recorder,
weights export, gate arithmetic) runs everywhere the suite runs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fluidSim_warp_blender51.sim import learned_guess as lg  # noqa: E402
from fluidSim_warp_blender51.sim import learned_guess_model as lgm  # noqa: E402
from tools.learned_guess import evaluate, export_dataset, export_weights, scenes, train  # noqa: E402

SMALL = scenes.SceneSpec(
    seed=-1, domain_max=(0.5, 0.5, 0.5),
    blocks=(scenes.Block(lo=(0.0, 0.0, 0.0), hi=(0.25, 0.5, 0.375)),))
CONFIG = {"width": 4, "modes": 2, "layers": 2, "proj_hidden": 8}


# -- scenes ---------------------------------------------------------------------

def test_scenes_are_deterministic_and_the_heldout_set_is_disjoint():
    seeds = list(scenes.TRAIN_SEEDS)[:20] + list(scenes.HELDOUT_SEEDS)[:5]
    assert scenes.scenes(seeds) == scenes.scenes(seeds)
    assert not set(scenes.TRAIN_SEEDS) & set(scenes.HELDOUT_SEEDS)
    for spec in scenes.scenes(seeds):
        assert spec.blocks
        for block in spec.blocks:
            assert all(0.0 <= lo < hi <= d + 1e-9
                       for lo, hi, d in zip(block.lo, block.hi, spec.domain_max))
        # the held-out shape (two separated blocks) never appears in training
        assert (len(spec.blocks) == 2 and spec.blocks[0].velocity == (0.0, 0.0, 0.0)
                and spec.blocks[1].lo[0] > spec.blocks[0].hi[0]) == spec.heldout


def test_block_particles_fill_the_block_on_a_lattice():
    pts = scenes.block_particles(SMALL.blocks[0])
    assert pts.shape == (8 * 16 * 12 * 8, 3)
    assert pts.min() > 0.0 and np.all(pts.max(axis=0) < (0.25, 0.5, 0.375))


# -- export_dataset ---------------------------------------------------------------

def test_recorder_records_features_and_targets_without_changing_the_solve(cuda_device):
    def run(record):
        solver = scenes.build_solver(SMALL, cuda_device)
        recorder = None
        try:
            if record:
                recorder = export_dataset.FeatureRecorder(solver.grid, cuda_device)
                solver.pressure_solver.learned_guess = recorder
            for frame in range(1, 4):
                if recorder is not None:
                    recorder.frame = frame
                solver.step_frame()
            return solver.particle_positions().copy(), recorder
        finally:
            solver.pressure_solver.learned_guess = None
            solver.release()

    plain, _ = run(False)
    recorded, recorder = run(True)
    assert np.array_equal(plain, recorded)          # zero guess, bit for bit
    assert len(recorder.rows) >= 3
    nx, ny, nz = recorder.dims
    for i, (feats, target, iterations, frame) in enumerate(recorder.rows):
        assert feats.shape == (4, nz, ny, nx) and target.shape == (nz, ny, nx)
        assert iterations >= 0 and 1 <= frame <= 3
        assert np.all((feats[1] >= 0.0) & (feats[1] <= 1.0))
        assert np.all(target[feats[1] == 0.0] == 0.0)
        # the previous-pressure channel is the previous solve's target
        if i:
            assert np.array_equal(feats[3], recorder.rows[i - 1][1])
    assert np.any(recorder.rows[-1][1] != 0.0)


def test_heldout_seeds_are_refused_by_the_exporter(tmp_path):
    with pytest.raises(SystemExit):
        export_dataset.main(["--out", str(tmp_path),
                             "--seeds", f"{scenes.HELDOUT_SEEDS.start}:"
                                        f"{scenes.HELDOUT_SEEDS.start + 1}"])


# -- export_weights -----------------------------------------------------------------

def test_exported_weights_load_in_the_addon_and_respect_the_size_limit(tmp_path):
    path = tmp_path / "w.npz"
    size = export_weights.export(CONFIG, train.init_state(CONFIG), str(path))
    assert size == path.stat().st_size < lgm.MAX_WEIGHTS_BYTES
    model = lgm.load_weights(str(path))
    assert (model.width, model.modes, model.layers) == (4, 2, 2)
    # the largest shipped-size config named in the spec notes still fits
    big = {"width": 24, "modes": 4, "layers": 4, "proj_hidden": 32}
    assert export_weights.export(big, train.init_state(big), str(tmp_path / "b.npz")) \
        < lgm.MAX_WEIGHTS_BYTES
    huge = {"width": 48, "modes": 6, "layers": 4, "proj_hidden": 32}
    with pytest.raises(lgm.WeightsError, match="limit"):
        export_weights.export(huge, train.init_state(huge), str(tmp_path / "h.npz"))
    assert not (tmp_path / "h.npz").exists()


def test_export_refuses_a_state_with_a_wrong_shape(tmp_path):
    state = train.init_state(CONFIG)
    state["pw_w_1"] = state["pw_w_1"][:, :2]
    with pytest.raises(lgm.WeightsError, match="pw_w_1"):
        export_weights.export(CONFIG, state, str(tmp_path / "w.npz"))


# -- train (torch) ------------------------------------------------------------------

def test_torch_model_matches_the_numpy_reference_and_learns():
    torch = pytest.importorskip("torch")
    state = train.init_state(CONFIG, seed=3)
    model = lgm.validate_weights(export_weights.weights_from_state(CONFIG, state))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 4, 8, 8)).astype(np.float32)
    params = {k: torch.tensor(v) for k, v in state.items()}
    got = train.model_forward(params, CONFIG, torch.tensor(x), train.Bases(2)).numpy()
    for b in range(2):
        np.testing.assert_allclose(got[b], lgm.forward_numpy(model, x[b]),
                                   rtol=1e-4, atol=1e-5)
    # one shape group, a learnable smooth target: the loss must drop
    groups = {(4, 8, 8): [(xi, np.tanh(xi[0]), np.ones((4, 8, 8), np.float32))
                          for xi in rng.standard_normal((8, 4, 4, 8, 8)).astype(np.float32)]}
    losses = []
    train.train(groups, CONFIG, epochs=6, lr=1e-2, batch=4,
                log=lambda line: losses.append(float(line.rsplit(" ", 1)[1])))
    assert losses[-1] < losses[0]


# -- evaluate ---------------------------------------------------------------------

def test_gate_arithmetic():
    rows = [{"frame": f, "iterations": it} for f, it in ((1, 50), (2, 6), (3, 8), (49, 99))]
    assert evaluate.median_iterations(rows) == 7.0
    assert evaluate.ratio(10.0, 5.0) == 2.0 and evaluate.ratio(0.0, 0.0) == 1.0
    assert evaluate.abba([2.0, 2.2], [1.0, 1.0]) == pytest.approx(2.1)
    met = evaluate.gate_verdict([1.6, 1.5], [2.0, 2.5], 1.1, True)
    assert met["met"]
    assert not evaluate.gate_verdict([1.6], [2.0, 1.9], 1.1, True)["met"]
    assert not evaluate.gate_verdict([1.6], [], 1.1, True)["met"]      # demos unmeasured
    assert not evaluate.gate_verdict([1.6], [3.0], 0.99, True)["met"]
    assert not evaluate.gate_verdict([1.6], [3.0], 1.2, False)["met"]


def test_evaluate_records_converged_solves_with_fixture_weights(cuda_device, tmp_path,
                                                                 monkeypatch):
    path = tmp_path / "w.npz"
    export_weights.export(CONFIG, train.init_state(CONFIG, seed=5), str(path))
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", str(path))
    off, off_msg = evaluate.record_solves(SMALL, cuda_device, 3, guess=False, exact=True)
    on, on_msg = evaluate.record_solves(SMALL, cuda_device, 3, guess=True, exact=True)
    assert off_msg == "" and on_msg == "on"
    summary = evaluate.summarise_scene(off, on)
    assert summary["all_on_converged"] and summary["solves"] == len(off) >= 3
