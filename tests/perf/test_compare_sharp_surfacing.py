from types import SimpleNamespace

import pytest

from tools.profiling import compare_sharp_surfacing as compare


@pytest.mark.parametrize("variant", compare.VARIANTS)
def test_each_sample_builds_a_fresh_field(variant, monkeypatch):
    warp = pytest.importorskip("warp")
    monkeypatch.setattr(warp, "synchronize_device", lambda _: None)
    settings = SimpleNamespace(**compare.VARIANTS["plain"])

    class Surface:
        built_key = None
        solver_sdf_key = "warm"
        render_extractor = SimpleNamespace(backend="sparse")
        builds = 0

        def invalidate(self):
            pass

        def invalidate_particle_state(self):
            self.built_key = None
            self.solver_sdf_key = None
            self.invalidate()

        def mesh(self):
            assert self.solver_sdf_key is None
            assert self.built_key is None
            self.built_key = self.solver_sdf_key = "warm"
            self.builds += 1
            return [], []

    surface = Surface()
    solver = SimpleNamespace(settings=settings, device="cpu", surface=surface)
    compare.measure_variant(solver, compare.VARIANTS[variant], repeats=3)
    assert surface.builds == 4
    assert vars(settings) == compare.VARIANTS["plain"]


@pytest.mark.parametrize(
    "argv", [("--frames", "0"), ("--emit-frames", "-1"), ("--repeats", "0")]
)
def test_invalid_frame_or_timing_settings_are_rejected(argv):
    with pytest.raises(SystemExit, match="2"):
        compare.parse(argv)
