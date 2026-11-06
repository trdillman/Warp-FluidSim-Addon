"""A fallible retired runtime must never leave a released candidate authoritative."""
from types import SimpleNamespace as NS

import pytest
from test_scene_bridge_settings import _install_scene_bridge_stubs


@pytest.mark.parametrize("_attempt", range(2))
@pytest.mark.parametrize("lane", ["initialize", "reinitialize", "reset"])
@pytest.mark.parametrize("fault", ["inputs", "solver", "cuda", None])
def test_replacement_retirement_has_coherent_authority(monkeypatch, lane, fault, _attempt):
    bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import simulation_admission as admission
    from fluidSim_warp_blender51.sim import memory_safety

    scene = NS(frame_start=1, fluidsim=NS(status=""))
    events = []

    def retire(label):
        events.append(label)
        if label == fault:
            raise RuntimeError(f"injected {label} retirement failure")

    class Runtime:
        device = "cpu"
        settings = NS(device="cpu")
        particles = NS(capacity=10)

        def __init__(self, name):
            self.name = name
            self.releases = 0

        def release(self):
            self.releases += 1
            retire(self.name)

    sess = bridge.Session()
    old, candidate = Runtime("solver"), Runtime("candidate")
    sess.solver = old
    sess.scene_inputs = NS(clear=lambda: retire("inputs"))
    sess.bind_scene(id(scene) if lane != "initialize" else -1, "old-key")
    new_inputs = NS(clear=lambda: retire("new-inputs"))
    snapshot = admission.AdmissionSnapshot(
        scene_identity=id(scene), scene_key="new-key", frame=1, frame_start=1,
        depsgraph_generation=None, configured_device="cpu", domain_transform=None,
        domain_min=(0, 0, 0), domain_max=(1, 1, 1), settings=NS(device="cpu"),
        profiling_enabled=False, profiling_sync=False, scene_inputs=new_inputs,
        descriptors=(), extra_plans=(), colliders=(), force_fields=(),
        height_source=None, seeds=(), footprints=(), collider_footprints=(),
        resources={}, capacity_plan=None)
    owner = admission.SimulationAdmission(
        sess, scene_identity=id, applied_state_capture=lambda *_a, **_k: None,
        reset_settings_apply=lambda *_a, **_k: None)
    monkeypatch.setattr(owner, "_preflight", lambda *_a, **_k: (snapshot, None))
    monkeypatch.setattr(owner, "_prepare_solver_replacement", lambda *_a: (candidate, ()))
    monkeypatch.setattr(owner, "_prepare_outputs", lambda *_a: None)
    from fluidSim_warp_blender51.blender import output_publication
    monkeypatch.setattr(output_publication, "update_outputs", lambda *_a: None)
    cuda_calls = []

    def release_cuda(_device):
        cuda_calls.append(_device)
        # The retirement operation fails; candidate cleanup remains usable.
        if len(cuda_calls) == 1:
            retire("cuda")

    monkeypatch.setattr(memory_safety, "release_cuda_memory", release_cuda)
    run = owner.reset if lane == "reset" else owner.initialize
    if fault:
        with pytest.raises(RuntimeError, match=f"{fault} retirement failure"):
            run(scene)
        assert not sess.active
        assert sess.scene_identity is None
        assert candidate.releases == 1
        assert "preserved" not in scene.fluidsim.status.lower()
        assert fault in sess.error
        assert events.count("new-inputs") == 1
    else:
        run(scene)
        assert sess.owns_scene(scene)
        assert sess.solver is candidate
        assert candidate.releases == 0
        assert not sess.error
    # Every old owner gets one retirement attempt even if an earlier one fails.
    assert old.releases == 1
    assert events.count("inputs") == 1
