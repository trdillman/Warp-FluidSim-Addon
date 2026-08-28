from types import SimpleNamespace

from tools import profile_user_path


def test_expected_surface_publications_follow_solver_frame_cadence():
    frames = [{"solver_frame": frame} for frame in range(1, 7)]

    assert profile_user_path._expected_surface_publications(frames, 2) == 4
    assert profile_user_path._expected_surface_publications(frames, 1) == 6


def test_expected_surface_publications_handle_empty_measurement():
    assert profile_user_path._expected_surface_publications([], 2) == 0


def test_warm_rows_drop_lazy_compilation_frames_from_six_frame_profile():
    rows = [{"frame": frame} for frame in range(6)]

    assert profile_user_path._warm_rows(rows) == rows[3:]
    assert profile_user_path._warm_rows(rows[:3]) == rows[:3]


def test_surface_timing_probe_separates_readback_from_host_work():
    ticks = iter((0.0, 1.0, 3.0, 4.0))
    probe = profile_user_path._SurfaceTimingProbe(clock=lambda: next(ticks))

    probe.begin_frame(frame=4)
    with probe.stage("surface_extract"), probe.device_readback():
        pass
    row = probe.end_frame(handler_total_ms=10.0)

    assert row["frame"] == 4
    assert row["surface_extract_host_ms"] == 2000.0
    assert row["surface_extract_readback_ms"] == 2000.0
    assert row["surface_extract_calls"] == 1
    assert row["surface_extract_readback_calls"] == 1
    assert row["surface_attributed_ms"] == 4000.0
    assert row["handler_remainder_ms"] == 0.0
    assert row["attribution_overflow_ms"] == 3990.0


def test_surface_timing_probe_reports_whole_handler_remainder_and_isolates_frames():
    ticks = iter((0.0, 0.001, 0.001, 0.003, 0.003, 0.004))
    probe = profile_user_path._SurfaceTimingProbe(clock=lambda: next(ticks))

    probe.begin_frame(frame=1)
    with probe.stage("surface_build"):
        pass
    with probe.stage("output_publication"):
        pass
    first = probe.end_frame(handler_total_ms=10.0)

    probe.begin_frame(frame=2)
    with probe.stage("surface_attribute_adaptation"):
        pass
    second = probe.end_frame(handler_total_ms=8.0)

    assert first["surface_build_host_ms"] == 1.0
    assert first["output_publication_host_ms"] == 2.0
    assert first["surface_attributed_ms"] == 3.0
    assert first["handler_remainder_ms"] == 7.0
    assert second["surface_attribute_adaptation_host_ms"] == 1.0
    assert second["surface_build_host_ms"] == 0.0
    assert second["surface_attributed_ms"] == 1.0
    assert second["handler_remainder_ms"] == 7.0


def test_surface_timing_probe_does_not_mislabel_unclassified_readback():
    ticks = iter((0.0, 0.002))
    probe = profile_user_path._SurfaceTimingProbe(clock=lambda: next(ticks))

    probe.begin_frame(frame=1)
    with probe.device_readback():
        pass
    row = probe.end_frame(handler_total_ms=5.0)

    assert row["unclassified_readback_ms"] == 2.0
    assert row["surface_attributed_ms"] == 0.0
    assert row["handler_remainder_ms"] == 5.0


def test_surface_evidence_binding_fails_closed_on_approval_or_missing_identity():
    evidence = {
        "claim": {
            "label": "RAW-PROFILER-EVIDENCE-NOT-APPROVAL",
            "production_approved": False,
        },
        "source": {
            "git_head": "a" * 40,
            "addon_git_tree": "b" * 40,
            "addon_source_sha256": "c" * 64,
            "addon_module_file": "C:/repo/fluidSim_warp_blender51/__init__.py",
            "profiler_sha256": "e" * 64,
        },
        "runtime": {
            "blender_version": "5.1.0",
            "python_version": "3.11.0",
            "warp_version": "1.16.0",
            "device": "cuda:0",
        },
        "process": {"pid": 1234},
        "contention": {"classification": "not_observed"},
        "workload": {"frames_requested": 6, "surface_enabled": True},
        "correctness": {"fingerprint_sha256": "d" * 64},
    }

    assert profile_user_path._surface_evidence_binding_errors(evidence) == []

    evidence["claim"]["production_approved"] = True
    evidence["source"]["addon_source_sha256"] = ""
    assert profile_user_path._surface_evidence_binding_errors(evidence) == [
        "claim.production_approved must remain false",
        "source.addon_source_sha256 is missing",
    ]


def test_surface_instrumentation_wraps_only_observed_boundaries_and_restores():
    class Surface:
        def build(self):
            return "built"

    class Extractor:
        def extract(self, _field):
            return "extracted"

    class HostFrame:
        def output_attributes(self, _settings, output_kind):
            return output_kind

    surface = Surface()
    surface.render_extractor = Extractor()
    solver = SimpleNamespace(surface=surface)
    output = SimpleNamespace(
        publish_payload=lambda *_args: "published",
        _surface_values=lambda *_args: "adapted",
    )
    originals = (
        Surface.build,
        Extractor.extract,
        HostFrame.output_attributes,
        output.publish_payload,
        output._surface_values,
    )
    probe = profile_user_path._SurfaceTimingProbe()
    restore = profile_user_path._install_surface_attribution(
        probe, solver, output, HostFrame)

    probe.begin_frame(frame=1)
    assert surface.build() == "built"
    assert surface.render_extractor.extract(None) == "extracted"
    assert HostFrame().output_attributes(None, "particles") == "particles"
    assert HostFrame().output_attributes(None, "surface") == "surface"
    assert output._surface_values(None) == "adapted"
    assert output.publish_payload(None) == "published"
    row = probe.end_frame(handler_total_ms=100.0)
    restore()

    assert row["surface_build_calls"] == 1
    assert row["surface_extract_calls"] == 1
    assert row["surface_attribute_adaptation_calls"] == 2
    assert row["output_publication_calls"] == 1
    assert (
        Surface.build,
        Extractor.extract,
        HostFrame.output_attributes,
        output.publish_payload,
        output._surface_values,
    ) == originals


def test_addon_source_digest_ignores_bytecode_but_binds_source_bytes(tmp_path):
    addon = tmp_path / "fluidSim_warp_blender51"
    addon.mkdir()
    source = addon / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    cache = addon / "__pycache__"
    cache.mkdir()
    bytecode = cache / "module.pyc"
    bytecode.write_bytes(b"first")

    first = profile_user_path._source_tree_digest(addon)
    bytecode.write_bytes(b"second")
    second = profile_user_path._source_tree_digest(addon)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    third = profile_user_path._source_tree_digest(addon)

    assert first == second
    assert first["file_count"] == 1
    assert first["sha256"] != third["sha256"]
