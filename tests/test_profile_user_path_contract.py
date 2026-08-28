from types import SimpleNamespace

import numpy as np
import pytest

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


def _valid_surface_evidence():
    return {
        "claim": {
            "label": "RAW-PROFILER-EVIDENCE-NOT-APPROVAL",
            "production_approved": False,
        },
        "source": {
            "git_head": "0123456789abcdef0123456789abcdef01234567",
            "addon_git_tree": "89abcdef0123456789abcdef0123456789abcdef",
            "addon_source_sha256": "0123456789abcdef" * 4,
            "addon_module_file": "C:/repo/fluidSim_warp_blender51/__init__.py",
            "profiler_sha256": "fedcba9876543210" * 4,
        },
        "runtime": {
            "blender_version": "5.1.0",
            "python_version": "3.11.0",
            "warp_version": "1.16.0",
            "device": "cuda:0",
        },
        "process": {"pid": 1234},
        "contention": {
            "classification": "graphics_clients_unavailable",
            "comparison_eligible": False,
        },
        "workload": {
            "frames_requested": 6,
            "frames_measured": 6,
            "surface_enabled": True,
        },
        "correctness": {
            "fingerprint_sha256": "13579bdf02468ace" * 4,
            "surface_attribute_collection_complete": True,
        },
        "verdict": {
            "result": "RAW-PROFILER-EVIDENCE-VALID",
            "validation_enabled": True,
            "contract_checks_passed": True,
            "contract_failures": [],
            "surface_stage_activated": True,
            "added_synchronization_calls": 0,
            "observed_synchronization_calls": 0,
            "comparison_eligible": False,
        },
    }


def test_surface_evidence_binding_fails_closed_on_approval_or_missing_identity():
    evidence = _valid_surface_evidence()

    assert profile_user_path._surface_evidence_binding_errors(evidence) == []

    evidence["claim"]["production_approved"] = True
    evidence["source"]["addon_source_sha256"] = ""
    assert profile_user_path._surface_evidence_binding_errors(evidence) == [
        "claim.production_approved must remain false",
        "source.addon_source_sha256 is missing",
    ]


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    (
        (("workload", "surface_enabled"), False,
         "workload.surface_enabled must be true"),
        (("workload", "frames_requested"), 0,
         "workload.frames_requested must be a positive integer"),
        (("workload", "frames_measured"), -1,
         "workload.frames_measured must be a positive integer"),
        (("process", "pid"), 0,
         "process.pid must be a positive integer"),
        (("source", "git_head"), "a" * 40,
         "source.git_head is not a credible 40-character hex identity"),
        (("source", "addon_source_sha256"), "f" * 64,
         "source.addon_source_sha256 is not a credible 64-character hex identity"),
        (("correctness", "fingerprint_sha256"), "0" * 64,
         "correctness.fingerprint_sha256 is not a credible 64-character hex identity"),
    ),
)
def test_surface_evidence_binding_rejects_invalid_runtime_facts(
        path, value, expected):
    evidence = _valid_surface_evidence()
    evidence[path[0]][path[1]] = value

    assert expected in profile_user_path._surface_evidence_binding_errors(evidence)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("surface_stage_activated", False,
         "verdict.surface_stage_activated must be true"),
        ("added_synchronization_calls", 1,
         "verdict.added_synchronization_calls must be zero"),
        ("contract_checks_passed", False,
         "verdict.contract_checks_passed must be true"),
        ("validation_enabled", False,
         "verdict.validation_enabled must be true"),
    ),
)
def test_surface_evidence_binding_rejects_unproven_final_verdict(
        field, value, expected):
    evidence = _valid_surface_evidence()
    evidence["verdict"][field] = value

    assert expected in profile_user_path._surface_evidence_binding_errors(evidence)


def test_surface_evidence_binding_rejects_missing_contract_verdict_and_failure():
    missing = _valid_surface_evidence()
    missing["verdict"].pop("contract_checks_passed")
    failed = _valid_surface_evidence()
    failed["verdict"]["contract_failures"] = ["viewport mean bounded"]

    assert (
        "verdict.contract_checks_passed is missing"
        in profile_user_path._surface_evidence_binding_errors(missing)
    )
    assert (
        "verdict.contract_failures must be empty"
        in profile_user_path._surface_evidence_binding_errors(failed)
    )


def test_unvalidated_surface_evidence_is_raw_only_and_never_comparison_eligible():
    evidence = _valid_surface_evidence()
    evidence["contention"]["comparison_eligible"] = True
    attribution_rows = [{
        "surface_build_calls": 1,
        "surface_extract_calls": 1,
        "surface_attribute_adaptation_calls": 1,
    }]

    verdict, errors = profile_user_path._finalize_surface_evidence(
        evidence,
        validation_enabled=False,
        checks=[],
        attribution_rows=attribution_rows,
        added_synchronization_calls=0,
    )

    assert verdict["result"] == "RAW-PROFILER-EVIDENCE-INVALID"
    assert verdict["comparison_eligible"] is False
    assert verdict["contract_checks_passed"] is False
    assert "verdict.validation_enabled must be true" in errors


def test_contract_failure_is_preserved_in_final_surface_verdict():
    evidence = _valid_surface_evidence()
    checks = [{"name": "viewport mean bounded", "ok": False}]

    verdict, errors = profile_user_path._finalize_surface_evidence(
        evidence,
        validation_enabled=True,
        checks=checks,
        attribution_rows=[{
            "surface_build_calls": 1,
            "surface_extract_calls": 1,
            "surface_attribute_adaptation_calls": 1,
        }],
        added_synchronization_calls=0,
    )

    assert verdict["contract_failures"] == ["viewport mean bounded"]
    assert "verdict.contract_checks_passed must be true" in errors
    assert "verdict.contract_failures must be empty" in errors


def test_wddm_graphics_client_gap_keeps_contention_evidence_raw_only():
    snapshot = {
        "available": True,
        "compute_processes": [{
            "pid": 1234, "process_name": "blender.exe", "used_memory_mb": 64}],
        "graphics_clients": {
            "available": False,
            "reason": "WDDM graphics clients are not enumerable",
        },
    }

    evidence = profile_user_path._contention_evidence(
        snapshot, snapshot, own_pid=1234)

    assert evidence["classification"] == "graphics_clients_unavailable"
    assert evidence["comparison_eligible"] is False
    assert "raw-only" in evidence["comparison_ineligible_reason"]


def test_existing_solver_synchronizations_do_not_count_as_added_output_sync():
    evidence = _valid_surface_evidence()

    verdict, errors = profile_user_path._finalize_surface_evidence(
        evidence,
        validation_enabled=True,
        checks=[{"name": "runtime contract", "ok": True}],
        attribution_rows=[{
            "surface_build_calls": 1,
            "surface_extract_calls": 1,
            "surface_attribute_adaptation_calls": 1,
        }],
        added_synchronization_calls=0,
        observed_synchronization_calls=6,
    )

    assert verdict["result"] == "RAW-PROFILER-EVIDENCE-VALID"
    assert verdict["added_synchronization_calls"] == 0
    assert verdict["observed_synchronization_calls"] == 6
    assert errors == []


def test_binding_rejects_comparison_eligibility_when_wddm_clients_are_unknown():
    evidence = _valid_surface_evidence()
    evidence["contention"]["comparison_eligible"] = True
    evidence["verdict"]["comparison_eligible"] = True

    assert (
        "contention.comparison_eligible requires all GPU clients to be observed"
        in profile_user_path._surface_evidence_binding_errors(evidence)
    )


def test_all_surface_attributes_account_for_vorticity_count_readback():
    assert profile_user_path.expected_surface_attribute_numpy_calls("all") == 10
    assert (
        profile_user_path.expected_surface_attribute_bookkeeping_numpy_calls("all")
        == 1
    )


class _FakeForeachCollection:
    def __init__(self, count, values_by_property):
        self._count = count
        self._values = values_by_property

    def __len__(self):
        return self._count

    def foreach_get(self, name, target):
        target[:] = self._values[name]


class _FakeAttribute:
    def __init__(self, name, data_type, domain, values, property_name):
        self.name = name
        self.data_type = data_type
        self.domain = domain
        components = 4 if data_type == "FLOAT_COLOR" else 1
        self.data = _FakeForeachCollection(
            len(values) // components,
            {property_name: np.asarray(values)},
        )


class _FakeAttributes(list):
    def get(self, name):
        return next((attribute for attribute in self if attribute.name == name), None)


def _fake_bpy_surface(attributes):
    mesh = SimpleNamespace(
        vertices=_FakeForeachCollection(
            2, {"co": np.asarray((0, 0, 0, 1, 1, 1), dtype=np.float32)}),
        loops=_FakeForeachCollection(0, {"vertex_index": np.zeros(0, np.int32)}),
        polygons=_FakeForeachCollection(
            0,
            {
                "loop_start": np.zeros(0, np.int32),
                "loop_total": np.zeros(0, np.int32),
            },
        ),
        attributes=_FakeAttributes(attributes),
    )
    surface = SimpleNamespace(type="MESH", data=mesh)
    return SimpleNamespace(data=SimpleNamespace(
        objects=SimpleNamespace(get=lambda _name: surface)))


def test_correctness_fingerprint_binds_every_surface_attribute_and_display_color():
    bpy = _fake_bpy_surface([
        _FakeAttribute(
            "fluidsim_speed", "FLOAT", "POINT", (1.0, 2.0), "value"),
        _FakeAttribute(
            "display_color", "FLOAT_COLOR", "POINT",
            (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0), "color"),
    ])

    evidence = profile_user_path._output_correctness_evidence(
        bpy,
        rows=[],
        output_counts=[],
        expected_surface_attributes=("fluidsim_speed",),
        surface_attribute_mode="SPEED",
    )

    assert evidence["surface_attribute_collection_complete"] is True
    inventory = evidence["surface_attributes"]
    assert [item["name"] for item in inventory] == [
        "display_color", "fluidsim_speed"]
    assert inventory[0] == {
        "name": "display_color",
        "domain": "POINT",
        "data_type": "FLOAT_COLOR",
        "count": 2,
        "components": 4,
        "byte_count": 32,
        "value_sha256": inventory[0]["value_sha256"],
    }
    assert inventory[1]["byte_count"] == 8
    assert len(inventory[0]["value_sha256"]) == 64


def test_correctness_fingerprint_fails_closed_when_attribute_collection_is_incomplete():
    bpy = _fake_bpy_surface([])

    evidence = profile_user_path._output_correctness_evidence(
        bpy,
        rows=[],
        output_counts=[],
        expected_surface_attributes=("fluidsim_speed",),
        surface_attribute_mode="NONE",
    )

    assert evidence["surface_attribute_collection_complete"] is False
    assert evidence["surface_attribute_collection_errors"] == [
        "missing published surface attribute 'fluidsim_speed'"
    ]


def test_correctness_fingerprint_requires_display_color_from_configured_mode():
    bpy = _fake_bpy_surface([
        _FakeAttribute(
            "fluidsim_speed", "FLOAT", "POINT", (1.0, 2.0), "value"),
    ])

    evidence = profile_user_path._output_correctness_evidence(
        bpy,
        rows=[],
        output_counts=[],
        expected_surface_attributes=("fluidsim_speed",),
        surface_attribute_mode="SPEED",
    )

    assert evidence["surface_attribute_collection_complete"] is False
    assert "missing published surface attribute 'display_color'" in evidence[
        "surface_attribute_collection_errors"]


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
