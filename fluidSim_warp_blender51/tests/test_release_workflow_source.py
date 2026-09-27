from __future__ import annotations

import re
from pathlib import Path

from fluidsim_testpkg import version

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(name: str) -> str:
    source = _workflow()
    start = source.index(f"\n  {name}:\n")
    following = re.search(r"\n  [a-z][a-z0-9_-]*:\n", source[start + 1:])
    end = start + 1 + following.start() if following else len(source)
    return source[start:end]


def test_ci_workflow_lives_where_github_reads_it():
    assert WORKFLOW.is_file()
    # GitHub only reads workflows at the repository root; a copy inside the
    # add-on folder silently never runs.
    assert not (ROOT / ".github").exists()


def test_ci_runs_on_pull_requests_and_master():
    source = _workflow()
    assert "\n  pull_request:\n" in source
    assert "branches: [master]" in source
    assert "main" not in re.findall(r"branches: \[([^\]]*)\]", source)[0]
    assert "permissions:\n  contents: read" in source


def test_ci_uses_hosted_runners_without_a_gpu():
    source = _workflow()
    assert "self-hosted" not in source
    assert source.count("runs-on: ubuntu-latest") == 2
    assert 'CUDA_VISIBLE_DEVICES: "-1"' in source


def test_checks_job_lints_verifies_compiles_and_tests():
    job = _job("checks")
    assert "working-directory: fluidSim_warp_blender51" in job
    assert "python -m ruff check ." in job
    assert "python tools/release/build_extension.py --verify-only" in job
    assert "python -m compileall" in job
    assert "python -m pytest -q" in job


def test_checks_job_validates_release_kernels_on_warp_cpu():
    job = _job("checks")
    for check in ("phase4", "phase5", "custom_effects"):
        step = f"python tools/release/validate_{check}_warp.py"
        assert job.count(step) == 1
        tail = job[job.index(step):].split("\n      - ", 1)[0]
        assert "--device cpu" in tail


def test_package_job_builds_installs_and_smoke_tests_on_blender():
    job = _job("package")
    assert "needs: checks" in job
    assert "python tools/release/build_extension.py" in job
    assert '--blender "$BLENDER"' in job
    assert "--smoke-platform linux-x64" in job
    assert "sha256sum -c -" in job
    assert "FLUIDSIM_EXTERNAL_WARP_ROOT" not in job
    assert "hashFiles('tools/release/wheels.lock.json')" in job
    assert "fluidSim_warp_blender51/wheels/*.whl" in job


def test_package_job_pins_the_certified_blender_by_checksum():
    job = _job("package")
    blender = re.search(r'BLENDER_VERSION: "([0-9.]+)"', job).group(1)
    assert blender == ".".join(str(part) for part in version.CERTIFIED_BLENDER_VERSION)
    assert re.search(r"BLENDER_SHA256: [0-9a-f]{64}\n", job)


def test_ci_derives_versions_instead_of_hardcoding_them():
    source = _workflow()
    assert "fluidsim_warp-" not in source
    assert version.VERSION_STRING not in source
    # The Warp pin lives only in wheels.lock.json and version.py.
    assert "warp-lang" not in source
    assert "1.17" not in source


def test_ci_uses_current_actions():
    source = _workflow()
    for action in re.findall(r"uses: (actions/[a-z-]+)@", source):
        assert action in {
            "actions/checkout",
            "actions/setup-python",
            "actions/cache",
            "actions/upload-artifact",
        }
    assert "actions/checkout@v7" in source
    assert "actions/setup-python@v7" in source
    assert "actions/upload-artifact@v7" in source
    assert "actions/cache@v6" in source


def test_ci_installs_the_pinned_requirements_file():
    requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert requirements == [
        "pytest==9.1.1",
        "ruff==0.16.2",
        "numpy==2.3.4",
        f"warp-lang=={version.CERTIFIED_WARP_VERSION}",
    ]
    source = _workflow()
    assert source.count(
        "cache-dependency-path: fluidSim_warp_blender51/requirements-dev.txt"
    ) == 2
    assert source.count(
        "python -m pip install --disable-pip-version-check "
        "-r fluidSim_warp_blender51/requirements-dev.txt"
    ) == 2


def test_installed_release_oracles_match_the_current_identity():
    for relative in (
        "../tools/release/smoke_test_blender.py",
        "../tools/release/validate_phase5_blender.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert (
            version.VERSION_STRING in source
            or repr(version.VERSION) in source
            or "version.py" in source
        ), relative
