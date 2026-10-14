from __future__ import annotations

from pathlib import Path

from fluidsim_testpkg import version

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


def test_github_actions_workflows_remain_removed():
    for root in (REPOSITORY, ROOT):
        workflows = root / ".github" / "workflows"
        assert not list(workflows.glob("*.yml"))
        assert not list(workflows.glob("*.yaml"))


def test_local_checks_use_the_pinned_requirements_file():
    requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert requirements == [
        "pytest==9.1.1",
        "ruff==0.16.2",
        "numpy==2.3.4",
        f"warp-lang=={version.CERTIFIED_WARP_VERSION}",
    ]


def test_local_release_kernel_validators_remain_available():
    for check in ("phase4", "phase5", "custom_effects"):
        assert (REPOSITORY / "tools" / "release" / f"validate_{check}_warp.py").is_file()


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
