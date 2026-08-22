"""Regression coverage for the explicit external-Warp developer override."""

from __future__ import annotations

import importlib
import sys


def test_external_warp_override_requires_explicit_opt_in(monkeypatch, tmp_path):
    discovery = importlib.import_module("fluidSim_warp_blender51.warp_discovery")
    candidate = tmp_path / "external"
    candidate.mkdir()
    monkeypatch.delenv("FLUIDSIM_ALLOW_EXTERNAL_WARP", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert discovery.enable_developer_override(str(candidate)) is False
    assert str(candidate) not in sys.path

    monkeypatch.setenv("FLUIDSIM_ALLOW_EXTERNAL_WARP", "1")
    assert discovery.enable_developer_override(str(candidate)) is True
    assert sys.path[0] == str(candidate.resolve())
