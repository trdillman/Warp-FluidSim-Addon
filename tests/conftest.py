"""Repository-wide pytest runtime isolation."""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def writable_warp_kernel_cache(tmp_path_factory):
    """Keep optional Warp JIT output inside a writable test-owned directory."""
    try:
        import warp as wp
    except ImportError:
        yield
        return
    previous_cache = wp.config.kernel_cache_dir
    previous_tempdir = tempfile.tempdir
    previous_env = {
        name: os.environ.get(name)
        for name in ("TEMP", "TMP", "CUDA_CACHE_PATH")
    }
    runtime_temp = str(tmp_path_factory.mktemp("warp-runtime-temp"))
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp-kernel-cache"))
    tempfile.tempdir = runtime_temp
    for name in previous_env:
        os.environ[name] = runtime_temp
    try:
        yield
    finally:
        wp.config.kernel_cache_dir = previous_cache
        tempfile.tempdir = previous_tempdir
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
