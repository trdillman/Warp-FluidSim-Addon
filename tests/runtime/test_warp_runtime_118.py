"""4.0 lane W: Warp 1.18 is certified, 1.17 still loads, the driver floor is named.

Warp 1.18 PyPI wheels are CUDA 13 builds that need an NVIDIA R580-series or
newer driver. A user on an older driver must hear that by name before a
download, and a user who already has 1.17 must not be stranded by 4.0.
"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fluidSim_warp_blender51 import version, warp_installer
from fluidSim_warp_blender51.runtime import warp_runtime


def _fake_warp(version_text: str, *, driver=(13, 3), devices=True):
    def get_devices():
        if not devices:
            return []
        return [SimpleNamespace(is_cuda=True, alias="cuda:0", name="Test GPU",
                                total_memory=1024**3)]

    return SimpleNamespace(
        __version__=version_text,
        __file__="/owned/warp/__init__.py",
        init=lambda: None,
        get_devices=get_devices,
        get_cuda_driver_version=lambda: driver,
        get_cuda_toolkit_version=lambda: (13, 4),
    )


def _probe(fake):
    with patch.dict(sys.modules, {"warp": fake}):
        try:
            return warp_runtime.probe_runtime(force=True)
        finally:
            warp_runtime._cached_probe.cache_clear()


class Warp118CertificationTests(unittest.TestCase):
    def test_certified_identity_is_warp_118_cuda13(self):
        self.assertEqual(version.CERTIFIED_WARP_VERSION, "1.18.0")
        self.assertEqual(version.CERTIFIED_WARP_DISTRIBUTION, "1.18.0+cu13")
        self.assertEqual(version.MIN_WARP_VERSION, "1.17.0")
        self.assertEqual(version.CERTIFIED_DRIVER_MIN, (13, 0))
        self.assertIn("R580", version.CERTIFIED_DRIVER_LABEL)

    def test_118_is_certified(self):
        self.assertEqual(warp_runtime.warp_version_support("1.18.0"), ("certified", ""))

    def test_117_loads_as_older_than_certified_with_a_warning(self):
        status, message = warp_runtime.warp_version_support("1.17.0")
        self.assertEqual(status, "older")
        self.assertIn("older than the certified 1.18.0", message)
        probe = _probe(_fake_warp("1.17.0", driver=(12, 9)))
        self.assertTrue(probe.available)
        self.assertIn("older than the certified", probe.warning)

    def test_below_floor_and_other_major_stay_unsupported(self):
        self.assertEqual(warp_runtime.warp_version_support("1.16.0")[0], "unsupported")
        self.assertEqual(warp_runtime.warp_version_support("2.0.0")[0], "unsupported")
        self.assertEqual(warp_runtime.warp_version_support("1.19.0")[0], "untested")

    def test_driver_floor_reason_is_named(self):
        self.assertEqual(warp_runtime.cuda_driver_floor_reason(13030), "")
        self.assertEqual(warp_runtime.cuda_driver_floor_reason(13000), "")
        reason = warp_runtime.cuda_driver_floor_reason(12090)
        self.assertIn("R580", reason)
        self.assertIn("12.9", reason)
        self.assertIn("1.18.0", reason)
        # An unknown driver version never blocks.
        self.assertEqual(warp_runtime.cuda_driver_floor_reason(0), "")

    def test_probe_names_the_driver_floor_when_118_finds_no_device(self):
        probe = _probe(_fake_warp("1.18.0", driver=(12, 8), devices=False))
        self.assertFalse(probe.available)
        self.assertIn("R580", probe.message)
        self.assertEqual(probe.driver_floor_reason,
                         warp_runtime.cuda_driver_floor_reason(12080))

    def test_probe_on_117_with_old_driver_says_to_stay_on_117(self):
        probe = _probe(_fake_warp("1.17.0", driver=(12, 8)))
        self.assertTrue(probe.available)
        self.assertIn("R580", probe.warning)

    def test_installer_refuses_download_below_driver_floor(self):
        downloads = []
        with patch.object(warp_installer, "foreign_warp_block_reason", return_value=None), \
             patch.object(warp_installer, "_owned_candidates", return_value=[]), \
             patch.object(warp_installer, "_online_access_block_reason", return_value=None), \
             patch.object(warp_installer, "query_cuda_driver_version", return_value=12080), \
             patch.object(warp_installer, "_download_wheel",
                          side_effect=lambda *a, **k: downloads.append(a)):
            result = warp_installer.install_warp()
        self.assertFalse(result.success)
        self.assertIn("R580", result.message)
        self.assertEqual(downloads, [])

    def test_installer_keeps_owned_117_when_driver_is_below_floor(self):
        with patch.object(warp_installer, "foreign_warp_block_reason", return_value=None), \
             patch.object(warp_installer, "_owned_candidates", return_value=["/owned"]), \
             patch.object(warp_installer, "_owned_distribution_version", return_value="1.17.0"), \
             patch.object(warp_installer, "_online_access_block_reason", return_value=None), \
             patch.object(warp_installer, "query_cuda_driver_version", return_value=12080), \
             patch.object(warp_installer, "_download_wheel",
                          side_effect=AssertionError("must not download")):
            result = warp_installer.install_warp()
        self.assertTrue(result.success)
        self.assertTrue(result.already_installed)
        self.assertIn("R580", result.message)


if __name__ == "__main__":
    unittest.main()
