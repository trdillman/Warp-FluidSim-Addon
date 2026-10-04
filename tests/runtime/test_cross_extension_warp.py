"""The active Warp module is process-wide even when extensions bundle wheels."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fluidSim_warp_blender51.runtime import warp_runtime


class CrossExtensionWarpTests(unittest.TestCase):
    def test_preloaded_older_warp_is_rejected_before_init_despite_newer_wheel_on_path(self):
        with tempfile.TemporaryDirectory() as root:
            foundry = Path(root, "warp_foundry_runtime")
            fluidsim = Path(root, "fluidsim_runtime")
            for directory, version in ((foundry, "1.16.0"), (fluidsim, "1.17.0")):
                package = directory / "warp"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(
                    f"__version__ = {version!r}\n"
                    "def init():\n"
                    "    raise AssertionError('incompatible Warp must not initialize')\n",
                    encoding="utf-8",
                )
            with patch.object(sys, "path", [str(foundry), *sys.path]):
                # Model Warp Foundry importing Warp before FluidSim starts.
                with patch.dict(sys.modules):
                    sys.modules.pop("warp", None)
                    loaded = importlib.import_module("warp")
                    sys.path.insert(0, str(fluidsim))
                    try:
                        result = warp_runtime.probe_runtime(force=True)
                        self.assertFalse(result.available)
                        self.assertEqual(result.warp_version, "1.16.0")
                        self.assertEqual(result.module_file, str(foundry / "warp" / "__init__.py"))
                        self.assertIn("share the top-level warp module", result.message)
                        self.assertIn(str(foundry / "warp" / "__init__.py"), result.message)
                        self.assertIn("restart Blender", result.message)
                        self.assertIs(sys.modules["warp"], loaded)
                        self.assertEqual(sys.path[0], str(fluidsim))
                    finally:
                        sys.modules.pop("warp", None)
                        warp_runtime._cached_probe.cache_clear()

    def test_compatible_warp_keeps_existing_cuda_probe(self):
        class CompatibleWarp:
            __version__ = "1.17.0"
            __file__ = "/shared/wheels/warp/__init__.py"

            @staticmethod
            def init():
                return None

            @staticmethod
            def get_devices():
                return [type("Device", (), {
                    "is_cuda": True,
                    "alias": "cuda:0",
                    "name": "Test GPU",
                    "total_memory": 1024**3,
                })()]

        with patch.dict(sys.modules, {"warp": CompatibleWarp}):
            try:
                result = warp_runtime.probe_runtime(force=True)
                self.assertTrue(result.available)
                self.assertEqual(result.warp_version, "1.17.0")
            finally:
                warp_runtime._cached_probe.cache_clear()


if __name__ == "__main__":
    unittest.main()
