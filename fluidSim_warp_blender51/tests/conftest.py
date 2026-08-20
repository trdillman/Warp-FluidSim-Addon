from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "fluidsim_testpkg"

if PACKAGE_NAME not in sys.modules:
    package = ModuleType(PACKAGE_NAME)
    package.__file__ = str(ROOT / "__init__.py")
    package.__package__ = PACKAGE_NAME
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package
