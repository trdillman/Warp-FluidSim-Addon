"""Load developer tools that execute against an explicitly selected add-on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_addon_relative_tool(module_name: str, tool_name: str) -> ModuleType:
    """Load a tools-side module with relative imports rooted at the target add-on."""
    qualified_name = f"{module_name}.release.{tool_name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing

    path = Path(__file__).resolve().with_name(f"{tool_name}.py")
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load release tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module
