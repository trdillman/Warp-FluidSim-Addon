"""Fail unless a real Blender GUI playback probe reported success."""

from __future__ import annotations

import json
import sys


with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
if not report.get("passed"):
    raise RuntimeError(f"saved-demo GUI timeline playback failed: {report}")
print("SAVED-DEMO-GUI-PLAYBACK-PASS")
