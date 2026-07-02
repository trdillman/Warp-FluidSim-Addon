"""Diagnostics report writer.

Aggregates profiler output, per-frame stats, benchmark results, and
validation outcomes into a single JSON document for offline inspection.
"""

from __future__ import annotations

import json
import platform
import time


def build_report(
    profiler=None,
    frame_stats: list | None = None,
    benchmark: dict | None = None,
    validation: list | None = None,
    extra: dict | None = None,
) -> dict:
    report = {
        "tool": "fluidSim",
        "backend": "NVIDIA Warp / CUDA",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    try:
        import warp as wp

        report["warp"] = wp.config.version
    except Exception:
        report["warp"] = "unavailable"
    if profiler is not None:
        report["compute"] = profiler.compute_profile()
        report["timers_average"] = profiler.summary()
    if frame_stats:
        report["frames"] = [
            s.as_dict() if hasattr(s, "as_dict") else dict(s) for s in frame_stats
        ]
    if benchmark:
        report["benchmark"] = benchmark
    if validation:
        report["validation"] = validation
    if extra:
        report.update(extra)
    return report


def write_report(path: str, report: dict) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path
