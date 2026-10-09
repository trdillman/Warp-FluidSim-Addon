"""Run several Blender acceptance scripts in one background Blender session.

Usage (from the repository root)::

    blender --python-use-system-env --background --factory-startup \
        --python tools/gate/run_blender_batch.py -- SPEC.json RESULTS.json

SPEC.json lists ``{"script", "args", "marker"}`` entries plus the gate's
``failure_markers``. Every script gets the state a standalone
``--factory-startup`` run would give it: factory settings with the default
scene, the add-on unregistered, the repository's add-on and test helper
modules dropped from ``sys.modules``, and its own ``sys.argv``. Its stdout and
stderr (including tracebacks Blender prints from handlers) are captured, and it
passes only when it finished, printed its success marker and printed no failure
marker. RESULTS.json is rewritten after every script, so a crash leaves the
finished entries behind. The gate reruns any entry that did not pass, alone in
a fresh Blender, so a leak between scripts cannot decide a verdict.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
import traceback

import bpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ADDON_NAME = "fluidSim_warp_blender51"
_REPO_PREFIXES = tuple(os.path.join(ROOT, part) + os.sep for part in (ADDON_NAME, "tests", "tools"))


class _Tee:
    """Write-through stream that also keeps what one script printed."""

    def __init__(self, stream, parts):
        self._stream = stream
        self._parts = parts

    def write(self, text):
        self._parts.append(text)
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _drop_repo_modules():
    """Forget the add-on and repo helper modules; Warp itself stays loaded."""
    this = os.path.abspath(__file__)
    for name, module in list(sys.modules.items()):
        path = os.path.abspath(getattr(module, "__file__", None) or "")
        if path == this:
            continue
        if name == ADDON_NAME or name.startswith(ADDON_NAME + ".") or path.startswith(_REPO_PREFIXES):
            sys.modules.pop(name, None)


def _reset():
    import addon_utils

    try:
        addon_utils.disable(ADDON_NAME, default_set=False)
    except Exception:  # noqa: BLE001 - absent or already disabled
        pass
    bpy.ops.wm.read_factory_settings(use_empty=False)
    _drop_repo_modules()


def _run_one(entry, failure_markers):
    script = os.path.join(ROOT, entry["script"])
    args = list(entry.get("args", ()))
    sys.argv = [bpy.app.binary_path, "--background", "--factory-startup", "--python", script]
    if args:
        sys.argv += ["--", *args]
    parts = []
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(stdout, parts), _Tee(stderr, parts)
    reason = None
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            reason = f"exit {exc.code}"
    except BaseException:  # noqa: BLE001 - a failing script must not end the batch
        traceback.print_exc()
        reason = "exception"
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = stdout, stderr
    text = "".join(parts)
    if reason is None:
        found = [marker for marker in failure_markers if marker in text]
        if found:
            reason = f"failure marker {found[0]!r}"
        elif entry["marker"] not in text:
            reason = f"missing marker {entry['marker']!r}"
    return reason


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, results_path = argv
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    os.chdir(ROOT)
    results = []
    for entry in spec["entries"]:
        print(f"\n== batch: {entry['script']} ==", flush=True)
        start = time.perf_counter()
        try:
            _reset()
            reason = _run_one(entry, spec["failure_markers"])
        except BaseException:  # noqa: BLE001 - a broken reset fails this entry only
            traceback.print_exc()
            reason = "reset failed"
        seconds = round(time.perf_counter() - start, 1)
        verdict = "PASS" if reason is None else "FAIL"
        print(f"BATCH-RESULT {verdict} {entry['script']} {seconds}s" + (f" ({reason})" if reason else ""),
              flush=True)
        results.append({"script": entry["script"], "passed": reason is None,
                        "reason": reason, "seconds": seconds})
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    print("BATCH-COMPLETE", flush=True)


main()
