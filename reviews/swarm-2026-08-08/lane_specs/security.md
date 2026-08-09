# Lane: Trust & Safety (Security)
- **Dimension name:** Trust & Safety
- **ID prefix:** SEC
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/security.json

Follow `_common.md` in this same directory.

## Scope & Approach
This is a **cross-cutting, pattern-driven** security pass over the WHOLE `fluidSim_warp_blender51/` package plus `tools/` and `fluidSim_synth/`. Do NOT re-report correctness bugs that other lanes already cover (numerics, undo, types). Report ONLY the security/trust angle.

Use Grep to find these dangerous patterns, then Read and assess each hit:
- `subprocess`, `os.system`, `os.popen`, `Popen`, `shell=True`, `check_output` — subprocess invocation
- `pickle`, `allow_pickle`, `yaml.load`, `marshal`, `shelve` — deserialization
- `\beval\b`, `\bexec\b`, `compile(`, `__import__` — code execution
- `ctypes`, `c_char_p`, `c_void_p`, `c_uint64`, `pointer`, `cast(` — FFI / raw memory
- file paths built from user/RNA input for cache, preset, export, import — `open(`, `Path(`, `os.path.join` with untrusted segments
- `blender_manifest.toml` parsing / wheel loading / extension permissions
- temp-file creation (`tempfile`, `mkstemp`, predictable names)
- network calls (`urllib`, `requests`, `socket`)

## Dimension Focus
1. **Command injection / subprocess safety:** any subprocess or shell call with user-controllable args, shell=True, or unescaped paths. Blender headless launches, gate runners, profilers.
2. **Unsafe deserialization:** pickle/allow_pickle on cache or preset data; yaml.load without SafeLoader; eval/exec on dynamic content; JSON used as a trust boundary without schema validation.
3. **ctypes / GPU FFI safety:** raw pointer arithmetic (cuda_reduction.py) — bounds checks, device lifecycle, arbitrary-memory read/write, stale cubin loading from disk.
4. **Path traversal / file safety:** cache, preset, export, import paths derived from RNA properties or scene data — `../` traversal, absolute-path injection, symlink races, non-atomic writes over user files.
5. **Manifest & extension trust:** blender_manifest.toml — wheel/dependency loading, declared permissions vs actual filesystem/network access, version-string handling.
6. **Input validation at trust boundaries:** preset import (foreign .json/.npz), scene-input coercion, unbounded int/float fed to allocators or ctypes.
7. **Secrets / temp files:** predictable temp paths, credentials in code, world-writable files.

## Severity (security-calibrated)
- critical: remote/arbitrary code execution, unsafe deserialization of untrusted data, command injection
- high: path traversal to arbitrary location, missing validation enabling crash/DoS on the host, unsafe eval/pickle on semi-trusted input
- medium: defense-in-depth gap, risky pattern with partial mitigation
- low: hardening / best-practice improvement
