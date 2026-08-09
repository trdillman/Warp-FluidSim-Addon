# Lane: Serialization & IO
- **Dimension name:** Serialization & IO
- **ID prefix:** SERIO
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/io.json

Follow `_common.md` in this same directory.

## Scope (review only these files)
- fluidSim_warp_blender51/io/cache.py  (~412) and io/__init__.py
- fluidSim_warp_blender51/blender/scene_bridge.py  (~2440 — Blender scene <-> sim bridge, cache/playback)
- fluidSim_warp_blender51/blender/sources.py

## Dimension Focus
- Cache data integrity: write-path correctness, partial-write handling, frame indexing, overwrite safety.
- Atomic writes: does cache write use temp-file + atomic rename to avoid corruption on crash? (Direct overwrite of a live cache file is a defect.)
- Version migration / backward-compat: is there a format version field, and can old caches be loaded or are they silently misread?
- Encode/decode round-trip: endian/dtype handling, numpy array save/load correctness, truncation or reshaping bugs.
- Frame cache playback: frame-change handlers, depsgraph evaluation, does playback read the correct frame, off-by-one in frame indexing.
- .vdb / .ply / point-cache export correctness if present.
