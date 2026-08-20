# Third-Party Notices

FluidSim Warp is distributed under GPL-3.0-or-later. **FluidSim delivery archives do not include Blender, NVIDIA Warp, CUDA runtime packages, Python wheels, or other third-party runtime binaries.**

NVIDIA Warp `warp-lang 1.16.0+cu12` is the certified external runtime. The supplied Linux wheel is the validation input available on this host; it does not restrict FluidSim's Windows x64 and Linux x64 product support. Users and CI systems must provision the matching platform runtime separately. FluidSim does not download, install, extract, or redistribute it at runtime.

## External runtime references

| Dependency or resource | Certified version/status | License or terms | Included in FluidSim ZIPs? |
|---|---:|---|---:|
| NVIDIA Warp (`warp-lang`) | 1.16.0+cu12 | Apache License, Version 2.0 | No |
| CUDA runtime/compiler components used by Warp | As provided by the separately installed Warp distribution | NVIDIA Software License Agreement | No |
| NVIDIA libmathdx components used by Warp | As provided by the separately installed Warp distribution | NVIDIA upstream terms | No |
| NumPy | Version supplied by Blender 5.1.0 or newer | BSD 3-Clause License | No separate FluidSim copy |
| Blender Python API | Blender 5.1.0 or newer | GNU General Public License | No |

The upstream Warp distribution carries its own complete license inventory, including `LICENSE.md`, `cuda-LICENSE.txt`, `nvrtc-LICENSE.txt`, and `libmathdx-LICENSE.txt`. Those files remain with the separately provisioned Warp runtime and are **not copied into FluidSim deliverables**.

## Distribution and runtime policy

- Blender and Warp packages are validation inputs only and must remain outside FluidSim delivery archives.
- `blender_manifest.toml` declares no Python wheels.
- Release inspection rejects `.whl` payloads, Blender runtime archives, Warp package directories, and Warp distribution metadata inside the installable extension.
- FluidSim performs no runtime network request, `pip` invocation, or automatic dependency installation.
- `release/wheels.lock.json` records the exact external Warp validation binary and checksum; it is metadata, not a bundled dependency.

## Bundled first-party render resource

`resources/geometry_nodes/fluidsim_render_nodes.blend` is generated from
`release/generate_geometry_nodes.py` using the supplied Blender 5.2.0 LTS
package. The six P4 node groups and generated resource are FluidSim project
content covered by the project GPL-3.0-or-later license. The file contains no Blender,
Warp, CUDA, wheel, or other third-party runtime binary payload.
