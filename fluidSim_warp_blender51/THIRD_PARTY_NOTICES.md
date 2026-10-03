# Third-Party Notices

FluidSim Warp is distributed under GPL-3.0-or-later. The extension does not
bundle NVIDIA Warp; instead, users install Warp via a button in Add-on
Preferences that downloads the certified wheel from PyPI.

NVIDIA Warp `warp-lang 1.17.0` (the PyPI build for CUDA 12, CUDA Toolkit 12.9)
is the certified runtime. The Windows x64 and Linux x64 wheels are pinned by
SHA-256 in `../tools/release/wheels.lock.json`. When the user presses
**Install Warp** in Add-on Preferences, FluidSim downloads the wheel for the
current platform from PyPI, verifies its SHA-256 checksum, and installs it
into Blender's Python using pip. The download requires network access (the
extension declares the `network` permission for this purpose).

## Third-party components

| Dependency or resource | Certified version/status | License or terms | Included in FluidSim ZIPs? |
|---|---:|---|---:|
| NVIDIA Warp (`warp-lang`) | 1.17.0 (CUDA 12 build) | Apache License, Version 2.0 | Yes, as the unmodified upstream wheel |
| CUDA runtime/compiler components used by Warp | As shipped inside the Warp wheel | NVIDIA Software License Agreement | Yes, inside the Warp wheel |
| NVIDIA libmathdx components used by Warp | As shipped inside the Warp wheel | NVIDIA upstream terms | Yes, inside the Warp wheel |
| NumPy | Version supplied by Blender 5.2.0 or newer | BSD 3-Clause License | No separate FluidSim copy |
| Blender Python API | Blender 5.2.0 or newer | GNU General Public License | No |

The Warp wheel carries its own complete license inventory in
`warp_lang-1.17.0.dist-info/licenses/`, including `LICENSE.md`,
`cuda-LICENSE.txt`, `nvrtc-LICENSE.txt`, and `libmathdx-LICENSE.txt`. The wheel
is redistributed byte-for-byte, so those notices travel with it and are
installed alongside Warp.

## Distribution and runtime policy

- `blender_manifest.toml` declares `network` permission for the PyPI download.
- `tools/release/wheels.lock.json` pins the exact wheel filenames, sizes,
  SHA-256 checksums, and PyPI URLs. The installer refuses any wheel that does
  not match these checksums.
- The Install Warp button is idempotent: pressing it when the certified version
  is already installed reports success without reinstalling.
- A developer-only override (`FLUIDSIM_ALLOW_EXTERNAL_WARP=1`) can point at a
  Warp checkout for development; it is never used by release packages.

## Bundled first-party render resource

`resources/geometry_nodes/fluidsim_render_nodes.blend` is generated from
`../tools/release/generate_geometry_nodes.py` using the supplied Blender 5.2.0 LTS
package. The six P4 node groups and generated resource are FluidSim project
content covered by the project GPL-3.0-or-later license. The file contains no Blender,
Warp, CUDA, wheel, or other third-party runtime binary payload.
