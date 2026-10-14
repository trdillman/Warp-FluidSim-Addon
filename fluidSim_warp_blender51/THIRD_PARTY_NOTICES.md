# Third-Party Notices

FluidSim Warp is distributed under GPL-3.0-or-later. The extension does not
bundle NVIDIA Warp; instead, users install Warp via a button in Add-on
Preferences that downloads the certified wheel from PyPI.

NVIDIA Warp `warp-lang 1.18.0` (the PyPI build for CUDA 13, CUDA Toolkit 13.4)
is the certified runtime. Warp 1.17.0 (CUDA 12) still loads, with a warning. The Windows x64 and Linux x64 wheels are pinned by
filename, size, SHA-256, and HTTPS URL in `warp_wheels.lock.json` (the same
document as `tools/release/wheels.lock.json` in the repository). When the
user presses **Install Warp** in Add-on Preferences, FluidSim downloads that
one wheel from PyPI, verifies its SHA-256 checksum, and extracts it into the
directory Blender assigns to this extension. The download uses the extension's
`network` permission and Blender's online-access flag. FluidSim does not
install a floating or "latest" Warp, and it does not write into another
add-on's directory or into Blender's user site-packages.

## Third-party components

| Dependency or resource | Certified version/status | License or terms | Included in FluidSim ZIPs? |
|---|---:|---|---:|
| NVIDIA Warp (`warp-lang`) | 1.18.0 (CUDA 13 build) | Apache License, Version 2.0 | No. Installed from PyPI by the preferences button |
| CUDA runtime/compiler components used by Warp | As shipped inside the Warp wheel | NVIDIA Software License Agreement | No. They arrive inside the installed Warp wheel |
| NVIDIA libmathdx components used by Warp | As shipped inside the Warp wheel | NVIDIA upstream terms | No. They arrive inside the installed Warp wheel |
| NumPy | Version supplied by Blender 5.2.0 or newer | BSD 3-Clause License | No separate FluidSim copy |
| Blender Python API | Blender 5.2.0 or newer | GNU General Public License | No |

The Warp wheel carries its own complete license inventory in
`warp_lang-1.18.0.dist-info/licenses/`, including `LICENSE.md`,
`cuda-LICENSE.txt`, `nvrtc-LICENSE.txt`, and `libmathdx-LICENSE.txt`. The
preferences button installs that unmodified wheel, so those notices are
installed alongside Warp. They are not copied into the FluidSim zip.

## Distribution and runtime policy

- `blender_manifest.toml` declares `network` permission for the PyPI download.
- `warp_wheels.lock.json` pins the exact wheel filenames, sizes, SHA-256
  checksums, and PyPI URLs. The installer refuses any wheel that does not
  match these checksums.
- The Install Warp button is idempotent: pressing it when this extension's own
  copy is already Warp 1.18.0 reports success and writes nothing.
- A developer-only override (`FLUIDSIM_ALLOW_EXTERNAL_WARP=1`) can point at a
  Warp checkout for development; it is never used by release packages.

## Bundled first-party render resource

`resources/geometry_nodes/fluidsim_render_nodes.blend` is generated from
`../tools/release/generate_geometry_nodes.py` using the supplied Blender 5.2.0 LTS
package. The six P4 node groups and generated resource are FluidSim project
content covered by the project GPL-3.0-or-later license. The file contains no Blender,
Warp, CUDA, wheel, or other third-party runtime binary payload.
