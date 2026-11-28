# FSPT experimental quickstart

Use FSPT to preview a supported FluidSim surface and render a still from a solved frame. These instructions apply to the reviewed experimental Windows source. Earlier native source tests verified the CUDA broker, a controlled device viewport and an actual Blender Final render. The ce6afc9a scheduling fix awaits an admitted native GUI rerun. The normal production gate is blocked and sustained Live remains unqualified. Package and installed evidence keep their exact source identities. See [device viewport qualification](DEVICE_VIEWPORT_QUALIFICATION.md) for the tested builds and scope.

## Install the provisional build

In Blender 5.2 LTS, open **Edit > Preferences > Get Extensions**, choose
**Install from Disk**, and select the provisional ZIP for your platform.
Keep the ZIP intact. In **Preferences > Add-ons > FluidSim Warp**, enable
the extension if needed and choose **Install Warp**. Warp is installed
separately. Use **Upgrade to Locked Warp** for an older extension-owned
installation, or **Install from File** to select the exact platform wheel
listed in the lock. Restart Blender if Preferences reports a staged change.
The recorded Windows source tests used Blender 5.2.0 and Warp 1.19.0.
The current Windows ZIP passed generic CPU smoke. Installed FSPT GPU
verification and native Linux execution remain open.

[Platform ZIP hashes and test scope](CURRENT_QUALIFICATION_STATUS.md#successor-provisional-package) ·
[FluidSim installation guide](../../fluidSim_warp_blender51/docs/user_guide.md) ·
[Warp wheel lock](../../fluidSim_warp_blender51/warp_wheels.lock.json)

## Open a supported scene

1. Open a disposable copy of your FluidSim scene in the experimental Blender environment. Select **FSPT (Experimental)** as the scene render engine, then select **Rendered** viewport shading. The **FSPT Experimental** panel in **Render Properties** exposes **CUDA Device**, **Viewport Samples**, **Viewport Resolution %**, **Final Samples**, **Max Bounces**, and **Seed**. The default CUDA device is `cuda:0`.
2. Use evaluated mesh geometry and a constant **Background** world. For each visible material graph, connect a constant-input **Diffuse BSDF** or **Glass BSDF** with roughness zero, or an **Emission** node, directly to the active **Material Output** Surface input. Use emissive meshes for scene lighting. The translator reports unsupported Principled and linked-input graphs, visible Blender light objects, and unsupported evaluated geometry such as point clouds. If needed, substitute a supported liquid material and hide unsupported presentation objects in the disposable copy. Keep your authored scene and simulation setup intact.
3. Leave the device-surface environment flag unset for ordinary use. The viewport asynchronously traces an evaluated mesh snapshot on CUDA and presents completed images through pinned host memory. Adjust **Viewport Samples** and **Viewport Resolution %** to trade preview cost for detail. Changes to the camera or render inputs can restart accumulation.

[Render engine and panel](../../fluidSim_warp_blender51/blender/fspt_engine.py) · [Supported scene and material translation](../../fluidSim_warp_blender51/blender/fspt_scene.py)

## Play, pause, and render a still

- Start **Preview** through FluidSim and let playback solve the frame you want. Use **Pause** to keep the current solver and timeline frame. Keep the viewport in **Rendered** shading and let samples accumulate on the unchanged surface. **Resume Preview** continues compatible paused work. FluidSim’s existing pending-settings workflow handles changes that require an apply or restart.
- For a camera still, pause on the exact solved timeline frame. In a cache workflow, use **Preview Cache** and choose a frame that the selected cache can validate. Set a perspective or orthographic camera, one enabled view layer and opaque film. Disable depth of field, motion blur, render border and multiview. Use **Render > Render Image** with your chosen **Final Samples**. Final uses a fresh evaluated snapshot and writes the **Combined** render result.
- If Final reports a waiting, invalid or wrong-frame publication, return to the eligible solved frame or advance through normal FluidSim playback to solve the desired frame. For cached playback, verify the selected sequence and frame through FluidSim. Final can refresh the surface for an already solved current frame. A render request never advances the solver to create a missing state. Let an active **Production Bake** release ownership before preparing that fluid for Final.

[FluidSim artist workflows](../ARTIST_WORKFLOWS.md#preview-pause-resume-apply-and-restart) · [Final publication checks](../../fluidSim_warp_blender51/blender/fspt_bridge.py#L222)

## Read the diagnostic

A viewport label reading **retained / update pending** describes an earlier eligible image while an update finishes. Use the labeled source and timeline when assessing the preview. If the renderer reports an unsupported shader or scene feature, address the named item in your experimental copy and retry. If it reports an unavailable CUDA device, correct the device or qualified runtime first. For a FluidSim output-specific error, use **Repair Outputs** where FluidSim offers it.

## Developer device verification

Launch a separate Blender process with `FLUIDSIM_FSPT_DEVICE_SURFACE=1` in its environment to exercise eligible device-surface components. Ambiguous native publication and manual fluid edits latch that view to evaluated geometry. The opt-in leaves Final on the evaluated path. The controlled device fixture has its own native receipts. Ordinary live GPU viewport eligibility, production optical quality and real-time claims remain open under the frozen contract.

[Device viewport qualification](DEVICE_VIEWPORT_QUALIFICATION.md) · [Current status](CURRENT_QUALIFICATION_STATUS.md) · [Acceptance contract](P0_ACCEPTANCE_CONTRACT.md)
