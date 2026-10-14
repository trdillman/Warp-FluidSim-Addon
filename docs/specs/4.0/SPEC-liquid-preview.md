# SPEC — Liquid Preview (screen-space fluid rendering in the viewport)

Owner module: new `blender/liquid_preview/` (`passes.py`, `shaders.py`, `handler.py`,
`reference.py`). Hooks: `blender/playback.py` (publish → "payload ready" event),
`blender/output_publication.py` (read-only access to the published position/whitewater
arrays; no new readback), `blender/viewport_overlay.py` (single draw registry shared with
the Deck). Depends on: nothing in the solver. Can land on the alpha.1 base.

## Trap this spec prevents

1. **A second device-to-host readback.** `frame_output.FrameOutput.positions()` is the
   one readback and it is counted (`device_readbacks`). Liquid Preview must consume the
   array Output Publication already produced for the PointCloud, by reference, in the
   same publication tick. A test pins `device_readbacks` unchanged with the mode on.
2. **Drawing from inside Output Publication.** Publication mutates IDs; drawing is a
   draw handler. The payload is handed over through a small immutable record
   (positions view, whitewater view + type attribute, live counts, frame id). The
   handler never touches `bpy.data`.
3. **Per-redraw host uploads of the whole array.** Upload to a `GPUVertBuf` once per
   *payload change* (frame id), not per redraw. Viewport orbiting re-runs the screen
   passes only. Growth re-allocates the VBO at the capacity-planner size, not the live
   count, to avoid per-frame re-creation.
4. **The curvature-flow trap.** 100+ iterations of curvature flow is what makes SSFR
   look slow. Use the narrow-range filter (Truong & Yuksel): two separable 1D passes
   plus one clean-up pass, window clamped to `[z - δ_low, z + δ_high]`, `δ = 10 r`,
   filter radius in pixels from `σ_i = ceil(H·σ / (2|z|·tan(α/2)))`.
5. **Depth mismatch with the scene.** Sprite depth must be written in the same
   projection as the viewport (`region_data.perspective_matrix`, window/view matrices
   from `gpu.matrix`), and the composite must respect the scene depth buffer so solids
   in front of the liquid occlude it. Read the viewport depth via
   `gpu.state.active_framebuffer_get().read_depth` in `POST_PIXEL` once per redraw at
   the offscreen resolution (half res by default).
6. **Vulkan/OpenGL divergence.** Only `GPUShaderCreateInfo` + `create_from_info`; no
   `bgl`, no raw GLSL version pragmas, no `gl_FragDepth` without declaring the
   `depth_write` in the create-info.
7. **Untestable visuals.** Blender 5.2 has `gpu.init()` for background mode. Acceptance
   is an offscreen render compared to a committed reference, not a screenshot.

## Design

Passes (all into `GPUOffScreen` targets at `scale` × viewport size, default 0.5):

1. **Depth sprites.** One quad per particle (instanced via a 4-vertex batch and a
   per-instance position/radius VBO). Fragment: `if (u²+v² > 1) discard; z = z_c + r·sqrt(1-u²-v²)`;
   write eye-space depth to an R32F target.
2. **Narrow-range filter.** Horizontal, vertical, clean-up. Uniforms: `δ_low`,
   `δ_high`, world radius, `H`, `tan(α/2)`.
3. **Normals.** From `∂P/∂x × ∂P/∂y` with the min-gradient silhouette rule.
4. **Thickness.** Additive sprite pass with blue-noise temporal jitter (one 64×64
   tile, frame-indexed) into an R16F target; Beer–Lambert `exp(-a(λ)·d)`.
5. **Composite** (`POST_PIXEL`): Schlick Fresnel `R0 + (1-R0)(1-N·V)^5`, refraction
   offset `uv + N.xy · d · κ` sampling the viewport colour, absorption tint, depth test
   against the scene depth, optional whitewater sprite layer on top (foam white, spray
   small/bright, bubbles tinted by absorption), alpha-blended.

Settings (scene-level display, OUTPUT_ONLY class in Pending Settings so they never
touch the solver): `liquid_preview_enabled`, `liquid_preview_scale`,
`liquid_preview_radius_multiplier`, `liquid_preview_absorption` (colour),
`liquid_preview_whitewater`. Deck exposes the toggle; the Look stage panel mirrors it.

Fallback: if shader compilation fails on a backend, log once, set a session flag,
draw nothing, and keep the PointCloud visible. Never raise inside a draw handler.

## Reference mockup

The Liquid Preview toggle in `docs/specs/4.0/liquid_deck_mockup.html` (https://claude.ai/artifact/5jjwB1o9BEDnYjseufEe1V) shows
the intended look and the UI contract: one Deck toggle and an `L` hotkey, the viewport
corner label switches to "Liquid Preview", thin regions get a bright Fresnel rim, thick
regions darken toward the absorption colour, and the densest regions read as foam. The
mockup's image is a 2D thickness splat, not SSFR; the SSIM references above are the
pixel truth.

## Acceptance

- `tests/ui/test_liquid_preview_reference.py` (Blender, `--background`, `gpu.init()`):
  render crown_splash frame 30 and dam_break frame 20 payloads at 960×540; compare to
  `tests/references/liquid_preview/*.png` with SSIM ≥ 0.97; regenerate references only
  with an explicit flag and a changelog line.
- `device_readbacks` and `host_frame_acquisitions` unchanged with the mode on (pinned).
- Budget: ≤ 4 ms per redraw at 1080p, 600k particles, scale 0.5, on the gate GPU,
  measured by `tools/profiling/profile_liquid_preview.py` (offscreen timers, 60-redraw
  median). Mode off: 0 ms and no handler registered.
- Gate: GUI gate gains a lane that toggles the mode on a demo and asserts the handler
  is registered, the composite target is non-empty, and the PointCloud remains
  published. Demo hash unchanged.
