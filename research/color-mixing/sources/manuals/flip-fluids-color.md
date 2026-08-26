# First-party manual clip: FLIP Fluids color mixing

- Original: https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Attributes-and-Data-Settings
- Title: *Domain Attributes and Data Settings*
- Owner/author: FLIP Fluids / Ryan Guy
- Version: wiki revision visible as edited 2026-05-29 (35 revisions), retrieved 2026-08-22
- Retrieved: 2026-08-22 UTC
- Redistribution: short factual excerpt with attribution; wiki/repository terms retained.

Primary-source excerpt (color section, 20 words):

> If enabled, particles will absorb color attributes from nearby particles. If disabled, particles will hold a static color value

Recorded facts: the product exposes `Color Attributes`, `Enable Mixing`, `Mix Rate`, and `Mixing Mode`; exports surface color as `flip_color` in `[0,1]^3`; describes RGB as additive/not physically accurate and Mixbox as recommended pigment-like mixing requiring a separately installed plugin. Its limitations page says multi-liquid support is currently color blending and variable viscosity, not variable surface tension/density, volumetric attributes, or separated meshes.
