# FLIP Fluids variable-density baseline — official project documentation clip

- Primary sources:
  - https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-World-Settings
  - https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Release-Notes
- Owner: Ryan Guy / FLIP Fluids project
- Version: 1.8.8 documentation and release notes
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow documentation quotation/factual notes; project documentation copyright/license applies.

Narrow official clip: variable density is “not suitable for simulating insoluble/immiscible fluids such as oil and water.”

Evidence extracted:

- Per-source densities can float or sink and may be exported as `flip_density`.
- The pressure solve may fail or require more substeps under variable density.
- Release notes place feature introduction in 1.8.5; 1.8.8 is the documented stable release dated 2026-07-27.

Supported claims: variable-density mixed liquid must not be marketed as immiscible liquid; solver conditioning and substep diagnostics are user-visible requirements.
