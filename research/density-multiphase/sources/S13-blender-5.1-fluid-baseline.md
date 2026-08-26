# Blender 5.1 liquid baseline — official documentation clip

- Primary sources:
  - https://docs.blender.org/manual/en/5.1/physics/fluid/type/domain/liquid/diffusion.html
  - https://docs.blender.org/api/5.1/bpy.types.FluidDomainSettings.html
  - https://docs.blender.org/api/5.1/bpy.types.FluidFlowSettings.html
- Owner: Blender Foundation
- Version: Blender 5.1 manual and Python API
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow factual notes; Blender documentation license/copyright applies.

Narrow official clip: liquid viscosity and surface tension are domain settings.

Evidence extracted:

- The 5.1 domain API exposes viscosity and surface-tension controls.
- Liquid flow objects expose emission controls but no physical per-liquid density or phase identifier.

Supported claims: stock Blender 5.1 supplies a single-liquid material baseline, not a documented multi-density or immiscible-liquid contract. The absence conclusion is an interface inference, not an explicit Blender limitation statement.
