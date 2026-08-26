# MultiFLIP — evidence clip

- Primary source: https://www.cs.ubc.ca/~rbridson/docs/boyd_msc.pdf
- Title: *MultiFLIP for Energetic Two-Phase Fluid Simulation*
- Author: Landon Boyd
- Version: University of British Columbia MSc thesis, 2011
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow research notes and equations; copyright remains with the author/UBC.

Narrow primary-source clip: particles are labelled liquid or air and update two colocated grid velocity fields.

Evidence extracted:

- One combined divergence is formed from phase-selected face fluxes, followed by a single pressure projection.
- A particle-derived interface is rebuilt each step; the thesis uses a merged signed field `(phi_liquid - phi_air) / 2`.
- Reported reconstruction radii include `0.36 dx` for ordinary particles and larger escaped-particle radii (1.5 dx in 2-D, 1.1 dx in 3-D).
- A global volume correction is part of the scheme.

Supported claims: persistent per-particle phase labels and one pressure solve are viable; two complete velocity fields carry high storage/bandwidth cost and are not the preferred two-liquid baseline.
