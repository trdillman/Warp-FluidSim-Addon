# Balanced-force geometric VOF — evidence clip

- Primary source: https://www.ljll.fr/~frey/papers/interfaces-free_surfaces/Popinet%20S.%2C%20An%20accurate%20adaptive%20solver%20for%20surface-tension%20driven%20interfacial%20flows.pdf
- DOI: https://doi.org/10.1016/j.jcp.2009.04.042
- Title: *An Accurate Adaptive Solver for Surface-Tension-Driven Interfacial Flows*
- Author: Stéphane Popinet
- Version: Journal of Computational Physics 228 (2009)
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow research summary and equations; publisher/author copyright applies.

Primary-source equations recorded:

- `rho(c) = c rho_1 + (1-c) rho_2`; viscosity is interpolated analogously.
- The method combines geometric VOF, height-function curvature, and balanced surface-tension/pressure discretization.

Evidence extracted: the stationary droplet test reaches machine-accuracy equilibrium in the reported balanced formulation.

Supported claims: balanced pressure/surface-tension discretization and a static-droplet test are high-value targets. The paper's quadtree/octree storage is unsuitable for the mandated flat 8x8x8 leaf table, though its numerical ideas remain relevant.
