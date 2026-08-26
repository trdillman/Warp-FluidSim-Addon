# Sharp interface jumps for incompressible flow — evidence clip

- Primary source: https://physbam.stanford.edu/papers/cam1999-27.pdf
- DOI: https://doi.org/10.1023/A:1011178417620
- Title: *A Boundary Condition Capturing Method for Multiphase Incompressible Flow*
- Authors: Myungjoo Kang, Ronald P. Fedkiw, Xu-Dong Liu
- Version: Journal of Scientific Computing 15 (2000)
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow equations and research summary; publisher/authors retain copyright.

Primary-source interface conditions recorded for no phase change:

- `[u] = 0`
- `[(p I - tau)n] dot n = sigma kappa`
- `[(p I - tau)n] dot t = 0`

Evidence extracted: density used for viscosity and projection must refer to a consistent interface state; derivative jumps belong in the intermediate-velocity construction rather than being duplicated.

Supported claims: variable viscosity changes the pressure/stress jump bookkeeping; `[p]=sigma kappa` is the inviscid or continuously smeared-viscosity simplification.
