# Ghost Fluid Method — evidence clip

- Primary source: https://physbam.stanford.edu/~fedkiw/papers/cam1999-31.pdf
- DOI: https://doi.org/10.1006/jcph.1999.6236
- Title: *A Non-Oscillatory Eulerian Approach to Interfaces in Multimaterial Flows (the Ghost Fluid Method)*
- Authors: Ronald P. Fedkiw, Tariq Aslam, Barry Merriman, Stanley Osher
- Version: Journal of Computational Physics 152 (1999)
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow research summary; publisher/authors retain copyright.

Narrow primary-source clip: ghost values impose material-interface discontinuities without numerically smearing the jump.

Evidence extracted:

- Interface conditions are encoded in modified stencil values near a sharp interface.
- The incompressible adaptation uses the pressure jump in the projection stencil; for inviscid surface tension, `[p] = sigma kappa`.

Supported claims: ghost-fluid pressure stencils are preferable to a diffuse body force when a sharp internal liquid interface is required; they complicate sparse neighbor/halo handling.
