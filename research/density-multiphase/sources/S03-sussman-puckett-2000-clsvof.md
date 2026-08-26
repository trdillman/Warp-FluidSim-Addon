# CLSVOF — evidence clip

- Primary source: https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2000/MSS-EGP_2000.pdf
- DOI: https://doi.org/10.1006/jcph.2000.6537
- Title: *A Coupled Level Set and Volume-of-Fluid Method for Computing 3D and Axisymmetric Incompressible Two-Phase Flows*
- Authors: Mark Sussman, Elbridge G. Puckett
- Version: Journal of Computational Physics 162 (2000)
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow research summary and formula names; paper copyright remains with publisher/authors.

Narrow primary-source clip: the method couples VOF conservation with level-set normals and curvature.

Evidence extracted:

- The volume fraction supplies conservative transport while the signed-distance representation supplies interface geometry.
- The paper evaluates surface-tension-dominated motion and severe air/water property ratios.

Supported claims: a CLSVOF-like hybrid is a strong fit when both phase-volume conservation and smooth geometry are required; it costs more fields and synchronization than level set alone.
