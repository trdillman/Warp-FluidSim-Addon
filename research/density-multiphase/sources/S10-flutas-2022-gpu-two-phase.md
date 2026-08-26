# FluTAS GPU two-phase solver — evidence clip

- Primary source: https://www.diva-portal.org/smash/get/diva2%3A1660324/FULLTEXT01.pdf
- DOI: https://doi.org/10.1016/j.cpc.2022.108602
- Title: *FluTAS: A GPU-Accelerated Finite Difference Code for Multiphase Flows*
- Authors: FluTAS project authors
- Version: Computer Physics Communications 273 (2022), article 108602
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow research summary; paper copyright/license applies.

Narrow primary-source clip: the solver uses OpenACC GPU execution, MTHINC interface transport, and FFT-based pressure solution.

Evidence extracted:

- A bounded volume indicator updates density and viscosity for incompressible two-fluid flow.
- Surface tension, gravity, predictor, Poisson solve, and correction remain on the accelerated path.

Supported claims: a GPU-resident two-phase sequence is practical. The dense/decomposed arrays and FFT pressure architecture do not directly fit a sparse flat leaf table.
