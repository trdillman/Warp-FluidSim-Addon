# FluidSim Research Workspace

This directory contains research-only inputs for future FluidSim planning. It does not certify implementation, runtime support, or production readiness.

## Lane ownership

- `color-mixing/`: transport and rendering of two or more miscible colored liquids.
- `density-multiphase/`: interacting liquids with different densities and miscibility, including water and oil.
- `state-transitions/`: evaporation/steam, condensation, freezing, melting, and related phase-change coupling.

Each lane owns its report, copied/clipped primary-source evidence, code samples, and provenance manifest. Do not place implementation changes outside the lane directory during this research phase.

## Required lane outputs

Each lane must contain:

1. `REPORT.md`, following `REPORT_TEMPLATE.md` without removing sections.
2. `sources/MANIFEST.md`, listing every local evidence artifact with origin, immutable revision or version, license, retrieval date, SHA-256, and why it matters.
3. `sources/` artifacts: downloaded papers/specifications, clipped official documentation, and/or source-code samples sufficient to substantiate the report.
4. `code/` for small, attributable source snapshots or minimal extracted examples. Do not vendor entire repositories unless narrowly necessary.

Local copies are evidence snapshots, not dependencies approved for redistribution or integration. Preserve source license and notice files beside code samples.

