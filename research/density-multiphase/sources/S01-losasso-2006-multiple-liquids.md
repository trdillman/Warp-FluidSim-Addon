# Multiple Interacting Liquids — evidence clip

- Primary source: https://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf
- DOI: https://doi.org/10.1145/1179352.1141960
- Title: *Multiple Interacting Liquids*
- Authors: Frank Losasso, Tamar Shinar, Andrew Selle, Ronald Fedkiw
- Version: ACM SIGGRAPH 2006 paper
- Retrieved: 2026-08-22 UTC
- Redistribution: bibliographic facts, equations, and a narrow abstract-derived clip for research; paper copyright remains with ACM/authors.

Narrow primary-source clip: “arbitrary numbers of liquid and gas phases” are represented with one particle level set per region.

Evidence extracted:

- A vector of regional level-set values is decoded during advection and projection rather than collapsing material identity to one scalar.
- Ghost values allow discontinuous pressure and velocity derivatives to be applied near interfaces without explicitly storing ghost cells.
- The method addresses differing density, viscosity, surface tension, triple junctions, and topology change.

Supported claims: regional material labels are a proven multi-liquid representation; sharp interface jump treatment is compatible with projection; generalized N-phase handling is substantially more complex than a two-liquid slice.
