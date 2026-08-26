# Primary-source clip: conservative advection

- Original paper: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Lentine11.pdf
- Title: *Mass and Momentum Conservation for Fluid Simulation*
- Authors: Michael Lentine, J. T. Grétarsson, Ronald Fedkiw
- Version: ACM SIGGRAPH/Eurographics Symposium on Computer Animation, 2011
- Retrieved: 2026-08-22 UTC
- Redistribution: bibliographic facts and a short quotation only; Eurographics/author copyright retained.

Primary-source excerpt (paper caption/summary, 11 words):

> For the traditional semi-Lagrangian ... cell i, making the scheme fully conservative.

Corroborating modern source: Naoyuki Hirasawa, Takashi Kanai, Ryoichi Ando, *A Flux-Interpolated Advection Scheme for Fluid Simulation*, The Visual Computer (2021), https://nhirasawa.com/publications/flux-advection/ and author manuscript https://nhirasawa.com/flux_main.pdf. That paper reformulates interpolation as cell-face flux so interior-face contributions cancel, improving conservation and numerical diffusion.

Recorded implication: unconditional stability or bounded interpolation does not imply global scalar conservation. A mass-budget test is required for any transported concentration.
