# Specification clip: CSS Color Module Level 4

- Original: https://www.w3.org/TR/2026/CRD-css-color-4-20260618/
- Title: *CSS Color Module Level 4*
- Owner: W3C CSS Working Group
- Version: Candidate Recommendation Draft, 2026-06-18
- Retrieved: 2026-08-22 UTC
- License: W3C permissive document license; attribution retained.

Primary-source excerpt (section 13.1, 20 words):

> physically mixing two colored lights ... CIE XYZ, display-p3-linear or srgb-linear color spaces are appropriate, because they are linear in light intensity

Recorded facts: interpolation results depend on the chosen space; legacy sRGB is neither linear-light nor perceptually uniform and can produce dark/gray mixes. This specification concerns display interpolation, not pigment physics or species conservation.
