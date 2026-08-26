# Specification clip: OpenPBR interior transmission

- Original: https://academysoftwarefoundation.github.io/OpenPBR/
- Title: *OpenPBR Surface Specification*
- Owner: Academy Software Foundation OpenPBR project
- Version: live specification retrieved 2026-08-22
- Retrieved: 2026-08-22 UTC
- License/redistribution: open specification; narrow equation/facts recorded with attribution.

Primary-source excerpt (interior section, 21 words):

> transmission_depth is the distance traveled inside the medium by white light before its color becomes exactly transmission_color by Beer's law

Recorded equation: the spec derives an RGB extinction coefficient as `mu_t = -ln(T) / transmission_depth`. This supports storing/deriving absorption or extinction coefficients for volumetric appearance instead of treating surface RGB as a physical concentration.
