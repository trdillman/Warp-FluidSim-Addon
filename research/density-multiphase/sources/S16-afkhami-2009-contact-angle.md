# VOF contact-angle treatment — evidence clip

- Primary source: https://web.njit.edu/~shahriar/download/AB_IJNMF09.pdf
- DOI: https://doi.org/10.1002/fld.1974
- Title: *Height Functions for Applying Contact Angles to 3D VOF Simulations*
- Authors: Shahriar Afkhami, Mark Bussmann
- Version: International Journal for Numerical Methods in Fluids (2009)
- Retrieved: 2026-08-22 UTC
- Redistribution: narrow equation and research summary; publisher/authors retain copyright.

Primary-source geometric condition recorded: `n_interface dot n_wall = cos(theta_e)` (orientation convention may reverse the sign).

Evidence extracted:

- Ghost heights impose the requested static contact angle; in the 2-D construction, `h_0 = h_1 + Delta / tan(theta)`.
- The horizontal-height construction is principally suitable away from extreme wetting/non-wetting angles, roughly 45–135 degrees.

Supported claims: static pair-specific contact angle is feasible after a reliable interface geometry channel exists; dynamic contact lines and extreme angles require further modeling.
