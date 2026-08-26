# Primary-source clip: Unconditionally Stable MacCormack

- Original: https://doi.org/10.1007/s10915-007-9166-4
- Author PDF: https://faculty.cc.gatech.edu/~jarek/papers/maccormack.pdf
- Title: *An Unconditionally Stable MacCormack Method*
- Authors: Andrew Selle, Ronald Fedkiw, ByungMoon Kim, Yingjie Liu, Jarek Rossignac
- Version: Journal of Scientific Computing 35, 350-371 (2008); author manuscript dated 2007-06-07
- Retrieved: 2026-08-22 UTC
- Redistribution: bibliographic facts and a short quotation only; publisher/author copyright retained.

Primary-source excerpt (paper abstract, 21 words):

> The back and forth error compensation and correction (BFECC) method advects the solution forward and then backward in time.

Recorded facts: the method uses two first-order semi-Lagrangian advections to obtain second-order accuracy, versus three for BFECC. The paper explicitly says correction can create new extrema, motivates limiters/reversion to first order, and warns of poor error estimates where forward/backward traces sample disparate regions such as boundaries.
