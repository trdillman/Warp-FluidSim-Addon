# First-party manual clip: SideFX SOP FLIP attributes

- Originals: https://www.sidefx.com/docs/houdini/fluid/sopconfigflip.html and https://www.sidefx.com/docs/houdini/nodes/sop/flipcontainer.html
- Titles: *FLIP Configure FLIP!* and *FLIP Container*
- Owner: Side Effects Software Inc.
- Version: live Houdini documentation retrieved 2026-08-22
- Retrieved: 2026-08-22 UTC
- Redistribution: short factual excerpt; SideFX copyright retained.

Primary-source excerpt (workflow page, 19 words):

> transfer UV texture information to a Cd color point attribute ... use the Cd attribute to dye the fluid particles

Recorded facts: the FLIP Container supports custom point-attribute/volume pairs; examples include scalar `temperature` and vector `Cd`. The surface workflow transfers particle `Cd` to the particle-fluid surface. This is a strong product baseline for particle color attributes and surface transfer, but the manual does not claim physically subtractive mixing.
