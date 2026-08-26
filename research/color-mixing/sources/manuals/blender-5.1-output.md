# First-party manual clip: Blender 5.1 attributes, mesh and volume output

- Originals: https://docs.blender.org/manual/en/5.1/modeling/geometry_nodes/attributes_reference.html ; https://docs.blender.org/manual/en/5.1/render/shader_nodes/shader/volume_principled.html ; https://docs.blender.org/manual/en/5.1/physics/fluid/type/domain/liquid/mesh.html
- Title: Blender 5.1 Reference Manual pages for attributes, Principled Volume, and liquid mesh
- Owner: Blender Foundation / Blender Manual contributors
- Version: Blender 5.1 Manual, last-updated family 2026-06-03
- Retrieved: 2026-08-22 UTC
- License: CC BY-SA 4.0.

Primary-source excerpt (Principled Volume, 12 words):

> Color Attribute: Volume grid for coloring the volume. Use color for smoke simulations.

Recorded facts: Blender attributes are named, typed, and domain-specific; values are interpolated when crossing domains. Principled Volume separately exposes scattering color, absorption color, density, and named volume-grid attributes. The liquid mesh is generated from liquid particles and can use higher resolution than the base simulation. These facts support explicit point/mesh/volume output channels and warn that surface interpolation can alter apparent color without changing simulation state.
