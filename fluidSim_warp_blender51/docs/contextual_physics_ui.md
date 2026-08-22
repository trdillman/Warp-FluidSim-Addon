# FluidSim UI Surface Contract

FluidSim exposes one settings model through three Blender-native presentation
surfaces:

1. **Scene Properties** is the complete scene-centric hierarchy.
2. **3D View → FluidSim** keeps a compact activity/status surface and mirrors
   the complete hierarchy beneath one collapsed **Full Settings** parent by
   delegating to the same panel draw and poll functions.
3. **Physics Properties → FluidSim** is contextual to the active domain, output,
   or input object.

## State ownership

The presentation surfaces never synchronize copied values. Scene-level controls
edit `Scene.fluidsim` or `Scene.fluidsim_effects`; input controls edit
`Object.fluidsim`. Generated output objects carry only stable identity tags.
Their contextual panels edit the corresponding scene-owned output settings.

## Context identity

- The domain is recognized by identity with `Scene.fluidsim.domain_object`.
- Surface, particle, combined whitewater, foam, bubble, and spray objects are
  recognized by `fluidsim_output_kind`.
- Input objects use the canonical `Object.fluidsim.role`.
- Object names are labels only and are never the authority for contextual UI or
  destructive actions.

## Mutation rules

- Assigning or clearing a domain changes only the scene domain reference; it does
  not delete the object.
- Removing an output requires confirmation and revalidates the exact owned
  object and output tag immediately before deletion.
- Create / Repair Outputs uses the existing idempotent render-node/output
  installation path.
- Input roles continue to use the existing selected-object tagging and clear
  operators.

## Maintenance rule

New Scene panel groups must automatically appear in the 3D View mirror, or add a
focused exclusion with a test and rationale. Contextual Physics panels must reuse
existing draw functions or shared helpers wherever the complete group applies;
they must not introduce a second RNA owner.
