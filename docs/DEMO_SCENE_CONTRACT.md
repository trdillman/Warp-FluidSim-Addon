# Demo Scene Contract

In this repository, **demo** is a runtime claim, not a synonym for any `.blend`
file, attractive scene, rendered result, or validation artifact. The canonical
shipped demos are the tracked files under `demos/*.blend`. Every one must prove
the real FluidSim workflow it advertises after a fresh reopen.

## Required runtime behavior

A demo scene must satisfy all of the following:

1. **Authoritative domain** — the scene contains a designated Domain object and
   resolves simulation settings from its object-owned `obj.fluidsim_domain`
   data, not a stale Scene mirror.
2. **Owned initialization** — intended visible source objects or explicitly
   owned seed objects recreate the initial liquid state. Reopening or resetting
   at `frame_start` must initialize a non-empty solver without relying on a
   solver session that existed only while the builder was running.
3. **Real playback path** — advancing the timeline invokes the production frame
   handlers and advances the FluidSim solver. Directly keyframed presentation,
   scripted object motion without simulation, or a build-time-only solve does
   not qualify.
4. **Changing owned output** — at least two advanced frames publish visibly
   changing owned particle, surface, or whitewater output. A single preloaded
   frame, saved preview mesh, or static before/after comparison is insufficient.
5. **Self-contained restart** — the tracked scene reopens at `frame_start`, can
   restart from that frame, and does not require an untracked temporary cache,
   hidden foreign seed, stale state from another scene, or machine-specific
   absolute path.
6. **Advertised capability executes** — the named feature must participate in
   the production runtime, not merely appear in UI state, labels, geometry, or
   metadata. Feature-specific acceptance must prove its activation conditions
   and observable effect.
7. **Enabled effects are reachable** — every effect a demo enables must be able
   to act under that demo's own settings. See the section below.

## Enabled effects are reachable

An enabled effect that the scene's own settings can never trigger is a UI-only
claim. It shows up in panels and metadata and changes nothing. Example: a
whitewater churn threshold above the highest churn the scene ever produces.
`tools/check_demo_blends.py` enforces this in two layers:

- **Static preconditions** (`EFFECT_PRECONDITIONS`): each enabled effect must
  have a nonzero driving coefficient. Whitewater needs `ww_churn_gain > 0` and
  `ww_emission_rate > 0`. Viscosity needs per-particle viscosity or
  `viscosity_constant > 0`. Surface tension needs
  `surface_tension_coefficient > 0`. Color mixing needs
  `species_mixing_rate > 0`.
- **Runtime proof for threshold-gated effects** (`WHITEWATER_ACTIVATION_FRAMES`):
  whitewater emits only where the flow's churn reaches `ww_churn_threshold`,
  and churn can only be measured, not bounded statically. Every
  whitewater-enabled demo must declare the window (frames after `frame_start`)
  within which live whitewater appears. The gate reopens the saved scene, plays
  that window through `scene.frame_set` (the production frame handler, with
  keyframed colliders evaluated), and requires a nonzero live count from the
  solver's `frame_stats`. It also requires the solver-applied threshold, rate
  and gain to equal the authored domain values. It reports peak churn against
  the threshold so a failure is diagnosable. A whitewater demo with no declared
  window fails.

Two traps apply to the runtime proof:

- The `FluidSim Whitewater` point cloud never drops below its spare-capacity
  floor, so its point count proves nothing.
- `stat_*` properties publish to `scene.fluidsim`, not the domain group.

Pick thresholds and rates from measured churn on the saved scene, with margin
under the lowest observed peak, since peak churn varies run to run on the GPU.
Record the chosen values in the demo's builder.

For a **Custom Effects demo**, the scene must additionally contain at least one
enabled, applied effect in the scene stack; initialize a real domain and
non-empty solver; execute that effect during solver advancement; and prove the
effect through runtime dispatch/call evidence or a deterministic output
difference from an otherwise equivalent disabled-effect control.

## What is not a demo

The following may be useful artifacts, but they are not demos unless they also
satisfy every requirement above:

- static before/after particle geometry or trajectory lines;
- renders, screenshots, preview meshes, or explanatory camera scenes;
- deterministic CPU fixtures or precomputed numerical comparisons;
- package-authoring, trust-state, or UI-only examples;
- benchmarks, test fixtures, acceptance scenes, and review scenes;
- a `.blend` saved after a builder advanced a transient in-memory solver;
- a scene that requires an external temporary cache merely to show output.

Call these artifacts **examples**, **validation fixtures**, **review scenes**,
or **rendered evidence** according to their actual purpose. Files under
`fluidSim_warp_blender51/resources/examples/` are distributed examples; their
location and appearance do not promote them to canonical demos.

## Storage and naming

- Canonical playable demos live under `demos/` and are inventoried by
  `tools/check_demo_blends.py`.
- Static or workflow-reference assets live under `resources/examples/`, tests,
  or ignored artifact directories and must not use language implying live demo
  compliance.
- A new `.blend` may be promoted into `demos/` only together with its builder,
  all-demo inventory entry, owned-source/seed expectations, and capability-
  specific runtime acceptance.
- Preview images and reports describe a demo; they never certify it.

## Acceptance evidence

`tools/check_demo_blends.py` is the executable all-demo gate. Its inventory must
match every tracked `demos/*.blend`, and it must verify reopen/reset,
authoritative domain ownership, nonzero solver state, and changing owned output
through the real frame path. A capability-specific demo also requires the
focused acceptance lane for that capability. User-facing playback claims
require the saved-demo GUI playback probe or the GUI-inclusive production gate.

Source and runtime evidence outrank scene labels, screenshots, generated
reports, and prior pass receipts. If the executable evidence is absent or
failing, describe the asset more narrowly; do not call it a demo.
