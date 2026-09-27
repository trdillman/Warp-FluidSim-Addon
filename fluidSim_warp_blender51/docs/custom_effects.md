# Custom Effects

Custom Effects is FluidSim's artist-facing system for adding Warp-driven behavior without exposing solver array plumbing in the normal workflow. It replaces the prototype Custom Warp Snippets panel. The low-level legacy operator IDs remain as migration adapters in the 1.x releases, but new scenes should use **Custom Effects** and **Warp Code**.

## Artist workflow

1. Open **Scene Properties → Custom Effects** and choose **Add Effect**.
2. Search the built-in library by intent, then adjust typed parameters and controller objects.
3. Click **Test Draft**. FluidSim validates the complete effective stack, generates deterministic source, forces Warp JIT compilation, performs bounded guarded launches, synchronizes the selected device, and checks finite output and canaries.
4. Click **Compile & Apply**. FluidSim builds a complete candidate and atomically replaces the active stack only after every enabled effect succeeds.
5. Click **Preview One Frame**, use Live Preview, or begin Production Bake.

The root panel follows the same compact four-part structure as the rest of the
FluidSim Scene Properties workflow: **Effect Stack**, **Selected Effect**,
**Workflow**, and **Status**. **Effect Packages** and **Warp Code** are closed
children so package administration and kernel authoring do not compete with the
artist path. The 3D View FluidSim panel mirrors the same Custom Effects draw
implementation beneath its collapsed **Full Settings** group.

The default library contains Vortex Force, Curl Turbulence, Radial Pulse, Planar Split, Velocity Damping, Temperature Lift, Particle Speed Limit, and Voxel Cooling. Spatial effects use owned Empty controller objects rather than raw world-coordinate constants.

## Demo truth boundary

The packaged Custom Effects `.blend` files under `resources/examples/` are
static validation/review examples. They visualize precomputed numerical
fixtures and package/trust workflows; they are not live simulation demos. Do
not use their geometry, renders, or stored stack metadata as proof that an
effect executes in FluidSolver.

A Custom Effects demo must meet `docs/DEMO_SCENE_CONTRACT.md`: it needs an
authoritative Domain object, owned liquid initialization, a non-empty solver,
an enabled and applied effect, frame-handler playback, changing owned output,
and runtime dispatch/call evidence or a deterministic disabled-effect control.

**Open in Text Editor** prefers an existing Blender Text Editor. When none is
open, it repurposes the current area. Generated source and Go to Error use the
same area policy, preserving the Scene Properties workflow whenever possible.

## Stack semantics

The stack is ordered. Use the arrows to reorder; order participates in the deterministic execution signature. Each row supports:

- **Enabled** — authored participation state.
- **Mute** — temporarily removes the effect from execution and the effective stack identity.
- **Solo** — when any enabled row is soloed, only enabled solo rows execute.
- **Duplicate** — creates a separate instance with a unique stable instance ID.
- **Remove** — removes the scene instance; user-owned controller objects are not deleted implicitly.

Disabled and muted effects are preserved in the scene but do not execute and do not contribute to the effective execution signature. Duplicate instance IDs are invalid and fail closed.

## Draft, tested, and applied state

FluidSim keeps separate draft and applied state.

| Status | Meaning |
|---|---|
| Disabled | The row is present but excluded by enabled/mute/solo filtering. |
| Untrusted | Executable code is disabled until its exact digest is trusted. |
| Draft | The scene contains authored changes not yet tested. |
| Invalid | Validation, compilation, or sandbox execution failed. |
| Tested | The current draft passed real Warp compilation and guarded execution. |
| Applied | The exact tested stack is installed in the live/runtime contract. |
| Dirty | Live Text, parameter, controller, order, cadence, or scope differs from applied state. |
| Runtime Error | An applied kernel failed during execution; the manager is latched failed until a new candidate is applied. |

Controller transforms and Blender Text datablocks are fingerprinted from their current values at every admission, cache, and resume boundary. The UI also computes live drift instead of trusting stale update callbacks.

## Trust boundary

Custom Warp source is trusted local executable code. Trust is bound to a SHA-256 digest of the exact definition and source content.

- Shipped built-ins are trusted only when their content digest matches the bundled inventory.
- Imported packages and embedded custom code are disabled and marked **Untrusted Imported**.
- **Review & Trust** records a local decision for the exact digest.
- Editing trusted source invalidates that decision.
- A package cannot grant trust to itself.
- User packages cannot shadow a shipped built-in effect ID.

Headless execution adds `--custom-code-policy`:

- `deny` — reject every active effect.
- `builtins-only` — execute only exact shipped built-in digests. This is the default.
- `trusted` — allow exact shipped built-ins and locally trusted content.

Policy rejection occurs before compilation or solver allocation and returns process exit code `17`.

## Production Bake and resume

The canonical effect fingerprint contains API/ABI identity, package identity/version, enabled effective order, source, stage, cadence, frame window, field access, typed parameter values, controller snapshots, and expert capability state.

Cache-v2 manifests, frame payloads, checkpoints, bake reports, headless reports, and support reports carry:

```text
effect_signature
effect_inventory
```

Production Bake is blocked when the live draft differs from the applied stack. Resume rejects a mismatch with `EFFECT_SIGNATURE_MISMATCH`. Older cache-v2 data without effect identity remains playable, but it cannot be used as a resume source when active Custom Effects would execute. Cache-v1 remains playback-only.

## Effect packages

`.fseffect` packages are strict deterministic ZIP archives containing `effect.json` and content-hashed source files. Import validates schema/API/ABI versions, exact member inventory, path safety, duplicate names, symlink exclusion, source/definition equality, per-file SHA-256, member count, uncompressed-size limits, total-size limits, and compression ratios before decompression.

Conflict options are:

- **Cancel** — preserve the installed library.
- **Keep Both** — create a new effect ID such as `original-id-copy-2`; changing only the filename is not considered a distinct effect.
- **Replace Installed** — atomically update an installed user effect. Built-in identities cannot be replaced.

Malformed packages remain visible as library diagnostics. **Compare Update** inspects a candidate package without installing it and reports whether it is not installed, identical, modified at the same semantic version, newer, older, ambiguous, or in conflict with a built-in identity. **Rename Package** creates a new validated package identity and display name in the local library while leaving existing embedded scene instances unchanged. The panel can also reveal the extension-managed library folder and remove an installed user package while retaining any embedded scene instance.

## Static validation examples

The release contains:

- `05_custom_effects_vortex.blend`
- `06_custom_effects_stack.blend`
- `07_warp_code_authoring.blend`

Each scene includes applied identity metadata, real supplied-Warp numerical evidence, and an Eevee F12 visualization. They remain static validation/review assets rather than live Domain-backed demos. The authoring example intentionally keeps custom Lateral Wave code disabled and untrusted in the distributed scene even though a separate validation fixture proves that exact source compiles and executes.

## Known limits

The supplied validation host exposes Warp CPU only. CPU proof covers schema, generated source, real JIT compilation, guarded execution, deterministic fixtures, package/trust/cache contracts, Blender UI/save/reopen, and F12 evidence. Full FluidSolver CUDA numerics, CUDA determinism, GPU memory/performance, Windows x64, and NVIDIA-renderer-specific behavior remain external release gates.
