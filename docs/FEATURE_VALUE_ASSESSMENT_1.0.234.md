# FluidSim Warp 1.0.234 Feature Value Assessment

Date: 2026-07-11
Candidate source: `C:\Users\Tyler\Downloads\FluidSim_Warp_1.0.233_blender51_STFLIP_surface_resolution_pipeline_extracted_gateway\fluidSim_warp_blender51`

## Identity finding

The directory name says `1.0.233`, but its add-on manifest declares `1.0.234`. Its 68 packaged files are byte-identical to `FluidSim_Warp_1.0.234_blender51_STFLIP_surface_resolution_pipeline_gateway_validated.zip`.

This assessment therefore treats the extracted directory as the authoritative 1.0.234 candidate.

## Executive decision

| Candidate feature | User value | Evidence maturity | Port cost/risk | Recommendation |
|---|---:|---:|---:|---|
| Decoupled surface resolution | Very high | Strong | Medium | Port first |
| Frame/manual surface cadence | High | Strong source/runtime evidence | Low-medium | Port with surface pipeline |
| Active surface filtering | High | Strong functional evidence, limited quality metrics | Low-medium | Port with surface pipeline |
| Dense extraction safety limit | Medium-high | Strong source evidence | Low | Port |
| Separate render-surface cache/invalidation | High | Strong functional evidence | Medium | Port |
| ST-FLIP temporal sampling | Potentially high | Real but experimentally validated | High | Do not production-port yet |
| ST phase field | Medium-high for ST-FLIP | Real activation; weak accuracy evidence | High | Port only inside experimental ST lane |
| Variable-coefficient ST pressure | Potentially high | Real convergence; no reference parity | Very high | Defer pending reference gate |
| Local particle CFL substepping | Medium | Real, but benefit not isolated | Medium | Evaluate as independent cherry-pick |
| Surface particle re-synchronization | Medium for ST-FLIP | Real implementation; no visual/error metric | Medium | Port only with ST-FLIP |
| ST-FLIP UI controls/attribute mode | Low alone | UI plumbing | Low | Port only after solver approval |
| “Gateway” | None as a product feature | Test-scene correction only | None | Do not port as a feature |

Recommended scope now: port the decoupled surface-resolution pipeline, cadence, filters, safety limit, and cache invalidation. Keep full ST-FLIP out of the production solver until it passes accuracy and conservation gates, not only activation/stability gates.

## Truth contract

The candidate is valuable only if:

1. higher surface resolution produces a materially more detailed viewport/export mesh without changing simulation resolution;
2. ST-FLIP permits large time steps with better fidelity or robustness than ordinary FLIP at the same large time step;
3. benchmark/demo artifacts describe behavior actually executed by Blender 5.1 and Warp on the GPU;
4. porting does not discard the current workspace's pressure hot-path optimization, runtime/viewport refactor, or unified whitewater UI.

## Live evidence reproduced

All commands used Blender 5.1, Warp 1.12.1, and `cuda:0` on an NVIDIA RTX 4080 SUPER. The 500k tests ran from a disposable byte-identical copy so the supplied extracted directory was not mutated.

### Core ST-FLIP validation

Result: PASS.

- particles remained valid: 210
- calibrated `m0`: 7.7251 for target 8 particles/cell
- maximum time residual: 0.020540, below 0.020833 bound
- phase reached 0.7528
- pressure converged

This proves that the ST kernels execute, residual carryover is active and bounded, the phase field is populated, and pressure completes. It does not prove improved physical accuracy.

### 500k / 24-frame benchmark

Result: PASS.

| Lane | Substeps | Wall time | Mean measured step | Final particles | Final surface vertices | Pressure |
|---|---:|---:|---:|---:|---:|---|
| FLIP low-CFL reference | 89 | 3.026 s | 69.474 ms | 500,000 | 3,382 | converged |
| FLIP high-CFL stress | 24 | 1.964 s | 29.836 ms | 500,000 | 5,361 | converged |
| ST-FLIP high-CFL | 24 | 1.732 s | 21.851 ms | 500,000 | 3,961 | converged |

The 1.75x wall-time improvement versus the low-CFL reference is mainly explained by reducing 89 substeps to 24. Ordinary high-CFL FLIP also uses 24 substeps and is only about 13% slower than ST-FLIP in this run. The benchmark has no reference-error, conservation-error, trajectory-error, collision-error, or image-quality comparison. It therefore demonstrates throughput and bounded execution, not that ST-FLIP produces a more correct large-step solution.

### 500k / 24-frame surface-resolution pipeline

Result: PASS.

| Surface mode | Voxel size | Vertices | Triangles | Active leaves | Active voxels |
|---|---:|---:|---:|---:|---:|
| 1x simulation grid | 0.100 | 1,835 | 3,503 | 24 | 12,288 |
| 2x render grid | 0.050 | 7,754 | 15,171 | 137 | 70,144 |

The 2x surface generated 4.23x as many vertices and 4.33x as many triangles while preserving comparable bounds. Filter-setting changes invalidated the cached mesh and changed its checksum. The render surface used a separate sparse grid while the simulation grid remained unchanged.

This directly proves the product value: users can buy visible surface detail with surfacing memory/work rather than multiplying the cost of pressure projection, transfers, collisions, and all simulation fields.

The test's reported total wall times are not a reliable isolated surfacing-cost comparison because the 2x lane ran after warm caches and was reported faster than 1x. A dedicated surface-build timer is still required before choosing default multipliers for production scenes.

## Claim matrix

| Claim | Required evidence | Current evidence | Status | Port consequence |
|---|---|---|---|---|
| ST-FLIP code is real | source + runtime | temporal kernels, residual state, 4D P2G weighting, phase field, and ST pressure path execute live | Conformant |
| ST-FLIP is faster than conservative low-CFL FLIP | comparable benchmark | 1.75x wall-time improvement, driven by 73% fewer substeps | Partial | Useful potential, but scenario-specific |
| ST-FLIP is better than ordinary FLIP at the same high CFL | reference/quality/error comparison | only activation, boundedness, retention, finite pressure, and timings | Missing | Do not market as production superiority yet |
| ST pressure is numerically correct | matrix/reference parity + convergence over cases | finite/converged residual in tested case | Partial | Keep experimental until parity gate exists |
| 2x surfacing improves output detail independently of physics | separate grid + denser valid mesh + comparable bounds | reproduced: 4.23x vertices, separate grid, comparable bounds | Conformant | High-value port |
| Surface filters affect output | runtime mesh comparison | cache key changes and checksum changes | Conformant | Port |
| Frame/manual cadence avoids needless substep rebuilds | source behavior + timing | implementation is present; no isolated cadence benchmark | Partial | Port, then profile |
| Gateway is a runtime capability | source/runtime behavior | only a corrected centered acceptance scene and documentation note | Nonconformant as feature | Treat as validation history only |

## Feature analysis

### 1. Decoupled surface-resolution pipeline — port

This is the highest-value feature in the candidate. It solves a common production tradeoff cleanly: simulation resolution controls physics cost, while render resolution controls visible mesh detail. It also preserves one surface source for viewport, cache, and export, reducing disagreement between outputs.

Port boundaries:

- `SurfaceResolutionMode` and `SurfaceUpdateCadence`
- render-surface grid/SDF/temp/extractor ownership
- surface resource keys and invalidation
- custom/multiplier voxel sizing
- leaf-capacity sizing and dense-dimension failure guard
- frame/manual rebuild semantics
- filter application
- scene properties and the surface portion of the UI
- viewport/cache/export consumption of the selected render mesh

Integration warning: adapt this to the current workspace's refactored `scene_bridge`, viewport output path, and unified Whitewater panel. Do not copy the older candidate files wholesale.

### 2. Surface cadence and filtering — port

Building a display mesh once per frame instead of once per simulation substep can remove redundant marching-cubes work during multi-substep frames. Manual/lazy rebuild is valuable for baking or diagnostic workflows. Existing filter controls currently have little value if the solver never consumes them; this candidate makes them functional.

Before selecting production defaults, add timers for particle splat, filtering, marching cubes, and host mesh upload. Keep `FRAME` as the likely default, but let measurement decide whether 2x should be opt-in or default.

### 3. Dense extraction safety and capacity controls — port

These are small, practical protections. A clear failure when the active dense extraction span exceeds the configured maximum is preferable to silent clipping or uncontrolled allocation. Leaf-capacity scale is necessary because halving voxel size can increase active tile demand sharply; the reproduced test grew active leaves from 24 to 137.

### 4. Full ST-FLIP temporal sampling — experimental only

The implementation is substantive, not a label or mock:

- persistent particle time residual
- stochastic time advancement with residual carryover
- adaptive jitter from local CFL
- separable spatial/temporal deposition weights
- phase reconstruction from deposited mass
- optional variable-coefficient projection
- temporary surface re-synchronization

Potential value is high for fast-moving, high-CFL free surfaces where conservative global substepping is expensive. However, the current benchmark only proves that the path runs quickly and stays numerically bounded. It does not establish that the output is closer to a low-step reference than ordinary high-CFL FLIP.

Required approval gates before production port:

1. compare ST-FLIP and ordinary high-CFL FLIP against a low-CFL reference using particle-distribution distance, surface SDF error, volume/mass error, kinetic-energy evolution, and collision penetration;
2. run multiple seeds and report variance because temporal jitter is stochastic;
3. test at least dam break, slosh, thin sheet/splash, moving collider, and source/emission cases;
4. run 100+ frames to expose accumulated residual or conservation drift;
5. validate variable-coefficient matrix application and projection against a CPU/reference formulation on small grids;
6. profile the ST path after integrating the workspace's pressure optimizations.

Until those pass, expose it only as `Experimental ST-FLIP` if ported for research.

### 5. Variable-coefficient ST pressure — defer

This is the highest-risk merge. The current workspace pressure solver has fused updates, recorded Warp launches, parity-specific commands, and bounded CUDA-graph batching. The candidate adds a different matrix/operator and disables the existing coarse multigrid initializer on that path. A direct copy would either bypass the current hot path or require a second optimized command/graph topology.

Port only after the ST accuracy gate demonstrates value. Then integrate the operator behind the existing pressure-solver ownership and benchmark fixed and variable paths separately.

### 6. Local particle CFL substepping — evaluate independently

This may provide collision robustness for long global steps without requiring the entire ST deposition/pressure model. The bundled benchmark does not isolate it, so its standalone value is unknown. A small independent experiment against moving/thin colliders is cheaper and lower risk than porting full ST-FLIP first.

### 7. Surface re-synchronization — conditional

This is conceptually necessary when rendering time-jittered particles, but it has no standalone value in ordinary FLIP. Port it only with experimental ST-FLIP and add an SDF/surface comparison proving that re-synchronization reduces temporal noise or geometry error.

## Proposed port order for approval

1. **Surface pipeline tranche:** decoupled surface grid, resolution modes, cadence, filtering, safety/capacity controls, and current-architecture viewport/cache/export integration.
2. **Independent robustness experiment:** isolate local particle CFL substepping and measure collision error/cost.
3. **ST-FLIP research gate:** add reference-error/conservation tests to the candidate logic before merging it into production files.
4. **Experimental ST tranche, only if gate passes:** temporal sampling, phase field, surface re-synchronization, then variable pressure last.

This order captures the candidate's demonstrated production value without risking the current 40%+ performance gain or presenting an activation benchmark as proof of physical superiority.

## Evidence artifacts

- `artifacts/eval_1.0.234_spatiotemporal_validation.log`
- `artifacts/eval_1.0.234_stflip_benchmark.log`
- `artifacts/eval_1.0.234_surface_resolution.log`
- disposable reproduced JSON under `artifacts/_eval_1.0.234/`
