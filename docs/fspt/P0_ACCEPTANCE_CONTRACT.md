# FSPT P0 acceptance contract

**Contract `fspt-p0-20261127-v1` — frozen engineering acceptance criteria, 27 November 2026.** These requirements define what a later candidate must prove. They do not say that the current implementation meets the targets. The P0 reconciliation and runtime evidence record owns the overall phase disposition.

The machine-readable contract is [`fspt_acceptance_contract_v1.json`](../../tests/render/fixtures/fspt_acceptance_contract_v1.json). It follows the [architecture roadmap, sequence 11](https://chatgpt.com/space/page_e736fb28ec2881918bc688878f62ab03), [live bridge design](https://chatgpt.com/space/page_bfe819484c448191951c75b707b2d507), and [implementation and qualification plan](https://chatgpt.com/space/page_fcaa1c84864081918959ec241a535863). The lead approved the numerical choices in this session before their use on candidate measurements. Changing an accepted threshold requires a new reviewed contract version; changing a report's profile or threshold after failure cannot qualify that result.

## 1. Source and authority

The recovered source checkpoint is `82f08a9b34c0e2971103acb87ed6d0a4cb63bb89` on `experimental/fspt-p1-20261125`, in `C:/Users/Tyler/projects/fluidsim-fspt-20261125`. It includes the separately recovered repair, committed publication identity, the evaluated bridge, reviewed device-copy primitives and an asynchronous CUDA backend. It does not yet establish the full live route or production readiness. The canonical dirty checkout and older P1 evidence remain separate and preserved.

This contract adds no solver ownership to FSPT. Existing admission, playback, output publication, cache validation and Production Bake remain authoritative. No renderer callback may advance the simulation, alter authored quality, or manufacture source identity from a timeline label. ST-FLIP remains removed from project scope.

`tools/run_production_gate.py`, `tools/gate/qualification.py` and `tools/performance_budget.py` remain the qualification machinery. The new pure evaluator, `tools/gate/fspt_contract.py`, can reject invalid trial records and calculate the joint FSPT decision. Its output scope is always `contract_evaluation_only`. It cannot establish that an image was actually rendered on CUDA, that an artifact is authentic, that a package was installed, or that the required independent reviews passed.

## 2. Established FluidSim targets remain intact

The native checkpoint's `tools/performance_budget.py` defines the existing release comparison as follows. The file, reference commits, workload semantics and owned exceptions are unchanged.

| Existing release requirement | Preserved rule |
| --- | --- |
| Workloads and release | `crown_splash`, `dam_break`, `paddle_stir`, `chromatic_vortex`, compared with release 4.0.1 and its pinned commits. |
| Performance | Order-balanced host-time ratio at most 1.05, subject only to the existing explicitly owned changes. |
| Trial schedule | Eight trials per variant in alternating ABBA/BAAB blocks; the existing guarded four-trial early stop and 0.06 margin remain exactly as implemented. |
| Frame range | Frames 1–48, using the existing synchronized solver timer contract. |
| Accuracy | Residual ratio at most 1.10; the existing owned Multigrid rule is 6.0. Existing intended-change and rigid-coupling policies remain unchanged. |
| Fidelity | Existing protected settings, particle populations, source inputs and accuracy checks continue to apply. |
| Release evidence | Storage hooks, reviewed distribution/runtime provenance, exact source, retained raw reports, GUI acceptance and installed-package checks remain mandatory. |

The repository search found no other established absolute FluidSim 24/30/60-FPS acceptance floor in the tracked README, AGENTS, release plans or profiling tools at this checkpoint. This finding does not relax an external or subsequently recovered stricter commitment. FSPT's additional D/A overhead limit is independent: it cannot borrow an allowance intended for a solver algorithm change.

## 3. Frozen workloads and source identity

The primary artist workflow uses the authored, reopenable `dam_break` demo at 24 Hz. Its historical initial population is approximately 256,000 particles; the acceptance run must report the actual count. The smaller supported-material surface-only copy remains an explicitly restricted **reference smoke**, not full material or whitewater qualification.

| Authored scene | Native SHA-256 at the recovered checkpoint |
| --- | --- |
| `demos/dam_break.blend` | `05e1918423e48fdf22ea702ff5ca2db1c8f7464be865619d382ad5d8118b92f8` |
| `demos/crown_splash.blend` | `f3ee4e00dbcbf003fab15733c8494b4f7932bd1e68d21b7ce3dd58af4f6f46d4` |
| `demos/paddle_stir.blend` | `fa0521b10fa6b0e029ab60e0c2ddcb08ff5f8d7cde9c7c536eea8edf7bb022c3` |
| `demos/chromatic_vortex.blend` | `678c3851daa79d3743b5e1cebf050abba81dbf3f20a94b8697f8db40b30eab0c` |

Keep the roadmap's fixed-surface tiers of 250,000, 1,000,000 and 4,000,000 triangles, and live tiers of 100,000, 500,000 and 1,000,000 particles. Record actual counts; a tier fixture must start within 5% of its named size. A live scene may subsequently change population through its frozen authored sources. These tiers are required qualification rows, not measured capacities. An unavailable or infeasible tier is recorded as Not run, Blocked or Failed and cannot disappear from the results. Constructing and measuring the larger tier assets belongs to M6; their generator revision, generated geometry, camera, authored settings and masks must be hashed before measuring a candidate.

Every run retains the source SHA and clean/dirty disposition, environment fingerprint, loaded paths, GPU UUID/driver, actual backend and image transport, fixture SHA, source/owner/epoch, actual solved frame and revision, requested timeline frame, surface generation, image identity and fixed quality settings. Cache loops carry both validated cache identity and loop ID. A live loop never turns a reset or replayed state into a newly solved state.

## 4. Quality presets and the joint Live gate

Every fixed-quality profile uses **1920×1080 native internal and output resolution**, seed 31, exposure 0, gamma 1, Standard/None/sRGB display interpretation, and scene-linear Rec.709 radiance for numerical comparisons. The total path cap is 12 scattering events; transmission events are included in that cap, with a transmission cap of 12. It does not permit 12 reflection events plus another 12 transmission events. Unbiased Russian roulette may begin after event 5 with the correct throughput compensation. Radiance clamping, adaptive sampling, hidden resolution changes and upscaling are disabled.

| Profile | Samples per newly completed image | Ordinary liquid NMAE | Caustic NMAE | Additional criteria |
| --- | ---: | ---: | ---: | --- |
| Live, three independent profiles | 1, 2 or 4 | ≤0.10 | ≤0.20 | Freshness, overhead, memory, analytic optics and temporal gates all pass together. |
| Paused refinement | 64 | ≤0.05 | ≤0.10 | No solver advance; edits invalidate only affected history; optical and feature gates pass. |
| Final | 1,024 | ≤0.05 | ≤0.10 | Exact source/frame; all required content and export controls pass. |

All three modes require at least 95% precision and 95% recall for qualified thin features. The reference profiles disable denoising and temporal reuse. A reconstructed profile may be added only with a separate immutable profile identity, retained raw/history-disabled output and all applicable quality/temporal/freshness checks. No unqualified reconstruction mode inherits the reference profile's pass. The current six-bounce, reduced-resolution live smoke is useful functional evidence and cannot satisfy these performance profiles.

The following **joint** limits apply to the primary authored-24-Hz Live claim. A fast solver with no useful fresh pictures fails.

| Metric | Frozen limit |
| --- | --- |
| Distinct solved states actually presented | At least 24 per wall-clock second. Repeated draws, new samples of the same state and recycled geometry/pixels do not count. |
| Simulated-time throughput | At least 1.0 simulated second per wall-clock second. |
| State-ready-to-presentation age | p95 ≤100 ms and worst observed age ≤250 ms. |
| Total solver interference | D/A completed-step wall-time ratio ≤1.05 **and** p95 completed-step-interval ratio ≤1.05. |
| Combined memory | Peak solver + renderer + scratch + replacement/retired generations + display/history VRAM ≤80% of observed physical VRAM. |
| Interaction | p95 edit-to-first-correct-image ≤100 ms; cancel acknowledgment ≤250 ms; usable controls ≤1 s. GPU drain/release latency is reported separately. |
| Bounded work | At most two geometry slots including retired generations, one in-flight trace quantum per device, and one newest pending descriptor per view. No recurring FSPT capture, tracing or polling when there is no active consumer. |

The separate fixed-geometry feasibility gate is **at least 30 newly traced completed images/s with p95 completion interval ≤33⅓ ms on the one-million-triangle native-1080p fixture**, at a profile that also passes its image quality. The 60-image/s point is reported separately. Fixed-image throughput does not establish live-state throughput.

Use nearest-rank p95: sort N observations and select index `ceil(0.95 × N) − 1`. Averages or enqueue timestamps cannot replace completed-state timings. Presentation receipts name the actual completed-image `view_draw` submission boundary. They are not physical scan-out measurements; original visible captures separately establish that the selected transport reaches the viewport.

For this contract, `state_ready_at` is the authoritative completed solver-state clock (`completed_at`), before any optional FSPT surfacing, capture, copy, acceleration build or tracing. Those later costs remain inside the freshness interval. A separate surface-publication readiness event may be reported for stage attribution, but it must not move the age clock forward. All event timestamps use one declared monotonic clock domain or a recorded calibrated conversion; wall-clock timestamps identify runs only.

## 5. A/B/C/D measurement protocol

| Lane | Required work |
| --- | --- |
| A | Existing FluidSim with FSPT disabled; preserve solver, stateful surface evolution and all required native output/preview obligations. |
| B | A plus the extra final-surface demand from the frozen recorded capture schedule. No FSPT bridge or tracing. If the native outputs already require that surface, B may coincide with A. |
| C | B plus admitted renderer snapshots, translation, copies and acceleration; no tracing, reconstruction or presentation. |
| D | C plus the actual selected renderer and presentation at the frozen profile. |

For each configuration, retain at least three complete paired runs, balanced in order. The fixed repeatable schedule is A-B-C-D, D-C-B-A, A-B-C-D, D-C-B-A: four runs satisfy the at-least-three requirement and give equal forward/reverse ordering. Each lane completes 120 warmup states and at least 600 measured states. Fixed geometry instead requires at least 600 newly traced images. Add a same-profile sustained Live soak of at least 600 seconds; it does not replace the paired trials. M5's 240 changing states and 50 lifecycle cycles remain separate functional gates.

Match authoritative solver settings, surface recipe, required outputs and actual completed solved steps in every lane; match the recorded optional capture demand in B/C/D. Record cold compilation/startup and allocation growth separately. Report `100 × (wall_D / wall_A − 1)`, B−A, C−B and D−C in milliseconds and percentages, actual simulated-time throughput, stage events, allocations, readbacks, skipped states and image age. Marginal wall costs are empirical; overlapping GPU event intervals must not be summed as if serial. Qualify the real adaptive scheduler separately from the matched-demand attribution experiment.

Instrument content fingerprints at the admitted source/capture boundary or through a declared audit pass, never by adding a full-scene hash to every draw. Count the cost of any FSPT-only measurement work in D. Do not hide optional extra surfacing or synchronization in A. The evaluator consumes the instrumented raw proof records; authenticity, clocks and instrumentation overhead still require native review.

## 6. Exact optical and image error definitions

Use the same certified Blender build's converged Cycles output, independent analytic references, and matched geometry/camera/scale/light/material/color interpretation. Start reference rendering at 4,096 samples; double until the change is below one quarter of the applicable **Final** tolerance, up to 65,536. If it has not converged, mark the comparison Blocked. Do not loosen the candidate gate. Retain the converging pair and independent seed groups.

Set reference white luminance to 1.0 scene-linear unit. Luminance is `0.2126 R + 0.7152 G + 0.0722 B`. For reference-defined mask M, normalized mean absolute error is

`mean_M(abs(Lcandidate − Lreference)) / max(mean_M(abs(Lreference)), 0.01)`.

The 0.01 floor is fixed in reference-white units, never estimated from candidate pixels or exposure. Compare unclamped scene-linear radiance before the display transform. RGB radiance must be finite and nonnegative within numerical tolerance; valid infinite no-hit Depth values are not radiance errors.

Masks for ordinary liquid, caustic receiver, thin features and controlled disocclusion are generated from independent reference geometry/visibility and reference output, hashed and frozen before candidate evaluation. All required masks must contain pixels. Caustic regions cannot be reclassified as ordinary or excluded after seeing a failure. For thin features, use reference features at least two pixels wide and at least 0.02 reference-white contrast; allow a one-pixel registration tolerance, then require both precision and recall ≥0.95. Candidate overcoverage cannot compensate for missing features. Retain a matching radiance-error check inside the feature mask as well as the coverage mask.

| Analytic gate | Error definition and frozen bound |
| --- | --- |
| Fresnel | Absolute error in energy reflection probability against the unpolarized dielectric solution, ≤0.005. Cover air↔water and air↔glass over normal, grazing and critical-angle neighborhoods. |
| Slab transmission | Absolute transmitted-energy difference divided by `max(abs(reference transmission), 0.01)`, ≤0.02. Include clear and tinted slabs with thickness and absorption specified in meters/inverse meters. |
| Snell | Absolute transmitted-direction angular error against Snell's law in degrees, ≤0.25°. Reject an invented transmitted ray in total internal reflection. |
| Total internal reflection | Transmitted energy divided by unit incident energy, ≤0.0001. Exercise above-critical directions from the dense medium. |

Use 99% confidence bounds, with half-width at most one quarter of the corresponding permitted error. An upper-bound error must satisfy `estimate + uncertainty ≤ limit`; a coverage rate must satisfy `estimate − uncertainty ≥ limit`. Stochastic optical controls use at least 1,048,576 rays and still must meet the uncertainty criterion. Use exact binomial bounds for sampled reflection/leakage counts and independent-seed confidence estimates for radiance metrics. A point estimate inside the limit is insufficient. Report the statistical procedure and samples; do not reinterpret “no observed leak” as zero uncertainty.

For temporal quality, compare each frame with its matched reference so physical motion is not mistaken for a defect. In the frozen disocclusion mask, history-induced excess residual over the history-disabled reference must not exceed 0.02 reference-white luminance after two newly rendered frames. Camera cuts and invalidating edits clear incompatible history before the next new result. Retained display images keep their original identity and do not qualify as newly rendered frames.

GPU/evaluated geometry comparison uses world-space positions with absolute tolerance `1e-6 m` and zero relative tolerance on the meter-scale fixtures; topology and material mapping are exact after a declared stable correspondence. Normal angular error is at most 0.1°. Required attributes use relative error at most `1e-5` and absolute floor `1e-6`. Exercise translations, reflected/nonuniform transforms, Waves and Sheet Detail, and explicit modifier fallback. This is a fixture error budget, not a claim that untested extreme coordinate scales are qualified.

## 7. Compatibility inventory and required product endpoints

This is an implementation gap inventory. “Required” does not mean “implemented,” and current refusals do not become permanent exclusions.

| Area | Recovered source evidence | Required endpoint |
| --- | --- | --- |
| Bounded static materials | `blender/fspt_scene.py` accepts direct constant-input Diffuse, Emission and zero-roughness Glass; constant Background. | Preserve these controls while expanding required graph semantics. |
| Default FluidSim Liquid | `blender/materials.py` constructs Principled with transmission 1.0, IOR 1.33, `fluidsim_color` through Attribute/RGB/MixRGB and `fluidsim_viscosity` through Attribute/MapRange into roughness. Current FSPT rejects that graph. | Full relevant Principled, attribute fallback, mapped roughness, normals/bump, linked values and animation. The surface-only Glass fixture cannot certify this material. |
| Particles and debug attributes | Principled base color from `fluidsim_color`; diagnostic materials use `display_color`. | Required visible point/particle semantics and evaluated attributes, or a qualified explicit geometry conversion. |
| Whitewater/foam/bubbles/spray | Principled materials use distinct roughness and emission values; typed outputs remain separate owner channels. | Correct visibility, opacity, emission, mixing and species coverage. No silent omission. |
| Liquid interior | P1 has no qualified absorption, requested volumetric scattering or nested-medium tracking. | Tinted depth dependence, rough/opaque liquids, glass-water-air boundaries and required overlapping media. |
| Texture and graph structure | P1 rejects linked shader inputs and unsupported graphs. | Image textures and spaces, UV/mapping, math/color operations, required procedurals, node groups, mixed shaders and animated inputs. |
| Lighting/world | Constant world and emissive mesh subset; Blender light objects are refused. | Point, sun, spot and area lights, HDRI/rotation, required world graphs, unit conventions, shadows and indirect illumination. |
| Geometry/cameras | Evaluated triangles, instances and perspective/orthographic camera foundation. | Required modifiers, attributes/normals, visibility, reflection/nonuniform transforms, camera shift and animation. |
| Final camera effects | Current depth-of-field and motion-blur modes are unqualified. | Both are required Final endpoints. Their absence from fixed-quality throughput presets is not a product exclusion. |
| Blender-specific nodes/settings | No universal compatibility claim. | Inventory each required node and define exact semantics or an artist-selected documented approximation; otherwise fail final rendering explicitly. |

The tracked demo inventory at the source checkpoint is `buoyancy_ladder`, `capillary_droplet`, `chromatic_vortex`, `crown_splash`, `dam_break`, `deep_plunge`, `paddle_stir`, `viscosity_race` and `water_wheel`. The same-build API/scene probe supplies observed saved graph and camera data; the source material inventory above cannot substitute for that native inventory. Preserve the all-demo gate and each demo's effect-reachability requirements.

Required optical fixtures remain clear/tinted slabs and water sphere; water in glass with air bubbles; rough/opaque liquid; frozen splash/thin sheets; animated pour/breakup and disocclusion; a small-light caustic receiver; and whitewater with emissive lighting. Use caustic receiver energy and spatial comparisons even if a later specialized technique is needed. Both independent reviewers use a 0–5 rubric: 0 missing/invalid, 1 major defects, 2 substantial defects, 3 usable with visible material defects, 4 correct with minor noncritical differences, 5 reference-consistent within the frozen numerical and temporal gates. Each of reflection/refraction, absorption, thin detail, caustics, temporal stability and workflow responsiveness must score at least 4, with no critical optical defect.

### Final depth of field and motion blur

The DOF fixture uses a 50 mm lens, f/1.4, 2 m focus distance, 36 mm sensor width and circular aperture, with point/small-emitter targets at 1 m, 2 m and 4 m. Compare the thin-lens circle of confusion and converged same-build Cycles point-spread result. Require absolute diameter error ≤0.5 pixel and relative error ≤5% for reference circles at least two pixels wide.

The motion fixture uses authored 24 Hz, centered shutter 0.5 frames (`−0.25` to `+0.25` frame), 2.4 m/s known motion, and a 10 m-wide orthographic view. Require blur endpoint error ≤0.5 pixel, centroid error ≤0.25 pixel and integrated-energy error ≤2%. Add topology births/deaths in real fluid. Shutter samples must come from owned exact time-sampled/cache geometry through established state policy; naive vertex-index interpolation across remeshing and render-triggered solver stepping are forbidden. Freeze any later motion-vector convention before offering that pass or temporal reconstruction route.

## 8. Frozen export contract

The required initial pass set is **Combined, Alpha, Depth, Normal, Albedo, Emission, Environment, Direct, Indirect and Transmission**.

| Pass | Definition |
| --- | --- |
| Combined | Scene-linear premultiplied RGBA. Before the display transform, RGB equals Emission + Environment + Direct + Indirect. |
| Alpha | Camera coverage in [0,1]. Glass keeps geometric coverage; transparent-film background is 0 and opaque-film background is 1. Preserve material alpha/cutout semantics and antialiasing. |
| Depth | Distance along the normalized primary ray to the first camera-visible surface, in meters; positive infinity for no hit. |
| Normal | First-hit evaluated shading normal in world coordinates, normalized without face-forward flipping; (0,0,0) for no hit. |
| Albedo | First-hit evaluated linear base/scattering color before illumination, including required textures/attributes; zero for no hit. |
| Emission | Directly camera-visible emitting geometry, zero-scatter paths only. |
| Environment | Primary-ray world radiance, zero-scatter paths only. |
| Direct | Radiance paths with exactly one non-null scattering event. |
| Indirect | Radiance paths with at least two non-null scattering events. |
| Transmission | Diagnostic subset of Direct + Indirect whose path includes at least one refraction/BTDF event. It overlaps those passes and must not be added a second time to Combined. |

OpenEXR carries 32-bit scene-linear radiance and named passes with no display transform baked in. PNG carries 16-bit display-referred straight-alpha RGBA through the active Blender color-management transform exactly once. Handle the EXR premultiplied/PNG straight conversion explicitly, including zero-alpha pixels. Reopen through an independent reader; compare scene-linear HDR, checkerboard and colored-background compositing. For the frozen numerical fixtures, Standard/None/sRGB is fixed; normal artist output honors the active configuration and records it.

P5 still requires a complete 240-frame fresh-process sequence with exact names/count/range, no duplicate or skipped source frames, cancel/resume and existing geometry/cache export regressions. Current opaque single-layer Combined still output does not satisfy the full pass/alpha/sequence contract. Required unsupported content must fail final export unless the artist explicitly chose a documented approximation.

## 9. Evidence and executable controls

Run the contract's host controls with:

```text
python -m pytest -q tests/render/test_fspt_acceptance_contract.py
```

Evaluate a retained raw paired Live trial with:

```text
python tools/gate/fspt_contract.py <trial.json>
```

The trial must bind the canonical JSON SHA-256 of the contract and exact profile/fixture, four lane state sequences, warmup counts, matched solver/surface/output and B/C/D demand fingerprints, completed-state clocks, independent completed-image/presentation identities, quality confidence intervals/reference/masks, and measured combined VRAM. The evaluator recomputes throughput, wall and tail overhead, age and memory ratios. Its controls reject missing lanes, absent output, reused source generations, repeated geometry/pixels, wrong epochs, changed timestep/quality, unmatched demand, nonfinite clocks, unconverged or late-frozen references, inadequate confidence and empty masks.

Use the existing `python tools/run_production_gate.py --include-gui --measure-budget` command for mandatory production qualification, with its reviewed distribution, certified bundled Python, locked Warp wheel, private profiles, storage policy and automatic blender-gate routing. Do not turn off gate hooks or change reference/threshold files to obtain a pass. Run one owned GPU/Blender lane at a time and retain every terminal return code and failed attempt.

Every visible milestone retains an original screenshot, raw render, temporal evidence where relevant, exact source/package/runtime/device/settings, reproduction command and receipt. The canonical FSPT image remains Provisional until a real live FluidSim output is qualified and reviewed. A screenshot obscured by startup UI or a different preview pass is diagnostic evidence, not a clean FSPT visual pass. Installed claims require the exact package in a fresh private profile and rerunning every claimed route.

