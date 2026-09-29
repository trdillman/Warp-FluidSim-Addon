# Dam break "Rocky cove" redesign

Date: 2026-09-29. Branch `feat/dam-break-cove`. Concept page:
https://claude.ai/artifact/QP9D9okDCjfRekdVp1i4oh (concept C, chosen by Tyler).

## Goal

The dam-break demo becomes a scene worth opening in the viewport. A reservoir
held by real dam walls releases at once, runs across a concrete apron and hits
a rocky cove with a sand beach, a sea stack and a timber jetty. Every solid is
a real mesh collider. The demo keeps its benchmark checks (front speed and
Martin & Moses front position) and adds checks for the new terrain.

Tyler's choices: concept C (rocky cove), "instant, with dam walls" release,
"viewport-first" look.

## 1. Layout

x runs downstream. The domain is unchanged:
(0, -1.00146496, 0) to (5.6, 1.00146496, 1.6), voxel 0.03 m, frames 1-120,
72 frames baked.

| Part | Object | Geometry |
|---|---|---|
| Reservoir | `DamBreak Source Column` (source, INITIAL_FILL, ANALYTIC, 8 ppv) | x 0-0.6, y -0.6-0.6, z 0-1.2. About 256k particles (the old column was 0.6 m wide, 128k). |
| Dam | `DamBreak Dam` (collider) | Two abutment boxes, x 0-0.75, \|y\| 0.66-1.00146 (flush with the domain wall), z -0.1-1.4. Two disjoint shells in one mesh. The inner faces stand two voxels clear of the column (see Collider rules). |
| Walls | `DamBreak Walls` (visual only, not a collider) | 0.15 m concrete walls just outside the domain, which is the real wall: one behind the reservoir and training walls along both sides out to where the headlands reach full height. All stand at the dam top (1.4 m) except the near (camera, -y) wall past the dam, which drops to the headland height so the camera sees over it. |
| Apron | `DamBreak Apron` (visual only, not a collider) | Concrete slab x 0-2.4, full width, z -0.04-0. The fluid runs on the domain floor. |
| Cove | `DamBreak Cove` (collider) | Closed heightfield solid over x 2.4-5.6, full width, bottom z -0.1. Top surface described below. Rock and sand material slots. |
| Sea stack | `DamBreak Sea Stack` (collider) | Noisy tapered elliptical prism, semi-axes 0.225 x 0.20 m, centre (3.87, 0.13), 14 sides, rings every 0.1 m, base z -0.1, top about 0.82 m. |
| Jetty | `DamBreak Jetty` (collider) | Deck x 4.2-5.6, y -0.675 to -0.425, z 0.66-0.75 (0.09 m, the 3-voxel minimum). Eight 0.09 m square posts at y -0.47 / -0.63 and x 4.3 / 4.7 / 5.1 / 5.5, from z -0.1 to 3 mm under the deck. |

Cove top surface, h(x, y):

- Beach: toe line x_t = 2.9 - 0.4 y^2. Sand height s = (x - x_t) / 5 (a 1:5
  slope), smoothly capped at 0.5 m (the dune top).
- Headlands: 0.9 m rock for \|y\| >= 0.6 over roughly x 2.5-3.6 (y > 0) and
  2.5-3.9 (y < 0). A steep seaward face and a steep cliff down to \|y\| = 0.5,
  a gentler rear slope. Zero for \|y\| <= 0.5, so the gap is 1.0 m wide.
- h = max(s, r) plus up to 4 cm of integer-hash value noise, weighted to the
  rock. A face is rock where the rock weight is above 0.5, otherwise sand.
- At x = 2.4 the top is at the floor, so the apron runs straight into the cove.

Collider rules, from the solver's SDF path (`k_rasterize_collider_material`):

- Each mesh is closed with outward normals. Shells inside one mesh never
  overlap. Separate collider objects may overlap, since their SDFs combine by
  min.
- Every solid part is at least 3 voxels (0.09 m) thick.
- Each collider costs one BVH query per active voxel per substep, so the scene
  has four colliders, not one per rock.
- No initial fill starts within two voxels of a collider face. The pressure
  solve pins grid faces next to solid cells to the collider's velocity, and
  the FLIP update drags fluid that samples them to rest: with the abutments
  flush against the column, about 1,600 particles hung on their inner faces
  as a visible film for two seconds. Two voxels keep the outer particle row
  out of that stencil (film 0-6 particles). The solver-side fix is tracked
  separately. Grid-aligning the faces (worse) and zero friction (no change)
  did not help.

Frictions (restitution 0): dam 0.2, cove 0.4, sea stack 0.5, jetty 0.3.

## 2. Release and look

- The water starts at rest. The old 0.7 m/s release kick is removed, because
  the dam walls now shape the release.
- Materials use viewport display colors, so the Workbench and Solid views look
  right without a render engine:
  - concrete (0.55, 0.55, 0.53);
  - sand (0.76, 0.66, 0.47);
  - rock (0.33, 0.31, 0.29);
  - timber (0.45, 0.30, 0.18).
- The demo sets its own deeper-blue water material through
  `scene.fluidsim.surface_material` and `particle_material`, and white foam
  through `whitewater_material`.
- Saved viewport: solid shading with color type MATERIAL. A raised
  three-quarter camera at (7.4, -3.1, 3.0), 40 mm lens, looks from the beach
  corner back at the dam, aimed at (2.5, 0, 0.3).
- Preview: Workbench render of frame 24, 640 x 420.
  The build fails if the preview has no water-colored pixels. The old preview
  showed only the sphere, with no water visible.
- Speed: record old and new ms/frame. If the new median step is above 2.5x the
  old median (99.5 ms), narrow the reservoir to 0.9 m. Measured on the
  final build: 70.3 ms median (25.7 s for 72 frames), against 99.5 ms (18.9 s)
  for the old demo with half the particles, so the full 1.2 m reservoir stays.

## 3. Checks and dependents

A new bpy-free module, `tools/dam_break_cove.py`, holds the layout constants,
the height function, the mesh generators and the "depth inside a solid" test.
The builder, the analyzer and the unit tests all import it, so they share one
definition of the geometry.

Analyzer (`tools/analyze_demos.py`, dam_break):

- The front-tracking window ends at `apron_end_x` (2.4, from meta), not at the
  far wall. It covers the monotonic-advance check and the front-speed fit.
- Column collapse keeps its pre-reflection rule. The window is the same
  prefix of frames: it closes at the first frame whose front reaches
  `apron_end_x`. A later frame never re-enters it, since water climbing the
  sand leaves the low band and the front can appear to retreat.
- New checks, with thresholds set from the measured bake with margin:
  - no collider penetration: no particle in any frame is more than 1.5
    voxels (45 mm) inside a collider (measured 15 mm);
  - flow passes the headland gap: at least 10% of the water is past the
    headlands (x > 3.6) at the last frame (measured 24.7%);
  - beach run-up: the wetted sand reaches 0.25 m (measured 0.50, the dune
    top).
- dam_break goes from 9 to 12 checks, and the report from 18/18 to 21/21.

Dependents that change:

- `tools/check_demo_blends.py`: new object names, source box, the four
  colliders (COLLIDER role, closed meshes, frictions), the apron is not a
  collider, and a re-measured whitewater activation frame.
- `tools/run_production_gate.py`: expected marker `DEMO-ANALYSIS-PASS (21/21)`.
- `tests/test_demo_analysis_contract.py`: synthetic meta gains
  `apron_end_x`, and the collapse-window rule is covered.
- `tests/blender_phase3_profile_safe_review.py` and
  `tests/blender_gui_playback_probe.py` need no change: the review adds its own
  probe objects, and `domain_sdf_walls` stays true.
- `tools/profile_gpu_solver.py`: keeps the frozen column preset and relabels
  it "classic dam-break benchmark", so the recorded viscosity numbers stay
  comparable. The two viscosity tools only select it by name.
- Docs: AGENTS.md demo notes and demos/ANALYSIS.md (regenerated). The
  CHANGELOG entry waits for the next version, since the release-identity test
  requires the changelog to lead with the current release. The
  README text block drops its reference to `blender/open_dam_break_playable.py`,
  which does not exist.
- meta.json gains `collider_objects`, `apron_end_x`, `reservoir_width`, and
  drops `collider_object` and the release-velocity fields.

## Acceptance

1. Geometry unit tests pass. They cover:
   - every collider mesh is closed (each edge shared by exactly two faces) with
     outward orientation;
   - every part is at least 3 voxels thick;
   - the build is deterministic;
   - the apron (x < 2.4) is clear;
   - the gap is 1.0 m;
   - no source lattice point starts inside a collider.
2. Building dam_break prints DEMO-BUILD-PASS, and the preview shows water.
   The preview is inspected by eye.
3. `python tools/analyze_demos.py` prints DEMO-ANALYSIS-PASS (21/21).
4. `tools/check_demo_blends.py` prints DEMO-BLEND-CHECK-PASS for all four demos.
5. Full pytest: no new failures against the 1643-pass baseline.
6. The GUI production gate passes twice with the same `demo_artifact_sha256`,
   which becomes the new reference.
7. Old and new ms/frame are recorded.

## Out of scope

The solver, the other three demos, and the gate's own Blender isolation
(tracked separately).
