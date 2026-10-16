# FluidSim performance comparison — completed, gate fails

All four demos now have eight trials per variant against 3.2.1. Aggregate enforced result: **REGRESSION**. This completes the requested bounded performance comparison; it does not qualify the full 4.0 release.

## Timing results

Candidate/baseline ratios use the unchanged median of within-block ratios, with two ABBA and two BAAB blocks per demo. Frame 1 is warm-up; frames 2–48 retain the established synchronized host/solver boundaries. Physical workload, cork, resolution, tolerances, substeps, solver limits and coupling were not reduced.

| Demo | Prior host / solver | Final host / solver | Ceiling | Margin below ceiling, host / solver (percentage points) | Result |
|---|---:|---:|---:|---:|---|
| crown_splash | 1.2640 / 1.3390 | 1.1650 / 1.2478 | 1.10 | -6.50 / -14.78 | regression |
| dam_break | 1.1816 / 1.2379 | 1.1220 / 1.1886 | 1.10 | -2.20 / -8.86 | inconclusive |
| paddle_stir | 0.9937 / 1.0344 | 1.0085 / 1.0652 | 1.10 | +9.15 / +3.48 | pass |
| chromatic_vortex | 1.0017 / 0.9687 | 1.0122 / 1.0018 | 1.05 | +3.78 / +4.82 | pass |

Crown, Dam and Paddle reports and reduced results are exactly unchanged from the completed earlier sets. Dam remains inconclusive because the ABBA orders disagree. No selective performance reruns were used. Historical-to-current margins include different environmental conditions and are not an isolated estimate of the correction.

The separate eight-run correction ABBA study measured after/before ratios 0.952535 host and 0.948531 solver (4.75% / 5.15% lower time), with identical checkpoint hashes/scalars across all eight runs. It has four trials per variant and is separate from the version-comparison gate.

## Authorized environmental policy

Tyler explicitly approved typical background GPU utilization up to 10% on this work machine. This is an environmental noise allowance, distinct from the 1.10 candidate/baseline performance ceiling. The latter and the 1.05 Vortex ceiling were not changed. The approved decision and implementable protocol are in evidence-vortex-quiet/protocol.json.

NVIDIA WDDM per-process utilization is unavailable (pmon reports dashes; per-process memory N/A). The monitor measures aggregate background only while the owned Blender process is absent: three samples one second apart before each launch, and three after a two-second post-exit settling interval. Prelaunch admission requires three consecutive samples at or below 10%, with a bounded 60-second wait. A post-exit sample above 10% rejects the attempt. During the complete child interval, total utilization and process identities are logged; owned benchmark utilization is never treated as background. Other Blender/GPU-listed compute workers cause conservative rejection. PowerToys/3uTools process presence alone does not.

The fresh Vortex set had 96 background samples: minimum 0%, median 1%, maximum 5%. There were 561 total new observations and no competing-worker detections. All 16 trial receipts passed these boundaries. **This proves the observed boundary readings, not a continuous in-trial 10% background ceiling or per-process attribution.** The result is a bounded quiet-window comparison under the user-approved work-machine policy.

The earlier completed Crown/Dam/Paddle sets retain their original process-presence monitoring protocol; the measured 10% boundary contract applies to the new Vortex set only. This protocol difference is explicit and does not retroactively recertify old observations. The policy predicates passed five controls: 10% inclusive acceptance, over-10% rejection, Blender rejection, compute-worker detection, and no in-trial rejection merely from benchmark-driven total utilization. Mandatory storage checks and pre/post telemetry also ran.

## Accepted and retained evidence

Final included set: 72 trials (8 correction plus 64 matched). Across the two runtime batches, 76 trials completed: 72 final included, 3 uncontaminated partial Vortex trials superseded as an incomplete block, and 1 trial rejected under the original strict process-presence policy. That rejection was not retroactively reversed. All four old Vortex outputs remain untouched in evidence-attempt4; the complete 16-trial replacement is in evidence-vortex-quiet. Earlier prelaunch/harness failures are retained separately.

## Iteration and residual evidence

Every solve is retained in raw reports, including warm-up. These summaries cover frames 2–48. Instrumented deep diagnostic timings were not included.

| Demo / variant | Trials | Solves | Median iterations | Total iterations | Unconverged | Maximum residual |
|---|---:|---:|---:|---:|---:|---:|
| crown_splash/baseline | 8 | 2816 | 25 | 80800 | 0 | 0.004307095 |
| crown_splash/candidate | 8 | 2864 | 5 | 14320 | 0 | 0.0023356812 |
| dam_break/baseline | 8 | 3216 | 50 | 180400 | 0 | 0.0081244912 |
| dam_break/candidate | 8 | 3232 | 10 | 29200 | 0 | 0.0084955321 |
| paddle_stir/baseline | 8 | 1880 | 50 | 77800 | 0 | 0.0092891599 |
| paddle_stir/candidate | 8 | 1872 | 5 | 12520 | 0 | 0.0092515288 |
| chromatic_vortex/baseline | 8 | 4176 | 10 | 41600 | 0 | 0.0010040755 |
| chromatic_vortex/candidate | 8 | 4176 | 10 | 41600 | 0 | 0.0010040755 |

## Identities and delivery

Candidate correction: b29b9684de071dd1c73a2bf209d8a9ec1601797f, fix/performance-budget-20261015. Baseline 3.2.1: 296fa95a9c8b1b904123846e9f7470cd99a3d64a. Candidate runtime files match the repair checkout; frozen inventories and original harness/reducer hashes were verified. Only rigid_coupling.py differs between frozen before/after runtime sources. Exact file manifests are retained in provenance.json.

Runtime: Blender 5.2.0, Warp 1.18.0, RTX 4080 SUPER, CUDA device 0, NVIDIA driver 610.88. Private Blender extensions/config/scripts/cache were used. Original evidence under F:/FluidSim-performance-evidence-20261015 remains preserved. No production code, release policy, installed extension, unrelated process or security setting was changed. No product merge/publication or release-identity change occurred.

The combined raw reports and full unchanged reducer output are evidence-vortex-quiet/combined-runs.json and budget.json. Protocol, full solve records, per-trial monitor receipts, process/telemetry JSONL, source hashes, original blocked report and supervisor scripts are included in the accompanying archive. Full release qualification still requires its separate mandatory checks and owner decisions.
