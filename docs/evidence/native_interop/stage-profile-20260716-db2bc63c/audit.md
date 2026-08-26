# Independent Audit

## Scope

Read-only review covered the opt-in CUDA timing implementation, no-repack diagnostic, lease/lifetime
ordering, benchmark validity, structured evidence, provenance, and SPEC-20260716T031814Z acceptance
criteria. Code-quality, security/lifetime, and test/spec lanes were reviewed independently.

## First review

The first review blocked acceptance on:

- missing cross-process spread aggregation (10.59% packed, 11.60% no-repack);
- framebuffer equivalence based only on a mean difference;
- an incorrect 100-frame label for 250-frame batches;
- missing executable/hash/process-limit provenance in the Dam Break artifact;
- documentation/checkpoint drift; and
- Dam Break `passed` not rejecting count/mode/host-byte/import-map failures.

No production synchronization, lease-lifetime, handle-ownership, command-execution, secret, or network
security defect was found.

## Repairs

- Added an explicit three-process aggregate with balanced-order, same-candidate/source/limits/count, and
  cross-process coefficient-of-variation gates.
- Replaced the canonical stage set after the original set failed the aggregate gate. Failed and invalid
  artifacts remain preserved and are excluded from authority.
- Added p99 and high-delta concentration limits to the framebuffer comparison.
- Used 500-frame batch labels derived from the configured constant and retained all raw samples.
- Added executable/hash/source/priority/affinity provenance to Dam Break.
- Made Dam Break success depend on 600,000 counts, presentation modes, zero native host-position bytes,
  and six imports/two maps, then reran it.
- Updated the user-facing documentation and canonical Blender checkpoint.

## Final verdict

PASS. All three independent lanes report no remaining blocker or high-severity finding in the scoped
stage-profiling work. The final aggregate passes at 3.60% packed and 4.30% no-repack cross-process
spread. The Dam Break rerun passes every provenance and functional gate. The full consumer Python suite
passes 232/232, syntax checks pass, and both repository `git diff --check` gates pass.

