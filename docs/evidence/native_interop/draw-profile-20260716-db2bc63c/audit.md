# Independent Read-Only Audit

Date: 2026-07-16

Result: pass after documentation repair. No remaining critical or high finding.

## Findings

1. High, resolved: an earlier checkpoint paragraph still described the Nsight capture as the next
   action. It now identifies that action as completed and superseded by the final
   `awaiting_user_review` handoff.
2. Wording, resolved: the evidence now distinguishes 22.88 ms/frame CUDA kernel execution from
   23.51 ms/frame total CUDA GPU activity including memcpy and memset.
3. No critical/high diagnostic or runtime defect: the profiler gates process limits, candidate/add-on
   identity, Vulkan, particle/draw counts, imports/maps, host bytes, exact publications, visibility, and
   cleanup.
4. The retain decision is supported: publication is about one percent of synchronized solver wall
   time, while Vulkan draw hardware timing was not captured reliably. Rebuilding Blender or implementing
   deferred acquire submission would be speculative.
5. GUI handoff proof is adequate: temporary add-on identity, candidate hash, process limits, Native GPU
   mode, profile safety, and one publication per frame change are directly evidenced.

The wall-clock redraw aggregates remain rejected and are not used as release-qualified GPU draw timing.
