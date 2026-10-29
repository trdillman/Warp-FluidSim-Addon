# Cloud Blender UI integration - 2026-10-15

Imported additively into the canonical FluidSim checkout at
`C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port`,
receiving master commit `296fa95a9c8b1b904123846e9f7470cd99a3d64a`.
No product source was changed. Existing unrelated worktrees were preserved.

## Source and transfer provenance

Owned Library item `libfile_012c8de6ce1c81919da989f687035aed`, backing
`file_00000000b6248211a3de78e18db8dda7`, filename
`cloud-blender-ui-project-bundle.tar.gz`, 746250 bytes, SHA256
`c9d4d45dbc8b238643c506e86384a5089ef08784d196ca8f4ba4b24137158a63`.
Retrieved through Library preparation and its unmodified transfer helper using
existing Ubuntu WSL for extended-attribute support. Archive members were
inspected before extraction: relative regular files/directories only, no links,
absolute paths, traversal, drive paths or duplicate payloads. All 31 payload
SHA256 values were checked before and after copying against the original
[bundle manifest](../bundle-manifest.json), preserved without changes.

Cloud evidence belongs only to PR30 source commit
`295c4cdf291cb96d02b3ded75f9e7eae315aed0a`.
The [source provenance](../artifacts/cloud_ui/20261015/source-provenance.json)
records 733/734 exact Git blob matches; the sole missing cloud file was
`demos/dam_break.blend`. All addon code/assets were present. The 733 recorded
blob IDs were cross-checked locally against that commit's 734-file Git tree.
This integration does not transfer the cloud acceptance claim to master.

## Retained evidence and boundaries

See [cloud results](../artifacts/cloud_ui/20261015/RESULTS.md),
[runtime pins](../tools/cloud_ui/runtime-lock.json),
[foreground readiness](../artifacts/cloud_ui/20261015/foreground-readiness.json),
[profile and cleanup report](../artifacts/cloud_ui/20261015/profile-safe-review.json),
[presentation readback](../artifacts/cloud_ui/20261015/presentation-reopen.json),
and [cloud guard test log](../artifacts/cloud_ui/20261015/launcher-guard-tests.log).
Original cloud paths and retained failed attempts remain unchanged.

Cloud Blender 5.2.2 validated the real sidebar and missing-runtime presentation,
ADVANCED presentation save/reopen, unchanged private profile and owned cleanup,
and four launcher guards. This is no CUDA, performance, Windows GUI,
installed Preferences, demo/playback/export or production acceptance.

Local integration verification is limited to hashes, Git provenance, document
links, Python syntax, Bash syntax and fail-closed guards. WSL guard tests reject
before executable launch; no Blender or GPU job was run on this computer.
No binary, wheels, caches or profiles were imported. No commit, push, merge,
publication or tracker edit was made.

## Local verification receipt

[Integration verification](../artifacts/cloud_ui/20261015/local-integration-verification.json) records the Windows syntax/hash/link
checks and four passing WSL guard tests. `.gitignore` adds exceptions only for
the retained evidence inventory and this receipt; future runtime/profile/cache
outputs remain ignored. No full Linux GUI rerun was performed.
