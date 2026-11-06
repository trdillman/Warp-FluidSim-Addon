# PR47 installer qualification

Owner: Codex, Tyler's requested installer integration. Run: 20261104-pr47.
Source: 707c26e0583928f009a03612b6a2c6b6eb5e158d; tree 85d76ab5a6844b2a9420cc6c1ca2d1863b85057b.
Isolated checkout: C:/workspace/fluidsim-pr47-20261104.
Output root: artifacts/ (private profiles, logs, packages); budget 4 GiB; expiry 2026-11-11 for disposable profiles only. Active lease: this qualification session. Preserve reports, screenshots, failures, package identities and source; no cleanup of foreign files.

1. Inspect current PR/reviews and tracker (sequence 82); preserve canonical dirty checkout.
2. Reproduce native Blender 5.2 Preferences with missing dependencies and a real owned Warp 1.18 installation copied into a private profile. Verify upgrade controls and offline handling without network downloads.
3. Correct demonstrated Preferences defects and certification wording at shared owners; retain version compatibility and lock behavior.
4. Run installer/runtime/packaging regressions and native Preferences acceptance; preserve exact identity and evidence.
5. Compare whitewater baseline evidence separately; never infer Warp 1.19 GPU certification.
6. Update PR47 via reviewed workflow, integrate only after applicable acceptance, and update tracker with bounded results.

Storage task-start check ran in canonical checkout. Its bounded container scan reported incomplete coverage; no cleanup was authorized. Git transport is unavailable; GitHub authenticated connector supplied only three missing blobs. Full candidate tree and signed commit were reconstructed and SHA-verified locally.
