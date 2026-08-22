# Warp Flow 0.14 demo lifecycle validation checkpoint

This branch is a durable, CI-executable reconstruction of the unfinished demo-validation task.

It contains a deterministic source checkpoint split into Base64 parts because the original corrected 0.13.1 runtime checkpoint was unavailable in the active chat runtime. The source is reconstructed from the surviving uploaded 0.13.0 deliverables plus the user-supplied 0.14.0 merge plan.

The workflow runs two exact Blender AppImage versions (5.1.2 and 5.2.0), installs NVIDIA Warp 1.16, launches a real Warp CPU kernel, and then runs all ten showcase demos through:

- frames 1–50;
- captures at frames 1, 10, 20, 30, 40, and 50;
- pause and same-frame idempotence;
- timeline reset and deterministic replay;
- save at frame 25;
- close/reopen in a new Blender process;
- continued playback to frame 50;
- exact continuous-versus-reopened output comparison.

The workflow uploads per-version screenshots, contact sheets, GIFs, JSON lifecycle records, `.blend` checkpoints, reconstructed source, and SHA-256 manifests.

## Important scope boundary

This checkpoint validates reconstructed solver/showcase behavior. It is **not** represented as the canonical completed 0.14.0 architectural merge because the plan's corrected 0.13.1 compiled-runtime source was not recoverable in this session.
