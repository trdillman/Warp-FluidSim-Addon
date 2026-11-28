# FluidSim FSPT checkpoint and restart

This staged record describes the verified Windows source and the evidence available at 08:37 UTC on November 28, 2026. The documents are separate from the tested checkout.

| Item | Bound state |
| --- | --- |
| Checkout | `C:\Users\Tyler\projects\fluidsim-fspt-20261125`, branch `experimental/fspt-p1-20261125`, draft PR 89 |
| Current commit and tree | `ce6afc9ae8b607a33b5c429e6743a60a86df8b1e`, tree `4f349bc837b8610b35a39f204f038d2db43c3ea4` |
| Current freeze | 884 inputs; source `894057a2db428085028ab3e6388e42caf4bd209d526dfe989472616c98756943`; verifier `b236a2c95e631a040d9958eb506ea859e5b899223bf71d4fd130fcff2646927d` |
| Independent admission JSON | [review-ce6afc9a.json](evidence/fspt-session-20261128/review-ce6afc9a.json), SHA-256 `9c84bc64a0410ca1581057dd934ab8c53288661f79540b0f8fb1bb2c7826ade4` |
| Current native work | 441 targeted Windows CPU render checks passed. Both standalone GUI lanes ended on exclusive admission, exit 75 before product results. The provisional Windows/Linux archives were built and inspected. Two fresh generic Windows CPU smoke processes and uninstall passed. The fresh GPU runtime observer stopped before exclusive admission. Named installed FSPT components did not start. |
| Retained proof | P0/M0 complete. 5c91 verified broker and controlled device/worker Final. 0607 passed full Windows CPU host and standalone 50-cycle viewport, failed its Live soak, and produced provisional archives with tested generic Windows install/uninstall. |
| Normal production qualification | Blocked after two 0607 exclusive contamination stops with unproven PID ownership. Normal performance budget, production manifest, P2/P3 and native Linux runtime remain open. |

[Exact qualifications, historical attempts and original hashes](docs/fspt/CURRENT_QUALIFICATION_STATUS.md) · [Experimental artist quickstart](docs/fspt/EXPERIMENTAL_QUICKSTART.md) · [Session evidence index](evidence/fspt-session-20261128/README.md)

## Restore source and evidence

1. Use the existing Windows worktree or an isolated checkout of the pinned successor. Verify HEAD, tree and the two fingerprints above. Restore from the verified remote branch or a Git source checkpoint if the existing worktree is unavailable. Native push receipt `artifacts/fspt-p0/push-ce6afc9a-082222/receipt.json` SHA-256 `2b292075f1d913b17814374f5db89f62aa3f3687120405abe62af0e7dee8893b` records the exact remote branch tip at 08:22:26 UTC.
2. The 0607 incremental archive `compact-delivery-0607daad9354-20261128T073137386031Z.zip` (841,091 bytes; SHA-256 `78617ab0b918fda645835eac8bb26eb39eb7d3fed4c07e3ade1b5884efa7c91e`) requires commit `82f08a9b34c0e2971103acb87ed6d0a4cb63bb89`. Follow its `RESTORE.md`. It predates ce6afc9a and preserves the 5c91 components and 0607 host proof.
3. The ce6afc9a text mirror and recovery checkpoint `fspt-cloud-checkpoint-20261128T082100Z-ce6afc9a-pending.zip` (22,143,383 bytes; SHA-256 `f4f51ae7263e7439ee20546badf2783f2854229384a23bb46a69728568ab66e5`) includes the reviewed overlay and identifies the Git source required to restore the exact native commit. Verify the pinned Git commit before restoring the native checkout. [Recovery archive metadata](evidence/fspt-session-20261128/dropbox-checkpoint-receipt.json)
4. Keep the [current ce6afc9a provisional package receipt](evidence/fspt-session-20261128/provisional-ce6afc9a-receipt.json) beside its platform ZIPs. Keep the historical 0607 receipts and ZIP hashes separately. The recovery record `fspt-0607-provisional-package-and-receipts.zip` has SHA-256 `f940b98f7ad42c23f0d63fa4cae55cebbfa125917e4781e1efae409ad92c1441`. The scope is standard CPU builder, generic Windows double-smoke/uninstall and Linux archive inspection.
5. Preserve earlier original PNG, NPY/NPZ, EXR, terminal, source and lease files under their original runs. Keep each successor report alongside its own source and release identities.

## Resume qualification

| Pending item | Evidence to retain |
| --- | --- |
| Standalone evaluated GUI | Obtain quiet exclusive admission. Retain exit status, acceptance reports, reviewed source and sampled clean lease, plus original files and hashes. Preserve separate 50-cycle viewport and 240 states/600-second soak outcomes with the 250 ms retained-age and 10-second progress guards. |
| Runtime observation | Retry after legitimate exclusive admission. Retain the exact reviewed distribution, interpreter, Warp origin/version, GPU and driver/toolkit, private profile and sampled clean lease receipt. [Blocked attempt](evidence/fspt-session-20261128/runtime-ce6afc9a-observation-status.json) |
| Named installed FSPT checks | After a verified native runtime observation, use the exact identified provisional Windows ZIP. Retain installed module and wheel hashes, terminal and lifecycle receipts, sampled GPU scope and original-image evidence. Preserve the provisional disposition. |
| Normal full gate | Resolve the scope ambiguity. The supported command remains `python tools/run_production_gate.py --include-gui --measure-budget` with verified distribution, bundled interpreter, pinned Warp, quiet exclusive admission and standard storage proofs. Retain its fresh normal manifest/raw budget if it completes. Strict release packaging retains its existing full-gate precondition. |
| Durable closeout | Verify the exact Git recovery archive, final evidence bundle and transfer hashes. Retain historical receipts if any successor evidence fails or ends early. |
