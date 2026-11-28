# FluidSim FSPT qualification evidence, 2026-11-28 UTC

The [qualification status](../../docs/fspt/CURRENT_QUALIFICATION_STATUS.md) and
[restart record](../../CHECKPOINT_AND_RESTART.md) describe the outcome at
each executed commit. These files are small retained records or projections.
The acceptance reports, original screenshots and arrays, Final EXR, and ZIP
payloads remain in their separately retained archives. Hashes quoted for
native receipts refer to their original bytes, including line endings.

| Scope | Retained records |
| --- | --- |
| Exact `ce6afc9a` source and verifier | [Native freeze observations](native-ce6afc9a-observations.json), [independent reviewed verifier provenance](review-ce6afc9a.json), [cloud recovery checkpoint receipt](cloud-checkpoint-receipt.json), [backup upload metadata](dropbox-checkpoint-receipt.json) |
| `ce6afc9a` GUI attempt and remote push | [Broker admission block and verified remote ref projection](successor-gui-admission-and-push.json) |
| `ce6afc9a` provisional package | [Native package receipt](provisional-ce6afc9a-receipt.json), [canonical builder report](provisional-ce6afc9a-builder-report.json), [private Windows CPU smoke report](provisional-ce6afc9a-windows-smoke-report.json), [independent hashes and builder log summary](provisional-ce6afc9a-independent-observations.json) |
| `ce6afc9a` runtime admission | [Native runtime observation terminal status](runtime-ce6afc9a-observation-status.json). The installed FSPT lanes did not start. |
| Historical `5c91c818` component proof | [Native broker, controlled device viewport and worker Final observations](ownership-5c91-observations.json) |
| Historical `0607daad` Windows CPU host and recovery | [Full CPU host receipt projection](host-suite-external-verified-receipt.json), [compact native recovery archive receipt](native-receipt.json) |
| Historical `0607daad` viewport and Live | [Static success and Live soak failure projection](native-gui-0607-projection.json), [soak failure diagnostic projection](live-soak-failure-diagnostic.json), [separate raw early-playback diagnostic](native-080500-trace.json), [derived per-generation trace timings](native-080500-trace-summary.json) |
| Historical `0607daad` provisional package | [Native archive and generic Windows smoke/uninstall receipt](provisional-0607-receipt.json) |

Each receipt applies to its recorded source and execution scope. Both
provisional receipts cover Linux archive inspection and CPU-only generic
Windows smoke. Named installed FSPT GPU acceptance has a separate boundary.
