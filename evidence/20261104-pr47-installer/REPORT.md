# PR47 installer qualification â€” 4 November 2026

## Source and ownership

Inspected existing PR47 head `707c26e0583928f009a03612b6a2c6b6eb5e158d`, tree `85d76ab5a6844b2a9420cc6c1ca2d1863b85057b`, against base `80bf4aaae0a108602ef1ef73cdf6c74e30dc8354`. Codex's review had no findings. CodeRabbit withdrew its request to alter quoted application status; the remaining current-installation wording was deferred to the next code push. This work updates both the actual runtime wording and its documentation together.

Tested product correction: local commit `9d63d62` (full SHA in publication receipt). Canonical checkout `29d6bd95bfe3beb3cfe9cc38c82c94f2577b9e83` and its unrelated dirty files were not edited. Work is isolated at `C:/workspace/fluidsim-pr47-20261104`. Git transport failed on SSH and HTTPS; authenticated GitHub reads supplied only three missing source blobs. Reconstructed signed commit and complete tree matched the exact PR identities above.

## Native Blender results

Windows x64; official portable Blender **5.2.0 LTS**, build **fbe6228777e7**; bundled Python 3.13; NVIDIA RTX 4080 SUPER (16 GiB); driver CUDA API **13.3**; owned Warp **1.18.0**, CUDA Toolkit **13.4**. MCP launcher created private extension/config resources and disabled online access. No extension wheel downloads occurred.

| Case | Result | Evidence |
|---|---|---|
| Exact PR47 Preferences before correction | FAIL: native draw raises `TypeError`, icon `GPU` absent in Blender 5.2 | `preferences-before.png`, `native-preferences.txt` |
| Missing Warp after correction | PASS: Preferences draws install controls and offline-file guidance | `preferences-missing.png` |
| Real owned Warp 1.18 after correction | PASS: status `older`, correct owned module path, CUDA device detected, visible **Upgrade to Locked Warp** and **Install from File** | `preferences-owned118-final.png`, `native-owned118.json` |
| Online install in Blender offline mode | PASS: refused before downloading | `native-owned118.json` |
| Missing/older wheel offline | PASS: exact locked-platform filename shown; current 1.18 files retained | `native-offline-rejections.json` |
| Corrupt wheel offline | PASS: rejected | `native-corrupt-wheel.json` |
| Native file picker | PASS: operator returned `RUNNING_MODAL` and opened File Browser; owned test window closed afterwards | `native-file-picker.json` |
| Exact real 1.19 wheel upgrade/staging and fresh-process 1.19 load | **NOT RUN**: no exact Windows 1.19 wheel was found in bounded local/cache searches; download transport unavailable | Remains an installer end-to-end acceptance gap |

The real 1.18 installation was copied into the private extension-owned path; no developer override or fake Warp was used for native detection. The original user's installation was read only. The test Blender process was closed after screenshots and file-picker checks.

## Shared defect and correction

Classification: **shared-path Preferences failure**. `FLUIDSIM_AP_preferences.draw` called `UILayout.label(icon="GPU")` before branching on missing/owned/foreign runtime states. All these Preferences lanes could therefore lose their runtime controls. Production caller is Blender's Add-on Preferences draw in `space_userpref.py`; the missing-Warp and owned-1.18 sibling branches were both checked after replacing the icon with valid `SYSTEM`.

Preserve PR47's `older` / `unsupported` install condition. Label the controls, installer messages, runtime status, README and user guide as **version-locked**, and explicitly show **full GPU qualification pending**. Internal historical `CERTIFIED_*` identifiers and `certified` compatibility tokens remain stable; they do not grant GPU acceptance. No simulation code, defaults, wheel lock, driver floor, replacement staging, ownership boundary or download behavior was changed.

## Regression and package checks

Blender bundled Python with existing Warp 1.18 and CUDA hidden: **762 passed, 2 skipped**, covering all add-on tests, repository/packaging tests and selected runtime/ownership/compatibility tests (`regressions.txt`). Changed product/source-regression files pass Ruff; whitespace passes. Initial system-Python collection and alternate package launches had environment failures; their logs remain local and are not presented as product failures or passes.

The unmodified default package command encountered two pre-existing Ruff findings in unchanged `test_properties_ui_contract.py` and `test_properties_ui_release_source.py`. A separate package run explicitly skipped that lint stage while retaining Python checks and Blender extension validation/build: **PASS**, recorded in `package-bounded.json` and `package-final.txt`. This bounded run cannot be called a clean default package gate. Its generated `release_ready: true` describes build-tool validation only, not release authorization or GPU qualification. Windows archive: 8,324,213 bytes, SHA-256 `f5a957be46a885eb56e24bdbb6db61696f087426a30a6938ee9b64f14bb6a97b`; Linux archive: 8,324,215 bytes, SHA-256 `343e1c8a15253e55e245bef6b586e53389c37966e7f606122fd8a6bc320c9e05`.

## GPU qualification remains separate

`python tools/run_production_gate.py` was attempted with explicit Blender 5.2 paths and the storage hook enabled. It stopped at its existing tracked/ignored-file admission check: six previously tracked evidence `.txt` files from 3 November match ignore rules. No production manifest/pass was issued, and the guard was not bypassed. See `production-gate.txt`. No full 1.19 GPU qualification is claimed.

A focused real Blender **5.2.0 / Warp 1.18 / cuda:0** baseline run independently reproduced:

`blender_whitewater_truth_acceptance.py:82: AssertionError: moving liquid must refresh solver SDF`

See `whitewater118.txt`. Solver/surface/whitewater source is unchanged by this installer correction. This demonstrates the failure exists on 1.18, independently of the 1.19 dependency update. The PR author's reported 1.19 failure remains reported evidence, not a newly rerun paired comparison. Root cause and affected production scope remain **unproven**: caller tracing, contrasting/sibling surface checks and a same-source 1.19 run are still needed. Do not repair the test, solver or defaults within this installer task merely to obtain a green gate.

## Integration boundary and next step

Publish corrections and bounded native evidence to **existing PR47**, retaining its review history. **Do not merge yet**: native installation/staging from the exact real 1.19 Windows wheel and a fresh-process 1.19 detection check are still missing. Obtain that wheel once (134,934,525 bytes; SHA-256 `bb4865dde4c2ae916d3d2a4e20ed1cd406e1d0a76ae722c3c574a9d43385c4d9`), reuse it offline, and finish those checks in a fresh private profile. Then obtain review on the updated head and integrate according to the existing workflow. Resolve full production admission/whitewater qualification separately without claiming 1.19 certified.

No merge or release was performed by this qualification work. Tracker update must distinguish implemented/review-published changes from incomplete native upgrade acceptance and production GPU qualification.
