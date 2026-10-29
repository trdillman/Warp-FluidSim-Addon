# FluidSim 4.0.1 candidate certification result

**Verdict: FAIL — not certified.** Observed 17 October 2026. This result applies to PR #32 head `46cdf3ce67b16a92ea47ad372bcd6e4bba8fe877`, not to an unspecified later candidate. The PR was still open, unmerged and at this head when rechecked at 04:36 UTC.

## Candidate and environment

- Repository: `trdillman/Warp-FluidSim-Addon`; base `454839e233d5676364d90ade106bd6da8d4c62a6`.
- All 741 tracked source blobs were materialized and verified against the exact Git tree; no product source was modified. This source snapshot has no Git database. Generated packaging assets, test scripts and evidence are separate from tracked product bytes.
- Actual foreground runtime: official Blender 5.2.2 LTS, build `d13f752e3b9c`, Linux x86_64, OpenGL llvmpipe LLVM 19.1.7. Private config/scripts/extensions/datafiles/cache directories were used.
- The exact Blender 5.2.0 runtime was not available in this restored workspace. Downloading its vendor checksum returned HTTP 403. No alternate route bypassed that response; fresh exact-5.2.0 cases are BLOCKED, not replaced by 5.2.2 passes.
- Actual locked Linux Warp 1.18.0 wheel SHA-256 `090962bfae7dd22ade0d8e0258f41635e0e58980f15bcc9e7a7d312819dfe440` was verified. The installed runtime reports no CUDA device. GPU simulation, CUDA/OpenGL, performance and full production-bake continuation acceptance remain BLOCKED.

## Confirmed failures

### 1. Installed Preferences does not render its installer controls

The complete Linux package was built and installed into a new private extension repository. All 233 installed package members match the archive. In actual foreground Blender, Edit → Preferences → Add-ons → FluidSim Warp shows version 4.0.1 and **Error (see console)**.

The traceback identifies `fluidSim_warp_blender51/preferences.py:263`:

`box.label(text="Runtime", icon="GPU")`

The actual runtime icon enum rejects `GPU`. This prevents the Preferences body from reaching its installer controls, including the new local-file entry point. It is a shared installed Preferences drawing failure under the tested runtime, not a demo-specific or absent-CUDA explanation. No Windows or fresh 5.2.0 UI result is inferred.

Evidence: `gui/preferences-error.png`, `gui/foreground.log`, `gui/foreground-state.json`, and `gui/installed-package-identity.json` in this run's artifact directory.

### 2. Productive small interrupted downloads exhaust the failure budget

The production `_download_wheel` implementation was executed with controlled transport fault injection. A 2,097,155-byte payload was interrupted after 65,536 bytes of positive progress per attempt. After eight attempts the implementation aborted at 524,288 bytes and removed its partial file, despite making progress each time.

Contrasting cases used the same implementation: 1 MiB productive segments completed successfully in three attempts; zero-progress attempts stopped after eight as expected. This reproduces the unresolved PR review finding at the shared online-download seam. Local-wheel installation bypasses that seam. It is an implementation-level fault-injection result, not a claim that a live network transfer was certified.

Evidence: `probe_download_progress.py`, `download-progress-results.json`, `download-progress.log`. Source condition: `received >= size + _CHUNK_SIZE` at `warp_installer.py:189`; `_CHUNK_SIZE` is 1 MiB. The documented budget is consecutive failures without progress.

## Passed bounded checks

| Check | Result and boundary |
| --- | --- |
| Exact source identity | 741/741 tracked blobs verified before and after tests, zero mismatches |
| Default package build checks | Ruff, compileall and configured add-on tests passed: 656 passed, 2 skipped. Complete platform ZIPs built using Blender 5.2.2. This is not the entire repository suite or a full production gate |
| Linux package identity | `fluidsim_warp-4.0.1-linux_x64.zip`, SHA-256 `ea032672f3970eb4cfc29473d9011c427202295c3d1b5799e3c3043f81c8d64d`; 233 installed files match |
| Installer unit suite | 47 passed, 1 Windows-path-limit skip. These unit checks do not establish visible Preferences usability |
| Rigid cache tests | 12 passed, 1 skipped. The CUDA-dependent resume-restoration regression is skipped, not accepted |
| Actual offline local-wheel operator | Installed headless Blender 5.2.2, online access disabled: Install from File returned FINISHED twice, loaded actual Warp 1.18.0 from extension-owned storage, no CUDA device. This proves the operator path, not access through the broken visible Preferences |
| Owned foreground cleanup | The launched Blender session exited normally; `Blender quit` retained and window inventory confirmed no Blender window remained |

The build tool's `release_ready: true` field is a packaging result only. It does not override this failed certification verdict or missing production/GPU evidence.

## Retained diagnostic attempts

An initial combined test invocation produced 13 fixture setup errors while Warp was unavailable/mocked in that process. After installing the actual locked runtime, the installer and rigid tests were run in separate fresh processes, producing the bounded results above. Initial logs remain retained and are not classified as 13 product regressions.

The first local-wheel probe attempted to assign a read-only online-access property under `--offline-mode`; this was a test-script error. The corrected probe asserts the already-disabled property, then invokes the actual installed operator. Both attempts remain recorded.

## Decision and next actions

1. Do not certify or release this candidate on the strength of passing package/unit checks.
2. If authorized, repair the Preferences enum use and productive-progress retry accounting; bind all new evidence to the resulting new candidate/package.
3. Rerun the failed visible Preferences and controlled download cases, their sibling/error paths, installed lifecycle and local-wheel checks.
4. Obtain the required exact runtime and eligible Linux NVIDIA environment for blocked release gates. Run the actual GPU checkpoint-resume, simulation/rendering/export and performance requirements from `LINUX_UI_CERTIFICATION_PLAN.md`.

No product fixes, commits, review comments, merges or publication were performed in this certification attempt. Remediation approval was requested separately.

## Evidence location

The run is retained under `/workspace/shared/fluidsim-401-certification-20261017/artifacts`. For canonical repository delivery, place the compact evidence under `evidence/certification-4.0.1-20261017/` and retain its hash manifest. Desktop delivery is a separate verified step; this report does not claim it has completed.

Sources: [PR #32](https://github.com/trdillman/Warp-FluidSim-Addon/pull/32), [download review finding](https://github.com/trdillman/Warp-FluidSim-Addon/pull/32#discussion_r4301305336), exact source tree and the retained runtime evidence above.
