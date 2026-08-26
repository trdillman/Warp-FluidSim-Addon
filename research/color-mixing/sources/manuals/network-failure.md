# Retrieval failure and fallback record

- Attempted: direct `Invoke-WebRequest` downloads from `raw.githubusercontent.com` for Taichi v1.7.4 source and license.
- Date: 2026-08-22 UTC.
- Exact failure: `An attempt was made to access a socket in a way forbidden by its access permissions.`
- Retry policy: no repeated identical retry.
- Materially different fallback: opened the official GitHub tag/file pages through the permitted web research service, copied only the exact inspected functions into local excerpts, recorded immutable tags/commits, and placed license notices beside each excerpt.
- Limitation: local artifacts are narrow clips, not byte-identical upstream downloads. Their SHA-256 values prove the locally reviewed evidence bundle only.
