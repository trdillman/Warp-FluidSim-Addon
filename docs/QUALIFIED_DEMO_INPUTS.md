# Preserved measured Windows demo inputs

> 3.0 note: these rows describe the 2.1 inputs. The 3.0 Resolution Divisions tree
> rebuilt `crown_splash.blend` and `dam_break.blend` in `fb530b6`. The gate rebuilds
> them on every run and the compressed bytes differ between runs, so the 3.0 release
> commits the blends produced by its qualifying gate run.

This integration adopts the four exact generated inputs used by the frozen
`9947db954abe64d3f0dd6d6feb2ed96866b56a7d` Windows qualification. They were
intentional dirty inputs in that measured checkout, not new simulation results.
The source checkout and historical evidence remain unchanged.

| Repository path | Bytes | SHA256 |
| --- | ---: | --- |
| demos/crown_splash.blend | 266754 | 9133c3be9ca89b64e90abdbc557bec2f3fbf7806fe99bdde7bb459df8ea3148c |
| demos/crown_splash/meta.json | 630 | f7e830348d7aa0c4cd4b52592b01f23ed998bf259888ee782c07afa367552ac6 |
| demos/dam_break.blend | 3305899 | fdd98be79041b6c864e69b2a847e1a8f3c538e1871a4a35cd5cf82c5bdc973fe |
| demos/dam_break/meta.json | 908 | 4c847bf784c70deadc994dba99333256449697f33487b76a13beada1d7e51b35 |

Original binding: `delivery/liquid-9947-windows/final-binding.json` under the
October 2 Liquid workspace. Its `demo_inputs_after_rebuild` rows bind these
exact on-disk bytes. Integration preparation hashes and verifies those rows
before copying only the four named files into the owned development branch.
Git may normalize JSON line endings; qualification comparisons use recorded
on-disk bytes and must check those bytes after any future checkout.

Preservation does not rebind old performance trials or grant this later commit
`FULL-PRODUCTION-GATE-PASS`. The current strict consumer checks report HEAD,
source/demo inventory and raw trial provenance. Follow RELEASING.md for final
release-commit evidence; do not edit historical HEAD/hash values to claim a pass.
