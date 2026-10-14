# 3.0 specs

Read `docs/RELEASE_3_0_PLAN.md` first. Each spec here exists because the item has a
specific way of going wrong that a fresh implementer would not see from the code. The
"Trap this spec prevents" section is the part to read twice.

Landing order (stacked where noted):

1. `SPEC-settings-split-completion.md` — first; every `properties.py` change stacks on it
2. `SPEC-default-unification.md` — stacks on 1
3. `SPEC-cache-v3.md` — independent of 1; before 5 and 6
4. `SPEC-multigrid-pressure.md` — independent; any time
5. `SPEC-transfer-modes-apic.md` — after 3
6. `SPEC-narrow-band-particles.md` — after 3 and 5
7. `SPEC-resolution-divisions.md` — after 1 and 3
8. `SPEC-interchange-export.md` — after 3; VDB part is a time-boxed spike

Not specified separately (plan covers them): Linux certification, REVIEW cards,
ST-FLIP removal.
