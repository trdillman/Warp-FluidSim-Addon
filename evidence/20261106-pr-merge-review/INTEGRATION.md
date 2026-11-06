# Combined PR integration checks

The candidate combines PR41, PR47–52, and PR54–62. PR61 contains the exact
PR47 and PR54–60 heads; PR62 contains PR51 and PR56. Conflicts preserve the
new qualification checks, painted Deck event routing, and display lifecycle reset.

The first combined no-CUDA suite completed with 3435 passes, 349 skips and
two failures. Both failures were old Deck fixtures replacing current_layout
while the modal now reads event_layout. The original negative assertions were
retained; the corrected Deck and gate-coverage suite passed 30 tests.

Independent review found a consumed-input omission in PR61: an analytic field
whose values come from VDB did not include that VDB in cache identity. Twelve
content/transform/reference cases failed before repair, while twelve unused-input
controls passed. The repair follows source-region precedence and pre-schema-3
region interpretation. Expanded coverage includes both field roles, both sampling
cadences, legacy schemas, unused volume controls and point-source precedence.
The final focused identity/preview suite passed 123 tests on Windows with bundled
Blender 5.2 Python and Warp1.19, CUDA hidden. Full integrated gates remain pending.

Before PR61/62 were integrated, candidate99fa7e8b built and inspected both platform
archives under Blender5.2.0 (fbe6228777e7); configured Ruff passed. The Windows
archive SHA256 was3bed9cc77bd563f30f4c62a82144af16997692c58b475a1b72c4d91a458cdd90.
This archive was installed using Blender's extension installer in a task-private
profile. The exact Windows Warp1.19 wheel (134934525 bytes;
bb4865dde4c2ae916d3d2a4e20ed1cd406e1d0a76ae722c3c574a9d43385c4d9) installed,
then staged a forced repair while imported. A separate Blender5.2 process applied
the repair and detected1.19.0 from the extension-owned .local directory. CUDA was
hidden; no user profile was replaced. These checks close that recorded installer
scenario only; they do not qualify the later integrated archive or GPU/GUI lanes.

The ccee3428 no-CUDA source suite completed: 3665 passed, 350 skipped, two
failures in a shared Blender Object double missing hide_get. Adding the real
visibility-read contract resolved both failures; all six affected tests passed.

PR43 integration repairs current display-policy eligibility and restoration,
restores the overlay visibility lease before an export hold, and latches restart
after renderer retirement failure. Preview cleanup failures are aggregated while
solver, scene inputs and CUDA cleanup continue. Focused owner/renderer/display
tests passed 69 cases and replacement-retirement tests passed 97 cases. These
CPU checks do not qualify the native SDK/PRK GPU path on Warp1.19; that runtime
and the final full source/production gate remain outstanding.

Final source follow-up:c6325d9c ran3686passes/350skips with3inventory failures.
Regenerated the228-field ownership map and documented the optional native-GPU
probe; all17affected inventory checks pass at67afc44d. The exact candidate built
both archives with748package checks passing/6skipped and configured lint passing.
Windows ZIP:cfe30227faf8735ba93a56b205fbc76d523e6a0ac5f10afa07887a90354d4fdc;
Linux ZIP:df84a5c5e0963ad05f1ca29330f8a49b619232183bbf5f54d248259fb929f47c.
The Windows ZIP installed in a new isolated Blender5.2 profile, installed the
locked Warp1.19wheel, staged a forced repair, then a fresh process applied it and
verified the extension-owned certified runtime. CUDA was hidden throughout.
This closes the final-archive installer/restart scenario only. Production GPU/GUI,
SDK/PRK and broader installer-negative scenarios remain unqualified.
