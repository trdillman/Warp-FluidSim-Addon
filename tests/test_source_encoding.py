"""Add-on source carries no UTF-8-read-as-cp1252 mojibake.

The operator split (b77f4b9) re-encoded session.py, so the capacity dialog
showed "â†’" for "→" and every status line "â€”" for "—".
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The lead byte of a UTF-8 punctuation/arrow sequence (E2) decoded as cp1252,
# followed by a cp1252 continuation glyph.
MOJIBAKE = re.compile("â[€†‚-„™œš]|Ã[\u0080-¿]|Â[ -¿]")


def test_addon_source_has_no_mojibake():
    offenders = []
    for path in sorted((ROOT / "fluidSim_warp_blender51").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MOJIBAKE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
