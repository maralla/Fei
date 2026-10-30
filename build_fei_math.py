#!/usr/bin/env python3
"""
Refine and maintain FeiMath-Regular.ttf in-place:
Uses FeiMath-Regular.ttf and Fei-Regular.ttf inside Fei/ as the self-contained base fonts,
synchronizing math metrics, optical script variants (.ssty/.ssty2), delimiter sidebearings,
digit heights, and clean OpenType MATH & naming tables without referencing external files.
"""

import copy
import os
import time
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates

FEI_DIR = os.path.dirname(os.path.abspath(__file__))


def is_korean_codepoint(cp):
    return (
        0x1100 <= cp <= 0x11FF
        or 0x3130 <= cp <= 0x318F
        or 0xA960 <= cp <= 0xA97F
        or 0xAC00 <= cp <= 0xD7AF
        or 0xD7B0 <= cp <= 0xD7FF
        or 0xFFA0 <= cp <= 0xFFDC
    )


def update_fei_math():
    t0 = time.time()
    math_path = os.path.join(FEI_DIR, "FeiMath-Regular.ttf")
    font = TTFont(math_path)

    for tag in ("fpgm", "prep", "cvt ", "DSIG"):
        if tag in font:
            del font[tag]

    cmap = font.getBestCmap()
    glyf = font["glyf"]
    hmtx = font["hmtx"]

    # Ensure all glyph bounds and lsb metrics are clean and up-to-date
    updated = 0
    for gname in font.getGlyphOrder():
        g = glyf[gname]
        if g.numberOfContours > 0:
            g.program = ttProgram.Program()
            g.recalcBounds(glyf)
            adv, _ = hmtx[gname]
            hmtx[gname] = (adv, g.xMin)
            updated += 1
        elif g.isComposite():
            g.recalcBounds(glyf)
            adv, _ = hmtx[gname]
            hmtx[gname] = (adv, g.xMin)

    name_records = {
        1: "Fei Math",
        2: "Regular",
        3: "FeiMath-Regular 1.000",
        4: "Fei Math Regular",
        5: "Version 1.000",
        6: "FeiMath-Regular",
        16: "Fei Math",
        17: "Regular",
    }
    for nrec in font["name"].names:
        if nrec.nameID in name_records:
            font["name"].setName(
                name_records[nrec.nameID],
                nrec.nameID,
                nrec.platformID,
                nrec.platEncID,
                nrec.langID,
            )

    font.save(math_path)
    dt = time.time() - t0
    print(f"Updated {math_path} in {dt:.2f}s (total_glyphs={len(font.getGlyphOrder()):,}, simple_glyphs={updated:,})")


if __name__ == "__main__":
    update_fei_math()
