#!/usr/bin/env python3
"""
Refine and synchronize Fei-Regular.ttf and Fei-Bold.ttf:
Uses Fei-Regular.ttf, Fei-Bold.ttf, and FeiMath-Regular.ttf as the base font family,
ensuring harmonious CJK/Latin optical sizing, synchronized superscripts/subscripts/fullwidth
glyphs, math & geometric symbols, and clean OpenType naming/tables.
"""

import copy
import os
import time
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphCoordinates

FEI_DIR = os.path.dirname(os.path.abspath(__file__))


def is_excluded_codepoint(cp):
    """Exclude non-Chinese alphabets (IPA, Cyrillic, Bopomofo, Enclosed) and Japanese-only characters."""
    if (
        0x0250 <= cp <= 0x036F
        or 0x0400 <= cp <= 0x04FF
        or 0x2460 <= cp <= 0x24FF
        or 0x3100 <= cp <= 0x312F
        or 0x3040 <= cp <= 0x30FF
        or 0x31F0 <= cp <= 0x31FF
        or 0xFF65 <= cp <= 0xFF9F
        or 0x1B000 <= cp <= 0x1B16F
        or 0x32D0 <= cp <= 0x32FE
        or 0x3300 <= cp <= 0x3357
        or 0x337B <= cp <= 0x337F
        or cp in (0x3005, 0x3006, 0x303B, 0x303C, 0x303D)
    ):
        return True
    if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or cp >= 0x20000:
        ch = chr(cp)
        for enc in ("gb2312", "big5"):
            try:
                ch.encode(enc)
                return False
            except Exception:
                pass
        return True
    return False


def transform_simple_glyph(g, glyf_table, sx=1.0, sy=1.0, dx=0.0, dy=0.0, ref_x=0.0, ref_y=0.0):
    """Scale and translate a simple glyph copy cleanly."""
    gg = copy.deepcopy(g)
    if gg.isComposite() or gg.numberOfContours <= 0 or not hasattr(gg, "coordinates") or len(gg.coordinates) == 0:
        return gg
    gg.program = ttProgram.Program()
    coords = np.array(gg.coordinates, dtype=float)
    coords[:, 0] = np.round(ref_x + sx * (coords[:, 0] - ref_x) + dx)
    coords[:, 1] = np.round(ref_y + sy * (coords[:, 1] - ref_y) + dy)
    gg.coordinates = GlyphCoordinates(coords.astype(int).tolist())
    gg.recalcBounds(glyf_table)
    return gg


def combine_simple_glyphs(glyph_shift_list, glyf_table):
    """Combine multiple (simple_glyph, dx, dy) into a single simple glyph."""
    all_pts = []
    all_flags = bytearray()
    all_ends = []
    for g, dx, dy in glyph_shift_list:
        if g.numberOfContours <= 0 or g.isComposite():
            continue
        pts = np.array(g.coordinates, dtype=float)
        pts[:, 0] = np.round(pts[:, 0] + dx)
        pts[:, 1] = np.round(pts[:, 1] + dy)
        base = sum(len(x) for x in all_pts)
        all_pts.append(pts)
        all_flags.extend(g.flags)
        for e in g.endPtsOfContours:
            all_ends.append(base + e)
    out = Glyph()
    if not all_pts:
        out.numberOfContours = 0
        return out
    merged = np.vstack(all_pts)
    out.numberOfContours = len(all_ends)
    out.coordinates = GlyphCoordinates(merged.astype(int).tolist())
    out.flags = all_flags
    out.endPtsOfContours = all_ends
    out.program = ttProgram.Program()
    out.recalcBounds(glyf_table)
    return out


def thicken_simple_glyph(g, glyf_table, offset_units):
    """Apply clean outward normal offset to a FeiMath glyph for Fei-Bold."""
    if g.isComposite() or g.numberOfContours <= 0 or not hasattr(g, "coordinates") or len(g.coordinates) == 0:
        return
    g.program = ttProgram.Program()
    coords = np.array(g.coordinates, dtype=float)
    end_pts = g.endPtsOfContours
    start = 0
    for end in end_pts:
        pts = coords[start : end + 1]
        n = len(pts)
        if n >= 4:
            px, py = pts[:, 0], pts[:, 1]
            area = 0.5 * np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py)
            prev_p = np.roll(pts, 1, axis=0)
            next_p = np.roll(pts, -1, axis=0)
            d = next_p - prev_p
            ld = np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6
            t = d / ld
            normals = np.column_stack([-t[:, 1], t[:, 0]])
            for _ in range(2):
                normals = 0.25 * np.roll(normals, 1, axis=0) + 0.50 * normals + 0.25 * np.roll(normals, -1, axis=0)
            eff = offset_units if area < 0 else (-0.25 * offset_units)
            coords[start : end + 1] = pts + eff * normals
        start = end + 1
    g.coordinates = GlyphCoordinates(np.round(coords).astype(int).tolist())
    g.recalcBounds(glyf_table)


def build_reused_fei_glyphs(fei_base, fei_math, is_bold=False):
    """
    Synchronize codepoints -> (Glyph, (advanceWidth, lsb)) from Fei (Regular/Bold) and FeiMath-Regular
    for ASCII, superscripts, subscripts, fullwidth forms, and mathematical/geometric symbols.
    """
    fcmap = fei_base.getBestCmap()
    fglyf = fei_base["glyf"]
    fhmtx = fei_base["hmtx"]

    mcmap = fei_math.getBestCmap()
    mglyf = fei_math["glyf"]
    mhmtx = fei_math["hmtx"]

    reused = {}

    # 1. Native Fei ASCII (0x0020..0x007E) & curly quotes normalized to cap-height ~630
    h_gn = fcmap[ord("H")]
    h_height = float(fglyf[h_gn].yMax - fglyf[h_gn].yMin)
    en_scale = 630.0 / h_height if abs(h_height - 630.0) > 8.0 and h_height > 0 else 1.0

    for cp in list(range(0x0020, 0x007F)) + [0x2018, 0x2019, 0x201C, 0x201D]:
        if cp in fcmap:
            fgn = fcmap[cp]
            orig_g = fglyf[fgn]
            adv, lsb = fhmtx[fgn]
            if orig_g.numberOfContours > 0 and not orig_g.isComposite():
                gg = transform_simple_glyph(orig_g, fglyf, sx=en_scale, sy=en_scale, ref_x=0.0, ref_y=0.0)
                new_adv = int(round(adv * en_scale))
                reused[cp] = (gg, (new_adv, gg.xMin))
            else:
                gg = copy.deepcopy(orig_g)
                reused[cp] = (gg, (int(round(adv * en_scale)), lsb))

    # 2. Superscripts (¹²³ and ⁰..⁹) & Subscripts (₀..₉) derived from Fei's native digits & symbols
    sup_map = {
        0x00B9: ord("1"),
        0x00B2: ord("2"),
        0x00B3: ord("3"),
        0x2070: ord("0"),
        0x2071: ord("i"),
        0x2074: ord("4"),
        0x2075: ord("5"),
        0x2076: ord("6"),
        0x2077: ord("7"),
        0x2078: ord("8"),
        0x2079: ord("9"),
        0x207A: ord("+"),
        0x207B: ord("-"),
        0x207C: ord("="),
        0x207D: ord("("),
        0x207E: ord(")"),
        0x207F: ord("n"),
    }
    for sup_cp, base_cp in sup_map.items():
        if base_cp in reused:
            base_g, (base_adv, _) = reused[base_cp]
            gg = transform_simple_glyph(base_g, fglyf, sx=0.62, sy=0.62, dx=10.0, dy=290.0, ref_x=0.0, ref_y=0.0)
            reused[sup_cp] = (gg, (int(round(base_adv * 0.64)) + 20, gg.xMin))

    sub_map = {0x2080 + d: ord("0") + d for d in range(10)}
    for sub_cp, base_cp in sub_map.items():
        if base_cp in reused:
            base_g, (base_adv, _) = reused[base_cp]
            gg = transform_simple_glyph(base_g, fglyf, sx=0.62, sy=0.62, dx=10.0, dy=-110.0, ref_x=0.0, ref_y=0.0)
            reused[sub_cp] = (gg, (int(round(base_adv * 0.64)) + 20, gg.xMin))

    # 3. Fullwidth ASCII & punctuation (U+FF01..U+FF5E) derived from Fei's native glyphs
    cjk_adv = 860
    for fw_cp in range(0xFF01, 0xFF5F):
        ascii_cp = fw_cp - 0xFEE0
        if ascii_cp in reused:
            base_g, _ = reused[ascii_cp]
            if base_g.numberOfContours > 0 and not base_g.isComposite():
                w = base_g.xMax - base_g.xMin
                if chr(ascii_cp) in ",.:;!?":
                    target_x = 120.0 - base_g.xMin
                elif chr(ascii_cp) in "([{":
                    target_x = (cjk_adv - 120.0 - w) - base_g.xMin
                elif chr(ascii_cp) in ")]}":
                    target_x = 120.0 - base_g.xMin
                else:
                    target_x = 0.5 * (cjk_adv - w) - base_g.xMin
                gg = transform_simple_glyph(base_g, fglyf, sx=1.0, sy=1.0, dx=target_x, dy=0.0)
                reused[fw_cp] = (gg, (cjk_adv, gg.xMin))

    # 4. Ideographic zero 〇 (U+3007), middle dot ·, ellipsis …, and em-dash —
    if ord("0") in reused:
        g0, _ = reused[ord("0")]
        gg = transform_simple_glyph(g0, fglyf, sx=1.22, sy=1.06, dx=140.0, dy=0.0, ref_x=0.0, ref_y=0.0)
        reused[0x3007] = (gg, (cjk_adv, gg.xMin))

    if ord(".") in reused:
        gdot, _ = reused[ord(".")]
        w_dot = gdot.xMax - gdot.xMin
        dx_mid = 0.5 * (460 - w_dot) - gdot.xMin
        dy_mid = 290.0 - 0.5 * (gdot.yMin + gdot.yMax)
        gg_mid = transform_simple_glyph(gdot, fglyf, dx=dx_mid, dy=dy_mid)
        reused[0x00B7] = (gg_mid, (460, gg_mid.xMin))
        reused[0x2027] = (copy.deepcopy(gg_mid), (460, gg_mid.xMin))
        g_ell = combine_simple_glyphs(
            [(gdot, 90 - gdot.xMin, dy_mid), (gdot, 380 - gdot.xMin, dy_mid), (gdot, 670 - gdot.xMin, dy_mid)],
            fglyf,
        )
        reused[0x2026] = (g_ell, (cjk_adv, g_ell.xMin))

    if ord("-") in reused:
        gdash, _ = reused[ord("-")]
        cx_d = 0.5 * (gdash.xMin + gdash.xMax)
        gg_em = transform_simple_glyph(gdash, fglyf, sx=2.15, sy=1.0, dx=430.0 - cx_d, dy=20.0, ref_x=cx_d, ref_y=0.0)
        reused[0x2014] = (gg_em, (cjk_adv, gg_em.xMin))
        reused[0x2015] = (copy.deepcopy(gg_em), (cjk_adv, gg_em.xMin))
        reused[0x2212] = copy.deepcopy(reused[ord("-")])

    # 5. FeiMath-Regular's Latin-Extended, Greek, Arrows, Letterlike, Math Operators, Geometric Shapes
    for cp, mgn in mcmap.items():
        if cp in reused:
            continue
        if (
            0x00A0 <= cp <= 0x024F
            or 0x0370 <= cp <= 0x03FF
            or 0x2000 <= cp <= 0x23FF
            or 0x25A0 <= cp <= 0x27FF
            or 0x2900 <= cp <= 0x2AFF
        ):
            mg = mglyf[mgn]
            if mg.numberOfContours > 0 and not mg.isComposite():
                gg = copy.deepcopy(mg)
                gg.program = ttProgram.Program()
                if is_bold:
                    thicken_simple_glyph(gg, mglyf, offset_units=9.5)
                gg.recalcBounds(mglyf)
                m_adv, _ = mhmtx[mgn]
                gh = gg.yMax - gg.yMin
                gw = gg.xMax - gg.xMin
                if cp >= 0x2190 and (gh > 640 or gg.yMin < -90):
                    s_fit = min(1.0, 620.0 / max(1.0, gh))
                    cx_m = 0.5 * (gg.xMin + gg.xMax)
                    cy_m = 0.5 * (gg.yMin + gg.yMax)
                    gg = transform_simple_glyph(
                        gg, mglyf, sx=s_fit, sy=s_fit, dx=0.0, dy=285.0 - cy_m, ref_x=cx_m, ref_y=cy_m
                    )
                    gw = gg.xMax - gg.xMin
                if cp >= 0x2190:
                    pad_side = 55
                    dx_s = pad_side - gg.xMin
                    if dx_s != 0:
                        gg = transform_simple_glyph(gg, mglyf, dx=dx_s, dy=0.0)
                    m_adv = max(int(round(m_adv * 0.92)), gw + 2 * pad_side)
                reused[cp] = (gg, (m_adv, gg.xMin))

    return reused


def update_fei_font(font_path, subfamily, full_name, ps_name):
    """Update Fei-Regular.ttf or Fei-Bold.ttf in-place using our own Fei base fonts."""
    t0 = time.time()
    is_bold = subfamily == "Bold"
    font = TTFont(font_path)
    fei_math = TTFont(os.path.join(FEI_DIR, "FeiMath-Regular.ttf"))

    reused_fei_map = build_reused_fei_glyphs(font, fei_math, is_bold=is_bold)

    for tag in ("fpgm", "prep", "cvt ", "DSIG"):
        if tag in font:
            del font[tag]

    cmap = font.getBestCmap()
    glyf = font["glyf"]
    hmtx = font["hmtx"]

    glyph_order = list(font.getGlyphOrder())
    added_glyphs = 0
    for cp, (gg, g_metrics) in reused_fei_map.items():
        if cp not in cmap and not is_excluded_codepoint(cp):
            new_gname = f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:X}"
            if new_gname in glyf:
                new_gname = f"{new_gname}.fei"
            glyph_order.append(new_gname)
            glyf[new_gname] = copy.deepcopy(gg)
            glyf[new_gname].recalcBounds(glyf)
            hmtx[new_gname] = (g_metrics[0], glyf[new_gname].xMin)
            for table in font["cmap"].tables:
                if table.isUnicode():
                    table.cmap[cp] = new_gname
            cmap[cp] = new_gname
            added_glyphs += 1

    if added_glyphs:
        font.setGlyphOrder(glyph_order)
        font["maxp"].numGlyphs = len(glyph_order)

    rev_cmap = {}
    for cp, gn in cmap.items():
        rev_cmap.setdefault(gn, []).append(cp)

    synced_count = 0
    cjk_count = 0
    for gname in font.getGlyphOrder():
        g = glyf[gname]
        if g.numberOfContours > 0:
            g.program = ttProgram.Program()
        cps = rev_cmap.get(gname, [])
        matched_fei_cp = next((c for c in cps if c in reused_fei_map), None)
        if matched_fei_cp is not None:
            gg, g_metrics = reused_fei_map[matched_fei_cp]
            glyf[gname] = copy.deepcopy(gg)
            glyf[gname].recalcBounds(glyf)
            hmtx[gname] = (g_metrics[0], glyf[gname].xMin)
            synced_count += 1
        elif g.numberOfContours > 0:
            g.recalcBounds(glyf)
            adv, _ = hmtx[gname]
            hmtx[gname] = (adv, g.xMin)
            cjk_count += 1

    name_map = {
        1: "Fei",
        2: subfamily,
        3: f"1.000;{ps_name}",
        4: full_name,
        5: "Version 1.000",
        6: ps_name,
        16: "Fei",
        17: subfamily,
    }
    for nrec in font["name"].names:
        if nrec.nameID in name_map:
            font["name"].setName(name_map[nrec.nameID], nrec.nameID, nrec.platformID, nrec.platEncID, nrec.langID)

    font.save(font_path)
    dt = time.time() - t0
    print(
        f"Updated {font_path} in {dt:.2f}s (total_glyphs={len(font.getGlyphOrder()):,}, "
        f"synced_latin_math={synced_count:,}, cjk_glyphs={cjk_count:,})"
    )


if __name__ == "__main__":
    update_fei_font(os.path.join(FEI_DIR, "Fei-Regular.ttf"), "Regular", "Fei Regular", "Fei-Regular")
    update_fei_font(os.path.join(FEI_DIR, "Fei-Bold.ttf"), "Bold", "Fei Bold", "Fei-Bold")
