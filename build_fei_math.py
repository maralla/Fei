#!/usr/bin/env python3
"""
Remodel FeiMath-Regular.ttf into the unified Anime / Cozy Hand-Drawn style:
Uses Fei/FeiMath-Regular.ttf and Fei/Fei-Regular.ttf inside Fei/ as the self-contained base fonts:
1. Rebuilds all accented Latin glyphs (0x00C0..0x017F, including 'ï' in naïve, 'è, û, é' in
   Crème brûlée, 'ç' in façade) directly on our new Anime base letters (including the new
   Anime dotless 'ı' stem) with full-sized, plump Anime diacritics (diaeresis '..' dots ~108u wide!).
2. Remodels Mathematical Italic (𝑎-𝑧, 𝐴-𝑍), Bold, Bold-Italic, Script/Calligraphic (ℋ, ℒ, ℱ, 𝓜),
   Blackboard Bold (𝔹, ℂ, ℝ, ℤ, ℕ, ℚ), Greek (α-ω, Α-Ω), and Math Operators (∑, ∏, ∫, √, ∂, ∇, ∞, etc.)
   with our Anime Capsule Terminal Rounding, Concave-Bay Opening, Separable Anime Cushion, and
   generous 472u x-height / 650u cap-height while preserving OpenType MATH table metrics and
   extensible assembly parts.
3. Updates .ssty and .ssty2 optical script variants to match the remodeled base glyphs.
"""

import copy
import os
import time
import unicodedata
from multiprocessing import Pool, cpu_count
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphCoordinates

FEI_DIR = os.path.dirname(os.path.abspath(__file__))


def smooth_1d_periodic(arr, passes=3):
    out = arr.copy()
    for _ in range(passes):
        out = 0.25 * np.roll(out, 1, axis=0) + 0.50 * out + 0.25 * np.roll(out, -1, axis=0)
    return out


def transform_simple_glyph(g, glyf_table, sx=1.0, sy=1.0, dx=0.0, dy=0.0, ref_x=0.0, ref_y=0.0):
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
    all_pts = []
    all_flags = bytearray()
    all_ends = []
    for g, dx, dy in glyph_shift_list:
        if g is None or g.numberOfContours <= 0 or g.isComposite():
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


def extract_simple_glyph(mg, mglyf):
    if not mg.isComposite():
        if mg.numberOfContours > 0 and hasattr(mg, "coordinates") and len(mg.coordinates) > 0:
            return copy.deepcopy(mg)
        return None
    coords, end_pts, flags = mg.getCoordinates(mglyf)
    if len(coords) == 0 or len(end_pts) == 0:
        return None
    out = Glyph()
    out.numberOfContours = len(end_pts)
    out.coordinates = GlyphCoordinates(coords)
    out.flags = flags
    out.endPtsOfContours = list(end_pts)
    out.program = ttProgram.Program()
    out.recalcBounds(mglyf)
    return out


def extract_contour_subset_glyph(g, contour_indices, glyf_table):
    """Extract specific contours from a simple glyph into a new simple Glyph."""
    coords = np.array(g.coordinates, dtype=float)
    flags = g.flags
    ends = g.endPtsOfContours
    out_pts = []
    out_flags = bytearray()
    out_ends = []
    start = 0
    for idx, end in enumerate(ends):
        if idx in contour_indices:
            c_pts = coords[start : end + 1]
            c_fl = flags[start : end + 1]
            base = sum(len(x) for x in out_pts)
            out_pts.append(c_pts)
            out_flags.extend(c_fl)
            out_ends.append(base + len(c_pts) - 1)
        start = end + 1
    out = Glyph()
    if not out_pts:
        out.numberOfContours = 0
        return out
    merged = np.vstack(out_pts)
    out.numberOfContours = len(out_ends)
    out.coordinates = GlyphCoordinates(np.round(merged).astype(int).tolist())
    out.flags = out_flags
    out.endPtsOfContours = out_ends
    out.program = ttProgram.Program()
    out.recalcBounds(glyf_table)
    return out


def anime_round_and_plump_contour(coords, flags, end_pts, plump_offset=2.2, round_passes=4):
    """
    Apply Anime Capsule Terminal & Exterior Corner Rounding + gentle felt-marker plumpness
    directly on a math/symbol/alphanumeric glyph's TrueType contours.
    """
    out_coords = coords.copy()
    out_flags = bytearray(flags)
    total_pts = len(out_coords)
    idx_all = np.arange(total_pts)
    diff_x = out_coords[None, :, 0] - out_coords[:, None, 0]
    diff_y = out_coords[None, :, 1] - out_coords[:, None, 1]
    dist_sq = diff_x * diff_x + diff_y * diff_y
    dist_all = np.sqrt(dist_sq) + 1e-6

    all_normals = np.zeros_like(out_coords)
    contours_meta = []
    start = 0
    for end in end_pts:
        pts = out_coords[start : end + 1]
        n = len(pts)
        px, py = pts[:, 0], pts[:, 1]
        area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        contours_meta.append((start, end, area))
        if n >= 5:
            d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
            t = d / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            normals = smooth_1d_periodic(np.column_stack([-t[:, 1], t[:, 0]]), passes=2)
            normals = normals / (np.hypot(normals[:, 0], normals[:, 1])[:, None] + 1e-6)
            all_normals[start : end + 1] = normals
        start = end + 1

    for start, end, area in contours_meta:
        pts = out_coords[start : end + 1].copy()
        fl = out_flags[start : end + 1]
        n = len(pts)
        if n >= 6 and area < 0:
            normals = all_normals[start : end + 1]
            v1 = pts - np.roll(pts, 1, axis=0)
            v2 = np.roll(pts, -1, axis=0) - pts
            l1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-6
            l2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-6
            u1, u2 = v1 / l1[:, None], v2 / l2[:, None]
            cz = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
            cos_a = np.sum(u1 * u2, axis=1)

            # 1. Inflate narrow needle tips into plump round anime capsule terminals
            pts_new = pts.copy()
            target_w = 48.0
            for i in range(n):
                if cz[i] < -0.06 and cos_a[i] < 0.28 and max(l1[i], l2[i]) < 120.0:
                    p_m1 = pts[(i - 1) % n]
                    p_p1 = pts[(i + 1) % n]
                    cross_vec = p_p1 - p_m1
                    cw = float(np.hypot(cross_vec[0], cross_vec[1]))
                    if 4.0 < cw < target_w:
                        u_cross = cross_vec / cw
                        expand = min(6.5, 0.36 * (target_w - cw))
                        if cz[(i - 1) % n] <= 0.04:
                            pts_new[(i - 1) % n] -= expand * u_cross
                        if cz[(i + 1) % n] <= 0.04:
                            pts_new[(i + 1) % n] += expand * u_cross
                    mid_nb = 0.5 * (pts_new[(i - 1) % n] + pts_new[(i + 1) % n])
                    pts_new[i] = 0.55 * pts[i] + 0.45 * mid_nb
                    if (fl[(i - 1) % n] & 1) == 1 and (fl[(i + 1) % n] & 1) == 1:
                        fl[i] = 0
            pts = pts_new

            # 2. Gentle outward plumpness guarded by opposing-wall distance
            if abs(plump_offset) > 0.1:
                c_rows = np.arange(start, end + 1)
                same_c = (idx_all[None, :] >= start) & (idx_all[None, :] <= end)
                c_pos = c_rows[:, None] - start
                j_pos = idx_all[None, :] - start
                cyc_d = np.minimum(np.abs(c_pos - j_pos), n - np.abs(c_pos - j_pos))
                ignore = same_c & (cyc_d <= 2)

                dx_sub = diff_x[c_rows, :]
                dy_sub = diff_y[c_rows, :]
                d2_sub = np.where(ignore, 1e12, dist_sq[c_rows, :])
                dist_sub = np.sqrt(d2_sub) + 1e-6
                dir_cos = (dx_sub * normals[:, 0:1] + dy_sub * normals[:, 1:2]) / dist_sub
                norm_dot = normals[:, 0:1] * all_normals[None, :, 0] + normals[:, 1:2] * all_normals[None, :, 1]
                is_opp = (dir_cos > 0.45) & (norm_dot < -0.30)
                ahead_dist = np.where(is_opp, dist_sub, 1e6).min(axis=1)
                max_allow = smooth_1d_periodic(np.maximum(0.0, 0.34 * (ahead_dist - 18.0)), passes=2)

                c_guard = smooth_1d_periodic(np.clip((0.50 - cz) / 0.25, 0.0, 1.0), passes=2)
                off_amt = smooth_1d_periodic(np.clip(plump_offset * c_guard, -2.0, max_allow), passes=2)
                pts = pts + off_amt[:, None] * normals

            # 3. Unit-Bisector Exterior Convex Corner Filleting
            for _ in range(round_passes):
                prev_p = np.roll(pts, 1, axis=0)
                next_p = np.roll(pts, -1, axis=0)
                v1 = pts - prev_p
                v2 = next_p - pts
                l1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-6
                l2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-6
                u1, u2 = v1 / l1[:, None], v2 / l2[:, None]
                cz = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
                cos_a = np.sum(u1 * u2, axis=1)

                bisec = -u1 + u2
                b_len = np.hypot(bisec[:, 0], bisec[:, 1]) + 1e-6
                u_bisec = bisec / b_len[:, None]

                is_convex = (cz < -0.04) & (cos_a < 0.78)
                max_d = np.minimum(11.0, 0.30 * np.minimum(l1, l2))
                sharpness = np.clip((0.78 - cos_a) / 1.05, 0.0, 1.0)
                d_move = np.where(is_convex, max_d * sharpness, 0.0)[:, None]
                pts = pts + d_move * u_bisec

            out_coords[start : end + 1] = pts
            out_flags[start : end + 1] = fl
    return out_coords, out_flags


def measure_math_glyph_stroke_units(coords, flags, ends):
    """Measure 85th-percentile stroke thickness of a glyph in font units (1000 UPM) via 250px raster EDT."""
    scale = 0.25
    pad = 24
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    W = int(np.ceil((x1 - x0) * scale)) + 2 * pad
    H = int(np.ceil((y1 - y0) * scale)) + 2 * pad
    if W < 10 or H < 10:
        return 55.0
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt

    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    s = 0
    polys = []
    for e in ends:
        c_pts = coords[s : e + 1]
        px, py = c_pts[:, 0], c_pts[:, 1]
        area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        pts_img = [(float((p[0] - x0) * scale + pad), float((y1 - p[1]) * scale + pad)) for p in c_pts]
        polys.append((area, pts_img))
        s = e + 1
    if not polys:
        return 55.0
    outer_sign = 1.0 if max(polys, key=lambda x: abs(x[0]))[0] >= 0 else -1.0
    # Draw outer contours (same sign as largest contour) first in 255, then inner holes in 0!
    for area, pts_img in sorted(polys, key=lambda x: -abs(x[0])):
        if len(pts_img) >= 3:
            is_outer = (area * outer_sign) >= 0
            d.polygon(pts_img, fill=255 if is_outer else 0)
    arr = np.array(im) > 128
    if not arr.any():
        return 55.0
    dt = distance_transform_edt(arr)
    return float(np.percentile(dt[arr], 85) * 2.0 / scale)


def adjust_math_glyph_stroke_thickness(coords, flags, end_pts, delta_w):
    """
    Universal winding-invariant stroke thickness adjustment:
    For ANY contour with signed area A, n_ink = -sign(A) * [-ty, tx] ALWAYS points from the ink
    into the surrounding white space (outward on outer contours, inward on counter holes, for both CW and CCW)!
    """
    if abs(delta_w) < 0.10:
        return coords
    out = coords.copy()
    total_pts = len(out)
    idx_all = np.arange(total_pts)
    diff_x = out[None, :, 0] - out[:, None, 0]
    diff_y = out[None, :, 1] - out[:, None, 1]
    dist_all = np.sqrt(diff_x * diff_x + diff_y * diff_y) + 1e-6

    all_normals = np.zeros_like(out)
    areas = []
    start = 0
    for end in end_pts:
        pts = out[start : end + 1]
        px, py = pts[:, 0], pts[:, 1]
        area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        areas.append((start, end, area))
        start = end + 1

    if not areas:
        return out
    outer_sign = 1.0 if max(areas, key=lambda x: abs(x[2]))[2] >= 0 else -1.0
    sgn = -outer_sign

    for start, end, area in areas:
        pts = out[start : end + 1]
        if len(pts) >= 4:
            d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
            t = d / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            normals = smooth_1d_periodic(sgn * np.column_stack([-t[:, 1], t[:, 0]]), passes=2)
            normals = normals / (np.hypot(normals[:, 0], normals[:, 1])[:, None] + 1e-6)
            all_normals[start : end + 1] = normals

    for start, end, area in areas:
        pts = out[start : end + 1].copy()
        n = len(pts)
        if n < 4:
            continue
        normals = all_normals[start : end + 1]
        c_rows = np.arange(start, end + 1)
        same_c = (idx_all[None, :] >= start) & (idx_all[None, :] <= end)
        c_pos = c_rows[:, None] - start
        j_pos = idx_all[None, :] - start
        cyc_d = np.minimum(np.abs(c_pos - j_pos), n - np.abs(c_pos - j_pos))
        ignore = same_c & (cyc_d <= 2)

        dx_sub = diff_x[c_rows, :]
        dy_sub = diff_y[c_rows, :]
        d_sub = np.where(ignore, 1e6, dist_all[c_rows, :])
        dir_cos = (dx_sub * normals[:, 0:1] + dy_sub * normals[:, 1:2]) / d_sub
        norm_dot = normals[:, 0:1] * all_normals[None, :, 0] + normals[:, 1:2] * all_normals[None, :, 1]

        is_outer = (area * outer_sign) >= 0
        if is_outer:
            if delta_w > 0:
                is_opp = (dir_cos > 0.45) & (norm_dot < -0.30)
                ahead_dist = np.where(is_opp, d_sub, 1e6).min(axis=1)
                max_allow = smooth_1d_periodic(np.maximum(0.0, 0.36 * (ahead_dist - 14.0)), passes=2)
                off = np.minimum(delta_w, max_allow)
            else:
                is_back = (dir_cos < -0.55) & (norm_dot < -0.30) & (d_sub >= 14.0)
                self_th = np.where(is_back, d_sub, 1e6).min(axis=1)
                min_allow = -smooth_1d_periodic(np.maximum(0.0, 0.44 * (self_th - 22.0)), passes=2)
                off = np.maximum(delta_w, min_allow)
            pts = pts + off[:, None] * normals
        else:
            if delta_w > 0:
                is_opp = (dir_cos > 0.40) & (norm_dot < -0.25)
                hole_diam = np.where(is_opp, d_sub, 1e6).min(axis=1)
                max_allow = smooth_1d_periodic(np.maximum(0.0, 0.34 * (hole_diam - 34.0)), passes=2)
                off = np.minimum(delta_w, max_allow)
                pts = pts + off[:, None] * normals
            else:
                pts = pts + (0.85 * delta_w) * normals
        out[start : end + 1] = pts
    return out


def remodel_math_glyph_anime(coords, flags, end_pts, kind="italic_lower"):
    """
    Remodel a mathematical alphanumeric, Greek, or operator glyph into the unified Anime style:
    - Applies anime_round_and_plump_contour (soft round capsule ends & rounded corners).
    - Applies the Separable Shear-Scale Anime Cushion (opening bowls and giving strokes a
      warm, bouncy hand-drawn elasticity without thinning any stems).
    - Harmonizes vertical bounds (472u x-height for lowercase, 650u cap-height for uppercase/digits)
      and equalizes stroke weight to ~55u (~11.0px) so Math Italic, Greek, Operators, and upright ASCII
      share a 100% uniform visual weight!
    """
    r_passes = 4 if kind != "preserve_bounds" else 2
    coords, flags = anime_round_and_plump_contour(coords, flags, end_pts, plump_offset=0.0, round_passes=r_passes)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    w = x1 - x0
    h = y1 - y0
    if w < 15.0 or h < 15.0:
        return coords, flags

    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rw, rh = 0.5 * w, 0.5 * h
    u = np.clip((coords[:, 0] - cx) / rw, -1.25, 1.25)
    v = np.clip((coords[:, 1] - cy) / rh, -1.25, 1.25)
    u2, v2 = np.clip(u * u, 0.0, 1.0), np.clip(v * v, 0.0, 1.0)

    if kind == "preserve_bounds":
        # For delimiters, radicals, assembly parts, and operators: gentle anime cushion that preserves exact endpoints!
        sx_v = 1.0 + 0.06 * (1.0 - v2)
        bx_v = 0.022 * (1.0 - v2)
        sy_u = 1.0 + 0.05 * (1.0 - u2)
        by_u = 0.022 * (1.0 - u2)
        coords[:, 0] = cx + rw * (sx_v * u + bx_v)
        coords[:, 1] = cy + rh * (sy_u * v + by_u)
        # Regulate stroke thickness of operators & delimiters towards 55.5u (~11.1px) before re-locking bounds
        if w > 35.0 and h > 35.0:
            cur_sw = measure_math_glyph_stroke_units(coords, flags, end_pts)
            err = 55.5 - cur_sw
            if abs(err) > 1.2:
                dw = float(np.clip(0.55 * err, -6.5, 4.0))
                coords = adjust_math_glyph_stroke_thickness(coords, flags, end_pts, dw)
        # Re-lock exact vertical and horizontal min/max so MATH table connectors/assemblies never shift
        nx0, ny0 = coords.min(axis=0)
        nx1, ny1 = coords.max(axis=0)
        if nx1 > nx0:
            coords[:, 0] = x0 + (w / (nx1 - nx0)) * (coords[:, 0] - nx0)
        if ny1 > ny0:
            coords[:, 1] = y0 + (h / (ny1 - ny0)) * (coords[:, 1] - ny0)
        return coords, flags

    is_double_struck = kind == "double_struck"
    is_bold_math = kind.endswith("_bold")
    base_kind = "upper_or_digit" if is_double_struck else (kind[:-5] if is_bold_math else kind)

    # For Alphanumerics & Greek: Separable Anime Cushion (plump round bowls, open chest, bouncy C-arc!)
    if not is_double_struck:
        sx_v = 1.0 + 0.10 * (1.0 - v2) + 0.03 * v
        bx_v = 0.028 * (1.0 - v2) + 0.015 * v
        sy_u = 1.0 + 0.08 * (1.0 - u2)
        by_u = 0.025 * (1.0 - u2)
        coords[:, 0] = cx + rw * 1.03 * (sx_v * u + bx_v)
        coords[:, 1] = cy + rh * (sy_u * v + by_u)

    nx0, ny0 = coords.min(axis=0)
    nx1, ny1 = coords.max(axis=0)
    cur_h = max(20.0, ny1 - ny0)

    target_ymin, target_ymax = ny0, ny1
    if base_kind == "upper_or_digit":
        target_ymin = -8.0 if ny0 < -4.0 else 0.0
        target_ymax = 650.0
    elif base_kind == "lower_x":
        target_ymin = -8.0 if ny0 < -4.0 else 0.0
        target_ymax = 472.0
    elif base_kind == "lower_asc":
        target_ymin = -8.0 if ny0 < -4.0 else 0.0
        target_ymax = 670.0
    elif base_kind == "lower_desc":
        target_ymin = -170.0
        target_ymax = 472.0
    elif base_kind == "lower_both":
        target_ymin = -170.0
        target_ymax = 660.0

    target_h = target_ymax - target_ymin
    sy = float(np.clip(target_h / cur_h, 0.88, 1.16))
    sx = float(np.clip(sy, 0.94, 1.12))
    coords[:, 0] = x0 + sx * (coords[:, 0] - nx0)
    coords[:, 1] = target_ymin + sy * (coords[:, 1] - ny0)

    if is_double_struck:
        # Keep the narrow white slit between the two parallel stems of double-struck letters (ℝ, ℂ, ℕ, ℤ) crisp and open!
        return coords, flags

    # 2-Pass Closed-Loop Stroke Thickness Equalizer so Math Italic, Greek, and Digits share uniform weight!
    target_sw = 76.0 if is_bold_math else 55.5
    for _ in range(2):
        cur_sw = measure_math_glyph_stroke_units(coords, flags, end_pts)
        err = target_sw - cur_sw
        if abs(err) < 1.0:
            break
        dw = float(np.clip(0.58 * err, -9.5, 5.5))
        coords = adjust_math_glyph_stroke_thickness(coords, flags, end_pts, dw)
    return coords, flags


def build_accented_latin_glyphs(fei_reg, fei_math):
    """
    Construct all accented Latin glyphs (0x00C0..0x017F, including 'ï' in naïve, 'è, û, é' in
    Crème brûlée, 'ç' in façade) directly from our newly remodeled Anime base letters in
    Fei-Regular.ttf + full-sized, plump Anime diacritics so 'ï' has two prominent ~105u round dots!
    """
    rcmap = fei_reg.getBestCmap()
    rglyf = fei_reg["glyf"]
    rhmtx = fei_reg["hmtx"]

    mcmap = fei_math.getBestCmap()
    mglyf = fei_math["glyf"]

    # 1. Extract the stem of 'i' (dotlessi, U+0131) and the plump dot of 'i' from Fei-Regular's 'i'
    gi = rglyf[rcmap[ord("i")]]
    ci = np.array(gi.coordinates, dtype=float)
    stem_idxs, dot_idxs = [], []
    s = 0
    for idx, e in enumerate(gi.endPtsOfContours):
        pts = ci[s : e + 1]
        if pts[:, 1].min() > 420:
            dot_idxs.append(idx)
        else:
            stem_idxs.append(idx)
        s = e + 1
    g_dotlessi = extract_contour_subset_glyph(gi, stem_idxs, rglyf)
    g_idot = extract_contour_subset_glyph(gi, dot_idxs, rglyf)

    # 2. Build full-sized, plump Anime diacritic glyphs (centered around x=0, y=540..680):
    #    - Diaeresis (U+0308 / U+00A8): TWO plump Anime i-dots (~108u wide each, separated by 155u!)
    dot_cx = 0.5 * (g_idot.xMin + g_idot.xMax)
    dot_cy = 0.5 * (g_idot.yMin + g_idot.yMax)
    g_dot_centered = transform_simple_glyph(g_idot, rglyf, sx=0.92, sy=0.98, dx=-dot_cx, dy=605.0 - dot_cy, ref_x=dot_cx, ref_y=dot_cy)
    g_diaeresis = combine_simple_glyphs([(g_dot_centered, -78.0, -3.0), (g_dot_centered, +78.0, +3.0)], rglyf)
    g_dotabove = transform_simple_glyph(g_dot_centered, rglyf, dx=0.0, dy=0.0)

    # Extract other diacritics (grave, acute, circumflex, tilde, ring, macron, breve, caron, cedilla)
    # from existing composite/simple accented glyphs in FeiMath and plump/scale them to match!
    def extract_top_accent(src_cp, min_y=455):
        if src_cp not in mcmap:
            return None
        gg = extract_simple_glyph(mglyf[mcmap[src_cp]], mglyf)
        if gg is None:
            return None
        c = np.array(gg.coordinates, dtype=float)
        acc_idxs = []
        st = 0
        for idx, en in enumerate(gg.endPtsOfContours):
            pts = c[st : en + 1]
            if pts[:, 1].min() >= min_y:
                acc_idxs.append(idx)
            st = en + 1
        if not acc_idxs:
            return None
        ga = extract_contour_subset_glyph(gg, acc_idxs, mglyf)
        # Plump and center around (0, 605)
        ca, fa = anime_round_and_plump_contour(
            np.array(ga.coordinates, dtype=float), ga.flags, ga.endPtsOfContours, plump_offset=2.5, round_passes=3
        )
        ga.coordinates = GlyphCoordinates(np.round(ca).astype(int).tolist())
        ga.flags = fa
        ga.recalcBounds(mglyf)
        acx = 0.5 * (ga.xMin + ga.xMax)
        acy = 0.5 * (ga.yMin + ga.yMax)
        return transform_simple_glyph(ga, mglyf, sx=1.10, sy=1.10, dx=-acx, dy=608.0 - acy, ref_x=acx, ref_y=acy)

    g_grave = extract_top_accent(0x00E8)       # è
    g_acute = extract_top_accent(0x00E9)       # é
    g_circumflex = extract_top_accent(0x00EA)  # ê
    g_tilde = extract_top_accent(0x00F1)       # ñ
    g_ring = extract_top_accent(0x00E5)        # å
    g_macron = extract_top_accent(0x0101)      # ā
    g_breve = extract_top_accent(0x0103)       # ă
    g_caron = extract_top_accent(0x010D)       # č

    accent_map = {
        0x0300: g_grave,
        0x0301: g_acute,
        0x0302: g_circumflex,
        0x0303: g_tilde,
        0x0304: g_macron,
        0x0306: g_breve,
        0x0307: g_dotabove,
        0x0308: g_diaeresis,
        0x030A: g_ring,
        0x030C: g_caron,
    }

    built = {}
    # Also register dotlessi (U+0131) and standalone spacing diaeresis (U+00A8)
    adv_i, _ = rhmtx[rcmap[ord("i")]]
    built[0x0131] = (copy.deepcopy(g_dotlessi), (adv_i, g_dotlessi.xMin))
    g_sp_dia = transform_simple_glyph(g_diaeresis, rglyf, dx=250.0, dy=0.0)
    built[0x00A8] = (g_sp_dia, (500, g_sp_dia.xMin))

    # Build all precomposed accented Latin letters in 0x00C0..0x017F using NFD decomposition!
    for cp in range(0x00C0, 0x0180):
        decomp = unicodedata.normalize("NFD", chr(cp))
        if len(decomp) != 2:
            continue
        base_ch, comb_ch = decomp[0], ord(decomp[1])
        base_cp = ord(base_ch)
        if base_cp not in rcmap:
            continue
        if comb_ch in accent_map and accent_map[comb_ch] is not None:
            g_acc = accent_map[comb_ch]
            if base_ch == "i":
                g_base = copy.deepcopy(g_dotlessi)
                base_adv = adv_i
                # For 'ï' (U+00EF), give slightly wider advance so the two plump dots have breathing room!
                if comb_ch == 0x0308:
                    base_adv = max(adv_i + 55, 340)
                    g_base = transform_simple_glyph(g_base, rglyf, dx=25.0, dy=0.0)
            else:
                g_base = copy.deepcopy(rglyf[rcmap[base_cp]])
                base_adv, _ = rhmtx[rcmap[base_cp]]

            bcx = 0.5 * (g_base.xMin + g_base.xMax)
            b_ymax = g_base.yMax
            # Place accent cleanly above the base letter (higher for uppercase/ascenders!)
            acc_ymin = g_acc.yMin
            gap = 52.0 if b_ymax < 520 else 42.0
            dy_acc = (b_ymax + gap) - acc_ymin
            dx_acc = bcx + (12.0 if b_ymax > 520 else 6.0)
            g_comp = combine_simple_glyphs([(g_base, 0.0, 0.0), (g_acc, dx_acc, dy_acc)], rglyf)
            # Ensure lsb >= 24 so wide accents (like 'ï' or 'î') never clip on the left!
            if g_comp.xMin < 24:
                shift_r = 24 - g_comp.xMin
                g_comp = transform_simple_glyph(g_comp, rglyf, dx=shift_r, dy=0.0)
                base_adv += shift_r
            w_comp = g_comp.xMax - g_comp.xMin
            final_adv = max(base_adv, g_comp.xMax + 28)
            built[cp] = (g_comp, (final_adv, g_comp.xMin))

    # Also remodel 'ç' (U+00E7) and 'Ç' (U+00C7) using Fei-Regular's exact 'c'/'C' + cedilla hook!
    for cp_ced, base_c in ((0x00E7, "c"), (0x00C7, "C")):
        if cp_ced in mcmap and ord(base_c) in rcmap:
            g_base = copy.deepcopy(rglyf[rcmap[ord(base_c)]])
            adv_c, _ = rhmtx[rcmap[ord(base_c)]]
            bcx = 0.5 * (g_base.xMin + g_base.xMax)
            by0 = float(g_base.yMin)
            # Build a cute rounded cedilla hook right below the bottom of 'c'/'C'
            hook_pts = np.array([
                [bcx - 16, by0 + 18],
                [bcx + 18, by0 + 18],
                [bcx + 12, by0 - 22],
                [bcx + 42, by0 - 38],
                [bcx + 52, by0 - 76],
                [bcx + 28, by0 - 115],
                [bcx - 22, by0 - 118],
                [bcx - 52, by0 - 98],
                [bcx - 36, by0 - 66],
                [bcx - 8,  by0 - 74],
                [bcx + 8,  by0 - 66],
                [bcx + 2,  by0 - 48],
                [bcx - 24, by0 - 34],
            ], dtype=float)
            g_hook = copy.deepcopy(g_idot)
            g_hook.numberOfContours = 1
            g_hook.endPtsOfContours = [len(hook_pts) - 1]
            g_hook.flags = bytearray([1] * len(hook_pts))
            g_hook.coordinates = GlyphCoordinates(np.round(hook_pts).astype(int).tolist())
            g_hook.recalcBounds(rglyf)
            g_ced = combine_simple_glyphs([(g_base, 0.0, 0.0), (g_hook, 0.0, 0.0)], rglyf)
            built[cp_ced] = (g_ced, (adv_c, g_ced.xMin))

    return built


def classify_math_glyph(gname, cps, vert_parts, horiz_parts):
    if gname in vert_parts or gname in horiz_parts:
        return "skip"
    if ".s" in gname and ".ssty" not in gname:
        return "preserve_bounds"
    if any(tag in gname for tag in (".dsp", ".up", ".updsp", "radical", "brace", "brack", "paren", "integral")):
        return "preserve_bounds"
    if not cps:
        return "preserve_bounds"

    cp = min(cps)
    # Mathematical Alphanumeric Symbols (U+1D400..U+1D7FF) & Greek (U+0370..U+03FF)
    if 0x1D400 <= cp <= 0x1D7FF or 0x0370 <= cp <= 0x03FF or 0x2100 <= cp <= 0x214F:
        name = unicodedata.name(chr(cp), "")
        if "DOUBLE-STRUCK" in name:
            return "double_struck"
        bold_suffix = "_bold" if "BOLD" in name else ""
        if "DIGIT" in name or "CAPITAL" in name:
            return "upper_or_digit" + bold_suffix
        if "SMALL" in name:
            # Check for ascenders / descenders among small Latin/Greek letters
            for word in name.split():
                if word in ("B", "D", "H", "K", "L", "T", "I", "THETA", "DELTA", "LAMBDA"):
                    return "lower_asc" + bold_suffix
                if word in ("G", "P", "Q", "Y", "GAMMA", "ETA", "MU", "RHO", "CHI", "PSI"):
                    return "lower_desc" + bold_suffix
                if word in ("F", "J", "BETA", "ZETA", "XI", "PHI"):
                    return "lower_both" + bold_suffix
            return "lower_x" + bold_suffix
    return "preserve_bounds"


def _math_remodel_worker(args):
    gname, coords_list, flags_bytes, ends_list, kind = args
    coords = np.array(coords_list, dtype=float)
    new_c, new_f = remodel_math_glyph_anime(coords, flags_bytes, ends_list, kind=kind)
    return gname, np.round(new_c).astype(int).tolist(), new_f


def update_fei_math():
    t0 = time.time()
    math_path = os.path.join(FEI_DIR, "FeiMath-Regular.ttf")
    reg_path = os.path.join(FEI_DIR, "Fei-Regular.ttf")

    font = TTFont(math_path)
    fei_reg = TTFont(reg_path)

    for tag in ("fpgm", "prep", "cvt ", "DSIG"):
        if tag in font:
            del font[tag]

    cmap = font.getBestCmap()
    glyf = font["glyf"]
    hmtx = font["hmtx"]

    rcmap = fei_reg.getBestCmap()
    rglyf = fei_reg["glyf"]
    rhmtx = fei_reg["hmtx"]

    # Identify extensible assembly part glyphs in MATH.MathVariants so we never alter their connector joints
    vert_parts, horiz_parts = set(), set()
    if "MATH" in font and font["MATH"].table.MathVariants:
        mv = font["MATH"].table.MathVariants
        for constr in mv.VertGlyphConstruction or []:
            if constr.GlyphAssembly:
                for p in constr.GlyphAssembly.PartRecords:
                    vert_parts.add(p.glyph)
        for constr in mv.HorizGlyphConstruction or []:
            if constr.GlyphAssembly:
                for p in constr.GlyphAssembly.PartRecords:
                    horiz_parts.add(p.glyph)

    # 1. Build clean Anime accented Latin glyphs (including 'ï' with full-sized diaeresis dots!)
    accented_map = build_accented_latin_glyphs(fei_reg, font)

    # 2. Sync upright ASCII letters & digits (A-Z, a-z, 0-9) from Fei-Regular.ttf,
    #    scaled to OpenType MATH dimensions (cap/digit=645, x-height=472 on y=0 baseline) so
    #    upright 'Attention', 'softmax', and digits '1, 2, 6' match math italic 'Q, K, V, x, n, p'!
    synced_ascii = 0
    for ch_code in list(range(ord("0"), ord("9") + 1)) + list(range(ord("A"), ord("Z") + 1)) + list(range(ord("a"), ord("z") + 1)):
        if ch_code in cmap and ch_code in rcmap:
            mgn = cmap[ch_code]
            rgn = rcmap[ch_code]
            gg = copy.deepcopy(rglyf[rgn])
            ch = chr(ch_code)
            adv, lsb = rhmtx[rgn]
            if gg.numberOfContours > 0 and not gg.isComposite():
                coords = np.array(gg.coordinates, dtype=float)
                x0, y0 = coords.min(axis=0)
                x1, y1 = coords.max(axis=0)
                cur_h = max(20.0, y1 - y0)
                overshoot = -10.0 if ch in "CGOSUcosu035689" else 0.0
                if "A" <= ch <= "Z" or "0" <= ch <= "9":
                    ty0 = -20.0 if ch == "Q" else (-12.0 if ch == "J" else overshoot)
                    ty1 = 645.0
                elif ch in "acemnorsuvwxz":
                    ty0, ty1 = overshoot, 472.0
                elif ch in "bdfhkl":
                    ty0, ty1 = 0.0, 665.0
                elif ch == "t":
                    ty0, ty1 = 0.0, 575.0
                elif ch == "i":
                    ty0, ty1 = 0.0, 605.0
                elif ch in "gpqy":
                    ty0, ty1 = -170.0, 472.0
                else:
                    ty0, ty1 = -170.0, 605.0
                sy = (ty1 - ty0) / cur_h
                sx = float(np.clip(sy, 0.98, 1.18))
                old_rsb = max(28, adv - x1)
                coords[:, 0] = x0 + sx * (coords[:, 0] - x0)
                coords[:, 1] = ty0 + sy * (coords[:, 1] - y0)
                gg.coordinates = GlyphCoordinates(np.round(coords).astype(int).tolist())
                gg.recalcBounds(glyf)
                adv = int(round(gg.xMax + old_rsb))
            glyf[mgn] = gg
            glyf[mgn].program = ttProgram.Program()
            glyf[mgn].recalcBounds(glyf)
            hmtx[mgn] = (adv, glyf[mgn].xMin)
            synced_ascii += 1

    # Sync accented Latin into FeiMath
    for cp, (gg, g_m) in accented_map.items():
        if cp in cmap:
            mgn = cmap[cp]
            glyf[mgn] = copy.deepcopy(gg)
            glyf[mgn].recalcBounds(glyf)
            hmtx[mgn] = (g_m[0], glyf[mgn].xMin)

    rev_cmap = {}
    for cp, gn in cmap.items():
        rev_cmap.setdefault(gn, []).append(cp)

    # 3. Prepare multiprocessing tasks to remodel all Mathematical Alphanumerics, Greek, & Operators
    tasks = []
    skipped_ascii_set = set(range(ord("0"), ord("9") + 1)) | set(range(ord("A"), ord("Z") + 1)) | set(range(ord("a"), ord("z") + 1)) | set(accented_map.keys())

    for gname in font.getGlyphOrder():
        if ".ssty" in gname:
            continue
        cps = rev_cmap.get(gname, [])
        if any(c in skipped_ascii_set for c in cps):
            continue
        g = glyf[gname]
        if g.isComposite() and any(0x00A0 <= c <= 0x024F for c in cps):
            gg = extract_simple_glyph(g, glyf)
            if gg is not None:
                glyf[gname] = gg
                g = gg
        if g.numberOfContours > 0 and not g.isComposite():
            kind = classify_math_glyph(gname, cps, vert_parts, horiz_parts)
            if kind != "skip":
                tasks.append((gname, list(g.coordinates), bytearray(g.flags), list(g.endPtsOfContours), kind))

    with Pool(cpu_count()) as pool:
        for gname, new_coords, new_flags in pool.imap_unordered(_math_remodel_worker, tasks, chunksize=128):
            g = glyf[gname]
            old_w = g.xMax - g.xMin
            old_adv, old_lsb = hmtx[gname]
            old_rsb = old_adv - g.xMax
            cps = rev_cmap.get(gname, [])
            kind = classify_math_glyph(gname, cps, vert_parts, horiz_parts)

            g.coordinates = GlyphCoordinates(new_coords)
            g.flags = new_flags
            g.program = ttProgram.Program()
            g.recalcBounds(glyf)
            if kind != "preserve_bounds":
                # Ensure lsb >= 28 and rsb >= 36 (and >= 44 for italic f u1D453 so f''(x) has clean space!)
                min_lsb = 28
                if g.xMin < min_lsb:
                    dx = min_lsb - g.xMin
                    c_arr = np.array(g.coordinates, dtype=int)
                    c_arr[:, 0] += dx
                    g.coordinates = GlyphCoordinates(c_arr.tolist())
                    g.recalcBounds(glyf)
                min_rsb = 88 if gname == "u1D453" else 36
                rsb = max(min_rsb, old_rsb)
                new_adv = g.xMax + rsb
                hmtx[gname] = (new_adv, g.xMin)
            else:
                if gname.startswith("parenleft.s"):
                    # Give tall left parentheses generous left sidebearing so function names like softmax( don't crowd!
                    target_lsb = 105
                    dx = target_lsb - g.xMin
                    if dx > 0:
                        c_arr = np.array(g.coordinates, dtype=int)
                        c_arr[:, 0] += dx
                        g.coordinates = GlyphCoordinates(c_arr.tolist())
                        g.recalcBounds(glyf)
                        old_adv += dx
                hmtx[gname] = (old_adv, g.xMin)

    # 4. Rebuild prime, double-prime, and triple-prime (minute, second, uni2034) with generous
    #    lsb (115u) and wide inter-prime separation (225u step) so f'(x), f''(x), f'''(x) never crowd or overlap!
    if 0x2032 in cmap:
        gn_p1 = cmap[0x2032]
        g_p1 = extract_simple_glyph(glyf[gn_p1], glyf)
        if g_p1 is not None and g_p1.numberOfContours > 0:
            # Take single prime contour, slim slightly if needed, and place at xMin = 115
            g_single = extract_contour_subset_glyph(g_p1, [0], glyf)
            w_p1 = g_single.xMax - g_single.xMin
            if w_p1 > 135:
                g_single = transform_simple_glyph(g_single, glyf, sx=130.0 / w_p1, sy=1.0, ref_x=float(g_single.xMin), ref_y=0.0)
            prime_lsb = 115.0
            g_single = transform_simple_glyph(g_single, glyf, dx=prime_lsb - g_single.xMin, dy=0.0)
            w_s = g_single.xMax - g_single.xMin
            step_x = w_s + 82.0  # 82 font units of clear white gap between adjacent prime strokes!

            g_double = combine_simple_glyphs([(g_single, 0.0, 0.0), (g_single, step_x, 0.0)], glyf)
            g_triple = combine_simple_glyphs([(g_single, 0.0, 0.0), (g_single, step_x, 0.0), (g_single, 2.0 * step_x, 0.0)], glyf)

            for cp_p, gp in ((0x2032, g_single), (0x2033, g_double), (0x2034, g_triple)):
                if cp_p in cmap:
                    gn_p = cmap[cp_p]
                    glyf[gn_p] = copy.deepcopy(gp)
                    glyf[gn_p].recalcBounds(glyf)
                    hmtx[gn_p] = (gp.xMax + 65, gp.xMin)

    # 5. Synchronize .ssty and .ssty2 script variants from their newly remodeled base glyphs!
    ssty_updated = 0
    for gname in font.getGlyphOrder():
        if gname.endswith(".ssty") or gname.endswith(".ssty2"):
            base_gn = gname.rsplit(".", 1)[0]
            if base_gn in glyf and glyf[base_gn].numberOfContours > 0 and not glyf[base_gn].isComposite():
                is_ssty2 = gname.endswith(".ssty2")
                bg = glyf[base_gn]
                # Preserve optical weight gain (+4.8u for .ssty, +7.5u for .ssty2) so 0.70x scripts match main weight!
                bc = adjust_math_glyph_stroke_thickness(
                    np.array(bg.coordinates, dtype=float),
                    bg.flags,
                    bg.endPtsOfContours,
                    delta_w=7.5 if is_ssty2 else 4.8,
                )
                bf = bytearray(bg.flags)
                sg = copy.deepcopy(bg)
                sg.coordinates = GlyphCoordinates(np.round(bc).astype(int).tolist())
                sg.flags = bf
                sg.program = ttProgram.Program()
                sg.recalcBounds(glyf)
                glyf[gname] = sg
                base_adv, _ = hmtx[base_gn]
                hmtx[gname] = (max(base_adv, sg.xMax + 34), sg.xMin)
                ssty_updated += 1

    for gname in font.getGlyphOrder():
        g = glyf[gname]
        if g.isComposite():
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
    print(
        f"Remodeled {math_path} in {dt:.2f}s (synced_ascii={synced_ascii}, "
        f"remodeled_math={len(tasks):,}, ssty_synced={ssty_updated:,})"
    )


if __name__ == "__main__":
    update_fei_math()
