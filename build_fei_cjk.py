#!/usr/bin/env python3
"""
Create the new unified Anime / Cozy Hand-Drawn look for Fei-Regular.ttf and Fei-Bold.ttf:
1. Subdivides any long contour segments (<= 55u) so non-linear stroke & radical remodeling operates
   at uniform vertex density without ever pinching or thinning strokes.
2. Plumper Felt-Marker Stroke Weight + Stride-5 Ball-Terminal Shoulder Inflation + Tip Retraction:
   Turns pointy Kaiti needles (撇, 捺, 提, 钩) into plump, round anime drop terminals (圆珠头),
   protected by an opposing-wall clearance guard so multi-stroke counters stay wide open.
3. Multi-Stride (stride 1 & 3) Unit-Bisector Exterior Corner Filleting:
   Shaves off 35u Kaiti 顿笔 shoulders (top-right of 口, 日, 目, 月, 高) into sweeping round anime arcs.
4. Component / Radical-Level Anime Restructuring (部件级动漫重构):
   - Puffs up small enclosed boxes and dots (口, 日, 丶, 氵 with span < 300u) by up to 1.18x,
   - Widens top radicals (艹, 雨, 亠, 宀) while tucking bottom legs,
   - Gives left/right radicals a warm inward anime hug tilt.
5. Separable Shear-Scale Anime Architecture + Bigger, Fuller Scale (English cap=665, x=485; CJK ~720u).
"""

import copy
import os
import time
from multiprocessing import Pool, cpu_count
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


def extract_simple_glyph(mg, mglyf):
    """Extract a simple Glyph from either a simple or composite glyph in mglyf."""
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


def subdivide_tt_contour(pts, flags, max_seg_len=55.0):
    """Exact de Casteljau subdivision of a TrueType quadratic contour to uniform segment density."""
    n = len(pts)
    if n < 3:
        return pts.copy(), bytearray(flags)
    exp_pts, exp_fl = [], []
    for i in range(n):
        p_cur = pts[i]
        f_cur = int(flags[i]) & 1
        p_nxt = pts[(i + 1) % n]
        f_nxt = int(flags[(i + 1) % n]) & 1
        exp_pts.append(p_cur)
        exp_fl.append(f_cur)
        if f_cur == 0 and f_nxt == 0:
            exp_pts.append(0.5 * (p_cur + p_nxt))
            exp_fl.append(1)
    m = len(exp_pts)
    first_on = next((idx for idx, f in enumerate(exp_fl) if f == 1), 0)
    exp_pts = exp_pts[first_on:] + exp_pts[:first_on]
    exp_fl = exp_fl[first_on:] + exp_fl[:first_on]

    out_pts, out_fl = [], []
    idx = 0
    while idx < m:
        p0 = exp_pts[idx]
        nxt_idx = (idx + 1) % m
        if exp_fl[nxt_idx] == 1:
            p1 = exp_pts[nxt_idx]
            dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            k = max(1, int(np.ceil(dist / max_seg_len)))
            for s in range(k):
                t = s / k
                out_pts.append(p0 * (1.0 - t) + p1 * t)
                out_fl.append(1)
            idx += 1
        else:
            q = exp_pts[nxt_idx]
            p2 = exp_pts[(idx + 2) % m]
            dist = float(np.hypot(q[0] - p0[0], q[1] - p0[1]) + np.hypot(p2[0] - q[0], p2[1] - q[1]))
            k = max(1, int(np.ceil(dist / max_seg_len)))
            for s in range(k):
                t0 = s / k
                t1 = (s + 1) / k
                b0 = (1 - t0) ** 2 * p0 + 2 * (1 - t0) * t0 * q + t0 ** 2 * p2
                q_sub = (1 - t0) * (1 - t1) * p0 + ((1 - t0) * t1 + t0 * (1 - t1)) * q + t0 * t1 * p2
                out_pts.append(b0)
                out_fl.append(1)
                out_pts.append(q_sub)
                out_fl.append(0)
            idx += 2
    return np.array(out_pts, dtype=float), bytearray(out_fl)


def sample_tt_contour_points(pts, flags):
    """Sample ordered points along a TrueType quadratic contour and filter micro-steps < 4.0 units."""
    n = len(pts)
    if n <= 3:
        return pts.copy()
    exp_pts = []
    exp_fl = []
    for i in range(n):
        p_cur = pts[i]
        f_cur = int(flags[i]) & 1
        p_nxt = pts[(i + 1) % n]
        f_nxt = int(flags[(i + 1) % n]) & 1
        exp_pts.append(p_cur)
        exp_fl.append(f_cur)
        if f_cur == 0 and f_nxt == 0:
            exp_pts.append(0.5 * (p_cur + p_nxt))
            exp_fl.append(1)
    first_on = next((idx for idx, f in enumerate(exp_fl) if f == 1), 0)
    exp_pts = exp_pts[first_on:] + exp_pts[:first_on]
    exp_fl = exp_fl[first_on:] + exp_fl[:first_on]

    m = len(exp_pts)
    dense = []
    idx = 0
    while idx < m:
        p0 = exp_pts[idx]
        nxt = (idx + 1) % m
        if exp_fl[nxt] == 0:
            q = exp_pts[nxt]
            p2 = exp_pts[(idx + 2) % m]
            dense.append(p0)
            b_mid = 0.25 * p0 + 0.5 * q + 0.25 * p2
            dense.append(b_mid)
            idx += 2
        else:
            dense.append(p0)
            idx += 1
    arr = np.array(dense, dtype=float)
    if len(arr) > 5:
        keep = [0]
        for k in range(1, len(arr)):
            if np.hypot(arr[k, 0] - arr[keep[-1], 0], arr[k, 1] - arr[keep[-1], 1]) >= 4.0:
                keep.append(k)
        if len(keep) >= 4:
            arr = arr[keep]
    return arr


def fit_quad_bezier(chain):
    p0 = chain[0]
    p2 = chain[-1]
    k = len(chain)
    if k <= 2:
        return 0.5 * (p0 + p2), 0.0
    diffs = np.diff(chain, axis=0)
    seg_l = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_l = np.concatenate([[0.0], np.cumsum(seg_l)])
    total_l = cum_l[-1]
    if total_l < 1e-3:
        return 0.5 * (p0 + p2), 0.0
    t = cum_l / total_l
    inner = chain[1:-1]
    ti = t[1:-1, None]
    w = 2.0 * ti * (1.0 - ti)
    w_sum = float(np.sum(w * w))
    if w_sum < 1e-6:
        return 0.5 * (p0 + p2), 999.0
    rhs = inner - ((1.0 - ti) ** 2) * p0 - (ti ** 2) * p2
    q = np.round(np.sum(w * rhs, axis=0) / w_sum)
    b = ((1.0 - ti) ** 2) * p0 + w * q + (ti ** 2) * p2
    err = float(np.max(np.hypot(b[:, 0] - inner[:, 0], b[:, 1] - inner[:, 1])))
    return q, err


def line_max_err(chain):
    p0 = chain[0]
    p2 = chain[-1]
    k = len(chain)
    if k <= 2:
        return 0.0
    v = p2 - p0
    lv = float(np.hypot(v[0], v[1]))
    inner = chain[1:-1]
    if lv < 1e-3:
        return float(np.max(np.hypot(inner[:, 0] - p0[0], inner[:, 1] - p0[1])))
    u = v / lv
    w = inner - p0
    proj = w[:, 0] * u[0] + w[:, 1] * u[1]
    perp = np.abs(w[:, 0] * u[1] - w[:, 1] * u[0])
    over = np.maximum(0.0, np.maximum(-proj, proj - lv))
    return float(np.max(np.hypot(perp, over)))


def compact_tt_contour(pts, flags):
    """Curvature-adaptive TrueType line + quadratic Bezier spline fitting + implied-midpoint coupling."""
    dense = sample_tt_contour_points(pts, flags)
    n = len(dense)
    if n <= 4:
        return pts, bytearray(flags)

    v1 = dense - np.roll(dense, 1, axis=0)
    v2 = np.roll(dense, -1, axis=0) - dense
    l1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-6
    l2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-6
    u1 = v1 / l1[:, None]
    u2 = v2 / l2[:, None]
    cos_1 = np.sum(u1 * u2, axis=1)

    v1_2 = dense - np.roll(dense, 2, axis=0)
    v2_2 = np.roll(dense, -2, axis=0) - dense
    u1_2 = v1_2 / (np.hypot(v1_2[:, 0], v1_2[:, 1])[:, None] + 1e-6)
    u2_2 = v2_2 / (np.hypot(v2_2[:, 0], v2_2[:, 1])[:, None] + 1e-6)
    cos_2 = np.sum(u1_2 * u2_2, axis=1)

    is_corner = (cos_1 < 0.72) & (cos_2 < 0.78)
    corner_idxs = np.where(is_corner)[0]
    start_idx = int(corner_idxs[0]) if len(corner_idxs) > 0 else 0
    dense = np.roll(dense, -start_idx, axis=0)
    is_corner = np.roll(is_corner, -start_idx)
    u2_rolled = np.roll(u2, -start_idx, axis=0)
    closed_pts = np.vstack([dense, dense[0]])
    closed_corner = np.append(is_corner, True)

    out_p = []
    out_f = []
    i = 0
    while i < n:
        best_j = i + 1
        best_kind = "line"
        best_q = None
        t_start = u2_rolled[i]

        max_j = min(n, i + 32)
        for j in range(i + 2, max_j + 1):
            if j > i + 2 and closed_corner[j - 1]:
                break
            sub = closed_pts[i : j + 1]
            chord = float(np.hypot(sub[-1, 0] - sub[0, 0], sub[-1, 1] - sub[0, 1]))
            t_end = u2_rolled[(j - 1) % n]
            cos_span = float(t_start[0] * t_end[0] + t_start[1] * t_end[1])

            is_long_wall = (chord >= 45.0) and (cos_span > 0.88)
            l_tol = 1.45 if is_long_wall else 1.05
            c_tol = 1.95 if is_long_wall else 1.45

            l_err = line_max_err(sub)
            if cos_span > 0.965 and l_err <= l_tol:
                best_j = j
                best_kind = "line"
                best_q = None
                continue
            if cos_span > 0.15:
                q, c_err = fit_quad_bezier(sub)
                if c_err <= c_tol:
                    best_j = j
                    best_kind = "quad"
                    best_q = q
                elif l_err > c_tol * 2.2 and c_err > c_tol * 1.8:
                    break
            else:
                break

        out_p.append(np.round(closed_pts[i]))
        out_f.append(1)
        if best_kind == "quad" and best_q is not None:
            out_p.append(best_q)
            out_f.append(0)
        i = best_j

    m = len(out_p)
    if m <= 4:
        return np.array(out_p, dtype=float), bytearray(out_f)
    drop_mask = [False] * m
    for k in range(m):
        if out_f[k] == 1 and out_f[(k - 1) % m] == 0 and out_f[(k + 1) % m] == 0:
            if drop_mask[(k - 2) % m]:
                continue
            q_prev = out_p[(k - 1) % m]
            q_next = out_p[(k + 1) % m]
            delta = out_p[k] - 0.5 * (q_prev + q_next)
            if float(np.hypot(delta[0], delta[1])) <= 2.4:
                out_p[(k - 1) % m] = np.round(q_prev + delta)
                out_p[(k + 1) % m] = np.round(q_next + delta)
                drop_mask[k] = True

    final_p = [out_p[k] for k in range(m) if not drop_mask[k]]
    final_f = [out_f[k] for k in range(m) if not drop_mask[k]]
    if len(final_p) < 3:
        return np.array(out_p, dtype=float), bytearray(out_f)
    return np.array(final_p, dtype=float), bytearray(final_f)


def smooth_1d_periodic(arr, passes=4):
    out = arr.copy()
    for _ in range(passes):
        out = 0.25 * np.roll(out, 1, axis=0) + 0.50 * out + 0.25 * np.roll(out, -1, axis=0)
    return out


def recover_swave_params(seed):
    rng = np.random.default_rng(seed)

    def rsign(lo=0.65, hi=1.0):
        return rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi)

    rng.uniform(0, 2 * np.pi, 2)
    rng.uniform(-0.012, 0.012)
    rng.uniform(0.81, 0.85)
    rng.uniform(0.78, 0.82)
    p1, p2, p3, p4 = rng.uniform(0, 2 * np.pi, 4)
    sx1, sx2, sx3 = rsign(), rsign(), rsign()
    sy1, sy2, sy3 = rsign(), rsign(), rsign()
    return p1, p2, p3, p4, sx1, sx2, sx3, sy1, sy2, sy3


def greens_theorem_moments(contours_list):
    total_A = 0.0
    sum_Mx = 0.0
    sum_My = 0.0
    sum_Ixx = 0.0
    sum_Iyy = 0.0
    for pts in contours_list:
        x0, y0 = pts[:, 0], pts[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        cross = x1 * y0 - x0 * y1
        total_A += 0.5 * float(np.sum(cross))
        sum_Mx += float(np.sum(cross * (x0 + x1))) / 6.0
        sum_My += float(np.sum(cross * (y0 + y1))) / 6.0
        sum_Ixx += float(np.sum(cross * (x0 * x0 + x0 * x1 + x1 * x1))) / 12.0
        sum_Iyy += float(np.sum(cross * (y0 * y0 + y0 * y1 + y1 * y1))) / 12.0

    if abs(total_A) < 10.0:
        all_p = np.vstack(contours_list)
        cx = 0.5 * (all_p[:, 0].min() + all_p[:, 0].max())
        cy = 0.5 * (all_p[:, 1].min() + all_p[:, 1].max())
        return 100.0, cx, cy, 180.0, 180.0

    cx_opt = sum_Mx / total_A
    cy_opt = sum_My / total_A
    var_x = max(100.0, (sum_Ixx / total_A) - cx_opt * cx_opt)
    var_y = max(100.0, (sum_Iyy / total_A) - cy_opt * cy_opt)
    return total_A, cx_opt, cy_opt, np.sqrt(var_x), np.sqrt(var_y)


def bezier_capsule_terminals_and_round_corners(coords, flags, end_pts, is_cjk=False):
    """Used for English/ASCII glyphs."""
    out_coords = coords.copy()
    out_flags = bytearray(flags)
    start = 0
    for end in end_pts:
        pts = out_coords[start : end + 1].copy()
        fl = out_flags[start : end + 1]
        n = len(pts)
        if n >= 6:
            px, py = pts[:, 0], pts[:, 1]
            area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
            if area < 0:
                prev_p = np.roll(pts, 1, axis=0)
                next_p = np.roll(pts, -1, axis=0)
                v1 = pts - prev_p
                v2 = next_p - pts
                l1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-6
                l2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-6
                u1 = v1 / l1[:, None]
                u2 = v2 / l2[:, None]
                cz = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
                cos_a = np.sum(u1 * u2, axis=1)

                pts_new = pts.copy()
                target_w = 44.0
                for i in range(n):
                    if cz[i] < -0.06 and cos_a[i] < 0.25 and max(l1[i], l2[i]) < 130.0:
                        p_m1 = pts[(i - 1) % n]
                        p_p1 = pts[(i + 1) % n]
                        cross_vec = p_p1 - p_m1
                        cw = float(np.hypot(cross_vec[0], cross_vec[1]))
                        if 4.0 < cw < target_w:
                            u_cross = cross_vec / cw
                            expand = min(5.5, 0.38 * (target_w - cw))
                            if cz[(i - 1) % n] <= 0.04:
                                pts_new[(i - 1) % n] -= expand * u_cross
                            if cz[(i + 1) % n] <= 0.04:
                                pts_new[(i + 1) % n] += expand * u_cross
                        mid_nb = 0.5 * (pts_new[(i - 1) % n] + pts_new[(i + 1) % n])
                        pts_new[i] = 0.56 * pts[i] + 0.44 * mid_nb
                        if (fl[(i - 1) % n] & 1) == 1 and (fl[(i + 1) % n] & 1) == 1:
                            fl[i] = 0
                pts = pts_new

                for _ in range(2):
                    prev_p = np.roll(pts, 1, axis=0)
                    next_p = np.roll(pts, -1, axis=0)
                    v1 = pts - prev_p
                    v2 = next_p - pts
                    l1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-6
                    l2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-6
                    u1 = v1 / l1[:, None]
                    u2 = v2 / l2[:, None]
                    cz = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
                    cos_a = np.sum(u1 * u2, axis=1)
                    is_local_corner = (
                        (cz < -0.05) & (cos_a < 0.72) & (np.minimum(l1, l2) < 85.0) & (np.maximum(l1, l2) < 145.0)
                    )
                    w = np.where(is_local_corner, np.clip((0.72 - cos_a) * 0.38, 0.0, 0.36), 0.0)[:, None]
                    mid = 0.5 * (prev_p + next_p)
                    pts = (1.0 - w) * pts + w * mid

                out_coords[start : end + 1] = pts
                out_flags[start : end + 1] = fl
        start = end + 1
    return out_coords, out_flags


def measure_glyph_stroke_units(coords, flags, ends):
    """Measure 85th-percentile stroke thickness of a glyph in font units (1000 UPM) via 250px raster EDT."""
    scale = 0.25
    pad = 24
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    W = int(np.ceil((x1 - x0) * scale)) + 2 * pad
    H = int(np.ceil((y1 - y0) * scale)) + 2 * pad
    if W < 10 or H < 10:
        return 62.0
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt

    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    s = 0
    polys = []
    for e in ends:
        c_pts = sample_tt_contour_points(coords[s : e + 1], flags[s : e + 1])
        px, py = c_pts[:, 0], c_pts[:, 1]
        area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        pts_img = [(float((p[0] - x0) * scale + pad), float((y1 - p[1]) * scale + pad)) for p in c_pts]
        polys.append((area, pts_img))
        s = e + 1
    for area, pts_img in sorted(polys, key=lambda x: x[0]):
        if len(pts_img) >= 3:
            d.polygon(pts_img, fill=255 if area < 0 else 0)
    arr = np.array(im) > 128
    if not arr.any():
        return 62.0
    dt = distance_transform_edt(arr)
    return float(np.percentile(dt[arr], 85) * 2.0 / scale)


def adjust_glyph_stroke_thickness(coords, flags, end_pts, delta_w):
    """Adjust a simple glyph's stroke thickness by delta_w per side, guarded by opposing-wall distance."""
    if abs(delta_w) < 0.15:
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
        n = len(pts)
        px, py = pts[:, 0], pts[:, 1]
        area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        areas.append((start, end, area))
        if n >= 4:
            d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
            t = d / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            normals = smooth_1d_periodic(np.column_stack([-t[:, 1], t[:, 0]]), passes=2)
            normals = normals / (np.hypot(normals[:, 0], normals[:, 1])[:, None] + 1e-6)
            all_normals[start : end + 1] = normals
        start = end + 1

    for start, end, area in areas:
        pts = out[start : end + 1].copy()
        n = len(pts)
        if n < 4:
            continue
        normals = all_normals[start : end + 1]
        if area < 0:
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

            if delta_w > 0:
                is_opp = (dir_cos > 0.45) & (norm_dot < -0.30)
                ahead_dist = np.where(is_opp, d_sub, 1e6).min(axis=1)
                max_allow = smooth_1d_periodic(np.maximum(0.0, 0.36 * (ahead_dist - 14.0)), passes=2)
                off = np.minimum(delta_w, max_allow)
            else:
                is_back = (dir_cos < -0.55) & (norm_dot < -0.30) & (d_sub >= 18.0)
                self_th = np.where(is_back, d_sub, 1e6).min(axis=1)
                min_allow = -smooth_1d_periodic(np.maximum(0.0, 0.42 * (self_th - 28.0)), passes=2)
                off = np.maximum(delta_w, min_allow)
            pts = pts + off[:, None] * normals
        elif area > 0 and delta_w > 0:
            pts = pts + (0.25 * delta_w) * normals
        out[start : end + 1] = pts
    return out


def redesign_english_bezier_anime(orig_g, glyf_table, ch, old_adv, old_lsb, punct_dy, is_bold=False):
    """Redesign an English/ASCII glyph into the bigger, fuller unified Anime look with equalized stroke weight."""
    gg = copy.deepcopy(orig_g)
    if gg.isComposite() or gg.numberOfContours <= 0 or not hasattr(gg, "coordinates") or len(gg.coordinates) == 0:
        return gg, (old_adv, old_lsb)

    old_w = gg.xMax - gg.xMin
    rsb = max(32, old_adv - (old_lsb + old_w))
    coords = np.array(gg.coordinates, dtype=float)
    flags = bytearray(gg.flags)
    end_pts = gg.endPtsOfContours

    coords, flags = bezier_capsule_terminals_and_round_corners(coords, flags, end_pts, is_cjk=False)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rw, rh = max(10.0, 0.5 * (x1 - x0)), max(10.0, 0.5 * (y1 - y0))

    if ch.isalnum():
        u = np.clip((coords[:, 0] - cx) / rw, -1.25, 1.25)
        v = np.clip((coords[:, 1] - cy) / rh, -1.25, 1.25)
        u2, v2 = np.clip(u * u, 0.0, 1.0), np.clip(v * v, 0.0, 1.0)
        # Separable Shear-Scale Anime Diffeomorphism (du'/du depends ONLY on v so center stems like 'T'
        # and outer stems like 'h', 'H', 'n' maintain identical stroke thickness!)
        sx_v = 1.0 + 0.12 * (1.0 - v2) - 0.025 * v
        bx_v = 0.020 * (1.0 - v2) + 0.018 * v
        sy_u = 1.0 + 0.08 * (1.0 - u2)
        by_u = 0.020 * (1.0 - u2)
        coords[:, 0] = cx + rw * 1.04 * (sx_v * u + bx_v)
        coords[:, 1] = cy + rh * (sy_u * v + by_u)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    cur_h = max(20.0, y1 - y0)

    round_bottom = ch in "CGOSUcosu035689"
    overshoot = -10.0 if round_bottom else 0.0
    if "A" <= ch <= "Z" or "0" <= ch <= "9":
        target_ymin = -22.0 if ch == "Q" else (-14.0 if ch == "J" else overshoot)
        target_ymax = 650.0
    elif ch in "acemnorsuvwxz":
        target_ymin = overshoot
        target_ymax = 472.0
    elif ch in "bdfhkl":
        target_ymin = 0.0
        target_ymax = 670.0
    elif ch == "t":
        target_ymin = 0.0
        target_ymax = 580.0
    elif ch == "i":
        target_ymin = 0.0
        target_ymax = 610.0
    elif ch in "gpqy":
        target_ymin = -170.0
        target_ymax = 472.0
    elif ch == "j":
        target_ymin = -170.0
        target_ymax = 610.0
    else:
        target_ymin = y0 + punct_dy
        target_ymax = y1 + punct_dy

    target_h = target_ymax - target_ymin
    sy = target_h / cur_h if ch.isalnum() else 1.07
    sx = float(np.clip(sy, 0.96, 1.14)) if ch.isalnum() else 1.07
    coords[:, 0] = x0 + sx * (coords[:, 0] - x0)
    coords[:, 1] = target_ymin + sy * (coords[:, 1] - y0)

    # Equalize stroke thickness across all alphanumeric glyphs (slimming overly thick 'T', 'I', '1' and plumping thin letters)
    if ch.isalnum():
        cur_sw = measure_glyph_stroke_units(coords, flags, end_pts)
        target_sw = 82.0 if is_bold else 63.0
        delta_w = float(np.clip(0.55 * (target_sw - cur_sw), -8.5, 5.5))
        coords = adjust_glyph_stroke_thickness(coords, flags, end_pts, delta_w)

    lsb = max(30, old_lsb)
    coords[:, 0] += lsb - coords[:, 0].min()
    gg.coordinates = GlyphCoordinates(np.round(coords).astype(int).tolist())
    gg.flags = flags
    gg.program = ttProgram.Program()
    gg.recalcBounds(glyf_table)
    new_w = gg.xMax - gg.xMin
    return gg, (lsb + new_w + rsb, gg.xMin)


def open_concave_bays_perfect(pts, target_w=27.0):
    """
    Exact 1-Point & 2-Point Crotch Concave-Bay Uncrossing & Opening:
    Detects narrow concave white bays on CW outer contours (including bays whose opposite walls
    crossed over into figure-8 loops) and separates the incoming and outgoing bay walls along
    the bay mouth normal n_bay to at least target_w units.
    """
    n = len(pts)
    if n < 12:
        return pts
    cur = pts.copy()

    for M in (3, 4):
        if n < 2 * M + 4:
            continue
        v_in = np.roll(cur, M, axis=0) - cur
        v_out = np.roll(cur, -M, axis=0) - cur
        l_in = np.hypot(v_in[:, 0], v_in[:, 1]) + 1e-6
        l_out = np.hypot(v_out[:, 0], v_out[:, 1]) + 1e-6
        u_in = v_in / l_in[:, None]
        u_out = v_out / l_out[:, None]

        cos_mouth = np.sum(u_in * u_out, axis=1)
        axis_vec = 0.5 * (u_in + u_out)
        axis_len = np.hypot(axis_vec[:, 0], axis_vec[:, 1]) + 1e-6
        u_axis = axis_vec / axis_len[:, None]
        n_bay = np.column_stack([-u_axis[:, 1], u_axis[:, 0]])

        mouth_sep = np.roll(cur, -M, axis=0) - np.roll(cur, M, axis=0)
        mouth_signed_w = mouth_sep[:, 0] * (-n_bay[:, 0]) + mouth_sep[:, 1] * (-n_bay[:, 1])

        score = cos_mouth * np.where(
            (l_in > 25.0) & (l_out > 25.0) & (mouth_signed_w > -8.0) & (mouth_signed_w < 42.0), 1.0, 0.0
        )
        is_crotch = (score > 0.62) & (score >= np.roll(score, 1)) & (score >= np.roll(score, -1))
        crotch_idxs = list(np.where(is_crotch)[0])
        kept = []
        for kc in sorted(crotch_idxs, key=lambda idx: -score[idx]):
            if all(min(abs(kc - k2), n - abs(kc - k2)) > 3 for k2 in kept):
                kept.append(kc)

        for kc in kept:
            ua = u_axis[kc]
            nb = n_bay[kc]
            ds_prev = abs(
                float((cur[(kc - 1) % n, 0] - cur[kc, 0]) * ua[0] + (cur[(kc - 1) % n, 1] - cur[kc, 1]) * ua[1])
            )
            ds_next = abs(
                float((cur[(kc + 1) % n, 0] - cur[kc, 0]) * ua[0] + (cur[(kc + 1) % n, 1] - cur[kc, 1]) * ua[1])
            )

            if ds_prev < 18.0 and ds_prev <= ds_next:
                base_in, base_out = (kc - 1) % n, kc
                start_m = 0
            elif ds_next < 18.0:
                base_in, base_out = kc, (kc + 1) % n
                start_m = 0
            else:
                base_in, base_out = kc, kc
                start_m = 1

            for m in range(start_m, M + 1):
                i_in = (base_in - m) % n
                i_out = (base_out + m) % n
                p_in = cur[i_in]
                p_out = cur[i_out]
                cur_w = float((p_out[0] - p_in[0]) * (-nb[0]) + (p_out[1] - p_in[1]) * (-nb[1]))
                step_idx = m - start_m + 1
                want_w = target_w * (0.68 if step_idx == 1 else (0.92 if step_idx == 2 else 1.0))
                if cur_w < want_w and abs(cur_w) < 42.0:
                    delta = 0.54 * (want_w - cur_w)
                    cur[i_in] = p_in + delta * nb
                    cur[i_out] = p_out - delta * nb
    return cur


def _cjk_anime_remodel_worker(args):
    gname, coords_list, flags_bytes, ends_list, seed, is_cjk_ideograph, is_bold = args
    orig_c = np.array(coords_list, dtype=float)
    orig_f = bytearray(flags_bytes)

    # Step 0: Un-cross and open narrow concave white bays on the contour FIRST!
    s = 0
    for e in ends_list:
        pts = orig_c[s : e + 1]
        area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
        if area < 0 and len(pts) >= 12:
            orig_c[s : e + 1] = open_concave_bays_perfect(pts, target_w=30.0 if is_bold else 27.0)
        s = e + 1

    # Subdivide long segments first if not already subdivided
    sub_p, sub_f, sub_e = [], bytearray(), []
    s = 0
    for e in ends_list:
        sp, sf = subdivide_tt_contour(orig_c[s : e + 1], orig_f[s : e + 1], max_seg_len=55.0)
        sub_p.append(sp)
        sub_f.extend(sf)
        sub_e.append(sum(len(x) for x in sub_p) - 1)
        s = e + 1
    pts_all = np.vstack(sub_p)
    fl_all = sub_f
    end_pts = sub_e

    # Step 1: Remove old random S-waves
    x0_i, y0_i = pts_all.min(axis=0)
    x1_i, y1_i = pts_all.max(axis=0)
    cx_i, cy_i = 0.5 * (x0_i + x1_i), 0.5 * (y0_i + y1_i)
    rw_i, rh_i = max(45.0, 0.5 * (x1_i - x0_i)), max(45.0, 0.5 * (y1_i - y0_i))
    u_i = np.clip((pts_all[:, 0] - cx_i) / rw_i, -1.2, 1.2)
    v_i = np.clip((pts_all[:, 1] - cy_i) / rh_i, -1.2, 1.2)
    p1, p2, p3, p4, sx1, sx2, sx3, sy1, sy2, sy3 = recover_swave_params(seed)
    w_scale = float(np.clip(min(rw_i / 0.83, rh_i / 0.80) / 280.0, 0.25, 1.0))
    pts_all[:, 0] -= w_scale * (
        16.5 * sx1 * (1.0 - v_i * v_i)
        + 11.0 * sx2 * np.sin(np.pi * v_i)
        + 9.5 * np.sin(1.4 * np.pi * v_i + p1) * np.cos(0.8 * np.pi * u_i + p2)
        + 11.5 * sx3 * (u_i * v_i)
    )
    pts_all[:, 1] -= w_scale * (
        16.5 * sy1 * (1.0 - u_i * u_i)
        + 11.0 * sy2 * np.sin(np.pi * u_i)
        + 9.5 * np.sin(1.4 * np.pi * u_i + p3) * np.cos(0.8 * np.pi * v_i + p4)
        + 10.0 * sy3 * (u_i * v_i)
    )

    total_pts = len(pts_all)
    density_t = float(np.clip((total_pts - 320.0) / 150.0, 0.0, 1.0))
    dense_slim = float(np.clip((total_pts - 380.0) / 140.0, 0.0, 1.0))

    idx_all = np.arange(total_pts)
    diff_x = pts_all[None, :, 0] - pts_all[:, None, 0]
    diff_y = pts_all[None, :, 1] - pts_all[:, None, 1]
    dist_sq = diff_x * diff_x + diff_y * diff_y
    dist_all = np.sqrt(dist_sq) + 1e-6

    all_normals = np.zeros_like(pts_all)
    all_c_guard = np.ones(total_pts, dtype=float)
    all_shoulder_swell = np.zeros(total_pts, dtype=float)
    all_tip_retract = np.zeros(total_pts, dtype=float)
    contours_meta = []

    start = 0
    for end in end_pts:
        pts = pts_all[start : end + 1]
        n = len(pts)
        px, py = pts[:, 0], pts[:, 1]
        area = 0.5 * float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        bbox = (px.min(), py.min(), px.max(), py.max())
        contours_meta.append((start, end, area, bbox))
        if n >= 6:
            d = np.roll(pts, -2, axis=0) - np.roll(pts, 2, axis=0)
            t = d / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            normals = smooth_1d_periodic(np.column_stack([-t[:, 1], t[:, 0]]), passes=3)
            normals = normals / (np.hypot(normals[:, 0], normals[:, 1])[:, None] + 1e-6)
            all_normals[start : end + 1] = normals

        if n >= 10:
            v1_2 = pts - np.roll(pts, 2, axis=0)
            v2_2 = np.roll(pts, -2, axis=0) - pts
            u1_2 = v1_2 / (np.hypot(v1_2[:, 0], v1_2[:, 1])[:, None] + 1e-6)
            u2_2 = v2_2 / (np.hypot(v2_2[:, 0], v2_2[:, 1])[:, None] + 1e-6)
            cz_2 = u1_2[:, 0] * u2_2[:, 1] - u1_2[:, 1] * u2_2[:, 0]

            k_tip = min(5, max(3, n // 5))
            v1_5 = pts - np.roll(pts, k_tip, axis=0)
            v2_5 = np.roll(pts, -k_tip, axis=0) - pts
            u1_5 = v1_5 / (np.hypot(v1_5[:, 0], v1_5[:, 1])[:, None] + 1e-6)
            u2_5 = v2_5 / (np.hypot(v2_5[:, 0], v2_5[:, 1])[:, None] + 1e-6)
            cz_5 = u1_5[:, 0] * u2_5[:, 1] - u1_5[:, 1] * u2_5[:, 0]
            cos_5 = np.sum(u1_5 * u2_5, axis=1)

            c_guard = smooth_1d_periodic(np.clip((0.50 - cz_2) / 0.25, 0.0, 1.0), passes=4)
            all_c_guard[start : end + 1] = c_guard

            if area < 0:
                is_tip_apex = (cos_5 < -0.35) & (cz_2 <= 0.02) & (cz_5 <= 0.04)
                tip_score = np.where(is_tip_apex, np.clip((-0.35 - cos_5) / 0.65, 0.0, 1.0), 0.0)
                all_tip_retract[start : end + 1] = smooth_1d_periodic(tip_score, passes=2)

                sh_swell = np.zeros(n, dtype=float)
                for step_s, w_s in ((2, 0.5), (3, 0.9), (4, 1.0), (5, 0.9), (6, 0.6), (7, 0.3)):
                    if step_s < n // 2:
                        sh_swell = np.maximum(
                            sh_swell,
                            w_s * np.maximum(np.roll(tip_score, step_s), np.roll(tip_score, -step_s)),
                        )
                all_shoulder_swell[start : end + 1] = smooth_1d_periodic(sh_swell, passes=3)
        start = end + 1

    # Closed-Form Self-Thickness Feedback Controller:
    # Measures each glyph's current stroke thickness cur_stroke_w from opposite back walls
    # and computes the exact base_plump needed so simple, medium, and complex characters
    # converge to a unified visual stroke weight (~12.0 px at 200pt in Regular, ~16.2 px in Bold)!
    dir_cos_all = (diff_x * all_normals[:, 0:1] + diff_y * all_normals[:, 1:2]) / dist_all
    norm_dot_all = all_normals[:, 0:1] * all_normals[None, :, 0] + all_normals[:, 1:2] * all_normals[None, :, 1]
    is_back_all = (dir_cos_all < -0.72) & (norm_dot_all < -0.55) & (dist_all >= 24.0) & (dist_all <= 105.0)
    back_d = np.where(is_back_all, dist_all, 1e6).min(axis=1)
    valid_back = back_d[back_d < 105.0]
    cur_stroke_w = float(np.percentile(valid_back, 65)) if len(valid_back) >= 8 else 54.0

    contours_list_0 = [pts_all[s : e + 1] for s, e, _, _ in contours_meta]
    _, _, _, rx0, ry0 = greens_theorem_moments(contours_list_0)
    sw0 = pts_all[:, 0].max() - pts_all[:, 0].min()
    sh0 = pts_all[:, 1].max() - pts_all[:, 1].min()
    gyr0 = 0.5 * (rx0 / max(180.0, sw0) + ry0 / max(180.0, sh0))
    opt_s0 = float(np.clip(1.0 + 0.65 * (0.285 - gyr0), 0.94, 1.05))
    est_scale = opt_s0 * 1.03

    target_w = (81.0 - 5.5 * dense_slim) if is_bold else (61.5 - 4.2 * dense_slim)
    needed_plump = 0.5 * ((target_w / est_scale) - cur_stroke_w)
    base_plump = float(np.clip(needed_plump, -0.4, 4.5 if is_bold else 5.0)) * (1.0 - 0.60 * dense_slim)
    shoulder_extra = (3.2 if is_bold else 3.8) * (1.0 - density_t) ** 1.2

    for start, end, area, _ in contours_meta:
        pts = pts_all[start : end + 1].copy()
        n = len(pts)
        if n >= 10 and area < 0:
            normals = all_normals[start : end + 1]
            c_guard = all_c_guard[start : end + 1]
            sh_swell = all_shoulder_swell[start : end + 1]
            tip_ret = all_tip_retract[start : end + 1] * (1.0 - 0.85 * density_t)

            if abs(base_plump) > 0.05 or shoulder_extra > 0.05:
                c_rows = np.arange(start, end + 1)
                same_c = (idx_all[None, :] >= start) & (idx_all[None, :] <= end)
                c_pos = c_rows[:, None] - start
                j_pos = idx_all[None, :] - start
                cyc_d = np.minimum(np.abs(c_pos - j_pos), n - np.abs(c_pos - j_pos))
                ignore = same_c & (cyc_d <= 3)

                dx_sub = diff_x[c_rows, :]
                dy_sub = diff_y[c_rows, :]
                d2_sub = np.where(ignore, 1e12, dist_sq[c_rows, :])
                dist_sub = np.sqrt(d2_sub) + 1e-6
                dir_cos = (dx_sub * normals[:, 0:1] + dy_sub * normals[:, 1:2]) / dist_sub
                norm_dot = normals[:, 0:1] * all_normals[None, :, 0] + normals[:, 1:2] * all_normals[None, :, 1]

                is_opp = (dir_cos > 0.45) & (norm_dot < -0.30)
                ahead_dist = np.where(is_opp, dist_sub, 1e6).min(axis=1)
                max_allow = smooth_1d_periodic(np.maximum(0.0, 0.36 * (ahead_dist - 20.0)), passes=3)

                want_off = (base_plump + shoulder_extra * sh_swell - 3.5 * tip_ret) * c_guard
                off_amt = smooth_1d_periodic(np.clip(want_off, -2.0, max_allow), passes=3)
                pts = pts + off_amt[:, None] * normals

            fillet_passes = int(round(6 - 4 * dense_slim))
            for _ in range(fillet_passes):
                v1_a = pts - np.roll(pts, 1, axis=0)
                v2_a = np.roll(pts, -1, axis=0) - pts
                l1_a = np.hypot(v1_a[:, 0], v1_a[:, 1]) + 1e-6
                l2_a = np.hypot(v2_a[:, 0], v2_a[:, 1]) + 1e-6
                u1_a, u2_a = v1_a / l1_a[:, None], v2_a / l2_a[:, None]
                cz_a = u1_a[:, 0] * u2_a[:, 1] - u1_a[:, 1] * u2_a[:, 0]
                cos_a = np.sum(u1_a * u2_a, axis=1)

                k3 = min(3, max(1, n // 6))
                v1_b = pts - np.roll(pts, k3, axis=0)
                v2_b = np.roll(pts, -k3, axis=0) - pts
                l1_b = np.hypot(v1_b[:, 0], v1_b[:, 1]) + 1e-6
                l2_b = np.hypot(v2_b[:, 0], v2_b[:, 1]) + 1e-6
                u1_b, u2_b = v1_b / l1_b[:, None], v2_b / l2_b[:, None]
                cz_b = u1_b[:, 0] * u2_b[:, 1] - u1_b[:, 1] * u2_b[:, 0]
                cos_b = np.sum(u1_b * u2_b, axis=1)

                not_concave = (cz_a <= 0.02) & (cz_b <= 0.04)
                is_box_corner = not_concave & (cz_b < -0.12) & (cos_b < 0.78) & (cos_b > -0.20)
                is_local_tip = not_concave & (cz_a < -0.04) & (cos_a < 0.84)

                bisec = -u1_b + u2_b
                b_len = np.hypot(bisec[:, 0], bisec[:, 1]) + 1e-6
                u_bisec = bisec / b_len[:, None]

                d_corner = np.where(
                    is_box_corner,
                    np.minimum(13.0, 0.26 * np.minimum(l1_b, l2_b)) * np.clip((0.78 - cos_b) / 0.90, 0.0, 1.0),
                    0.0,
                )
                d_tip = np.where(
                    is_local_tip,
                    np.minimum(9.0, 0.30 * np.minimum(l1_a, l2_a)) * np.clip((0.84 - cos_a) / 1.00, 0.0, 1.0),
                    0.0,
                )
                d_move = np.maximum(d_corner, d_tip)[:, None]
                pts = pts + d_move * u_bisec

            pts_all[start : end + 1] = pts
        elif n >= 6 and area > 0 and dense_slim > 0.05:
            h_cx = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
            h_cy = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
            hole_scale = 1.0 + (0.28 if is_bold else 0.22) * dense_slim
            pts[:, 0] = h_cx + hole_scale * (pts[:, 0] - h_cx)
            pts[:, 1] = h_cy + hole_scale * (pts[:, 1] - h_cy)
            pts_all[start : end + 1] = pts

    # Step 3: Exact Topological Component Grouping (each CW outer contour + its CCW inner holes!)
    n_c = len(contours_meta)
    parent = list(range(n_c))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n_c):
        if contours_meta[i][2] > 0:
            x0h, y0h, x1h, y1h = contours_meta[i][3]
            hcx, hcy = 0.5 * (x0h + x1h), 0.5 * (y0h + y1h)
            best_j = None
            best_area = 1e18
            for j in range(n_c):
                if contours_meta[j][2] < 0:
                    x0o, y0o, x1o, y1o = contours_meta[j][3]
                    if x0o - 15 <= hcx <= x1o + 15 and y0o - 15 <= hcy <= y1o + 15:
                        box_a = (x1o - x0o) * (y1o - y0o)
                        if box_a < best_area:
                            best_area = box_a
                            best_j = j
            if best_j is not None:
                union(i, best_j)

    groups = {}
    for i in range(n_c):
        groups.setdefault(find(i), []).append(i)

    gx0, gy0 = pts_all.min(axis=0)
    gx1, gy1 = pts_all.max(axis=0)
    g_span_w, g_span_h = gx1 - gx0, gy1 - gy0
    gcx, gcy = 0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1)

    if is_cjk_ideograph and max(g_span_w, g_span_h) > 220.0 and len(groups) > 1:
        sparse_w = (1.0 - density_t) ** 1.5
        for _, c_idxs in groups.items():
            g_rows = np.concatenate([np.arange(contours_meta[i][0], contours_meta[i][1] + 1) for i in c_idxs])
            sub_p = pts_all[g_rows]
            cx0, cy0 = sub_p.min(axis=0)
            cx1, cy1 = sub_p.max(axis=0)
            cw, ch_h = cx1 - cx0, cy1 - cy0
            ccx, ccy = 0.5 * (cx0 + cx1), 0.5 * (cy0 + cy1)

            rel_u = (ccx - gcx) / max(60.0, 0.5 * g_span_w)
            rel_v = (ccy - gcy) / max(60.0, 0.5 * g_span_h)
            max_c_span = max(cw, ch_h)

            if max_c_span < 280.0 and sparse_w > 0.05:
                raw_puff = float(np.clip(1.09 - 0.08 * (max_c_span / 280.0), 1.01, 1.09))
                puff = 1.0 + sparse_w * (raw_puff - 1.0)
            else:
                puff = 1.0 - 0.045 * dense_slim

            comp_sx = puff * (1.0 + 0.045 * sparse_w * rel_v)
            comp_sy = puff * (1.0 + 0.025 * sparse_w * rel_v + 0.055 * dense_slim * max(0.0, rel_v))
            theta_c = 0.020 * sparse_w * rel_u
            cos_c, sin_c = np.cos(theta_c), np.sin(theta_c)
            comp_dx = (-10.0 * sparse_w + 15.0 * dense_slim) * rel_u
            comp_dy = (-8.0 * sparse_w + 16.0 * dense_slim) * rel_v + 8.0 * sparse_w * (1.0 - rel_u * rel_u)

            lx = sub_p[:, 0] - ccx
            ly = sub_p[:, 1] - ccy
            pts_all[g_rows, 0] = ccx + comp_sx * (cos_c * lx - sin_c * ly) + comp_dx
            pts_all[g_rows, 1] = ccy + comp_sy * (sin_c * lx + cos_c * ly) + comp_dy

    # Step 4: Global Separable Shear-Scale Anime Architecture + Harmonious Bigger CJK Scale (~10% larger!)
    x0, y0 = pts_all.min(axis=0)
    x1, y1 = pts_all.max(axis=0)
    span_w, span_h = x1 - x0, y1 - y0
    adv = 915
    if is_cjk_ideograph and max(span_w, span_h) > 220.0:
        contours_list = [pts_all[s : e + 1] for s, e, _, _ in contours_meta]
        area, cx_opt, cy_opt, rx_gyr, ry_gyr = greens_theorem_moments(contours_list)
        piv_x = 0.50 * cx_opt + 0.25 * (x0 + x1)
        piv_y = 0.50 * cy_opt + 0.25 * (y0 + y1)
        rw, rh = max(60.0, 0.5 * span_w), max(60.0, 0.5 * span_h)
        gyr_ratio = 0.5 * (rx_gyr / max(180.0, span_w) + ry_gyr / max(180.0, span_h))
        opt_scale = float(np.clip(1.0 + 0.65 * (0.285 - gyr_ratio), 0.94, 1.05))

        u = np.clip((pts_all[:, 0] - piv_x) / rw, -1.2, 1.2)
        v = np.clip((pts_all[:, 1] - piv_y) / rh, -1.2, 1.2)
        u2 = np.clip(u * u, 0.0, 1.0)
        v2 = np.clip(v * v, 0.0, 1.0)

        waist_lift = 0.080 * (1.0 - dense_slim)
        sx_v = 1.0 + 0.14 * (1.0 - v2) + 0.075 * (1.0 - 0.5 * dense_slim) * v
        bx_v = 0.045 * (1.0 - v2) + 0.016 * v
        sy_u = 1.0 + 0.12 * (1.0 - u2) - 0.022 * u
        by_u = 0.050 * (1.0 - 0.5 * dense_slim) * (1.0 - u2) + waist_lift * (1.0 - v2) - 0.015 * u

        u_anime = sx_v * u + bx_v
        v_anime = sy_u * v + by_u

        pts_all[:, 0] = 458.0 + opt_scale * rw * 0.97 * u_anime
        pts_all[:, 1] = 322.0 + opt_scale * rh * (0.96 + 0.03 * dense_slim) * v_anime

        w = pts_all[:, 0].max() - pts_all[:, 0].min()
        lsb = int(round(0.5 * (adv - w)))
        pts_all[:, 0] += lsb - pts_all[:, 0].min()

    out_p, out_f, out_e = [], bytearray(), []
    for s, e, _, _ in contours_meta:
        cp_c, cf_c = compact_tt_contour(pts_all[s : e + 1], fl_all[s : e + 1])
        if len(cp_c) >= 3:
            out_p.append(cp_c)
            out_f.extend(cf_c)
            out_e.append(sum(len(x) for x in out_p) - 1)
    if not out_p:
        return gname, None
    merged = np.round(np.vstack(out_p)).astype(int).tolist()
    return gname, (merged, out_f, out_e, adv)


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
    fcmap = fei_base.getBestCmap()
    fglyf = fei_base["glyf"]
    fhmtx = fei_base["hmtx"]

    mcmap = fei_math.getBestCmap()
    mglyf = fei_math["glyf"]
    mhmtx = fei_math["hmtx"]

    reused = {}

    h_gn = fcmap[ord("H")]
    h_ymin = float(fglyf[h_gn].yMin) if fglyf[h_gn].numberOfContours > 0 else 0.0
    punct_dy = -h_ymin if h_ymin > 15.0 else 0.0

    for cp in list(range(0x0020, 0x007F)) + [0x2018, 0x2019, 0x201C, 0x201D]:
        if cp in fcmap:
            fgn = fcmap[cp]
            orig_g = fglyf[fgn]
            adv, lsb = fhmtx[fgn]
            if orig_g.numberOfContours > 0 and not orig_g.isComposite() and cp <= 0x007E:
                ch = chr(cp)
                gg, new_m = redesign_english_bezier_anime(orig_g, fglyf, ch, adv, lsb, punct_dy, is_bold=is_bold)
                reused[cp] = (gg, new_m)
            elif orig_g.numberOfContours > 0 and not orig_g.isComposite():
                gg = transform_simple_glyph(orig_g, fglyf, dy=punct_dy)
                reused[cp] = (gg, (adv, gg.xMin))
            else:
                gg = copy.deepcopy(orig_g)
                reused[cp] = (gg, (adv, lsb))

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
    sup_gain = 10.5 if is_bold else 8.2
    for sup_cp, base_cp in sup_map.items():
        if base_cp in reused:
            base_g, (base_adv, _) = reused[base_cp]
            gg = transform_simple_glyph(base_g, fglyf, sx=0.66, sy=0.65, dx=12.0, dy=282.0, ref_x=0.0, ref_y=0.0)
            if gg.numberOfContours > 0 and not gg.isComposite():
                c_sup = adjust_glyph_stroke_thickness(
                    np.array(gg.coordinates, dtype=float), gg.flags, gg.endPtsOfContours, delta_w=sup_gain
                )
                gg.coordinates = GlyphCoordinates(np.round(c_sup).astype(int).tolist())
                gg.recalcBounds(fglyf)
            reused[sup_cp] = (gg, (max(int(round(base_adv * 0.68)) + 24, gg.xMax + 24), gg.xMin))

    sub_map = {
        **{0x2080 + d: ord("0") + d for d in range(10)},
        0x208A: ord("+"),
        0x208B: ord("-"),
        0x208C: ord("="),
        0x208D: ord("("),
        0x208E: ord(")"),
        0x2090: ord("a"),
        0x2091: ord("e"),
        0x2092: ord("o"),
        0x2093: ord("x"),
        0x2095: ord("h"),
        0x2096: ord("k"),
        0x2097: ord("l"),
        0x2098: ord("m"),
        0x2099: ord("n"),
        0x209A: ord("p"),
        0x209B: ord("s"),
        0x209C: ord("t"),
        0x1D62: ord("i"),
        0x1D63: ord("r"),
        0x1D64: ord("u"),
        0x1D65: ord("v"),
        0x2C7C: ord("j"),
    }
    for sub_cp, base_cp in sub_map.items():
        if base_cp in reused:
            base_g, (base_adv, _) = reused[base_cp]
            gg = transform_simple_glyph(base_g, fglyf, sx=0.66, sy=0.65, dx=12.0, dy=-125.0, ref_x=0.0, ref_y=0.0)
            if gg.numberOfContours > 0 and not gg.isComposite():
                c_sub = adjust_glyph_stroke_thickness(
                    np.array(gg.coordinates, dtype=float), gg.flags, gg.endPtsOfContours, delta_w=sup_gain
                )
                gg.coordinates = GlyphCoordinates(np.round(c_sub).astype(int).tolist())
                gg.recalcBounds(fglyf)
            reused[sub_cp] = (gg, (max(int(round(base_adv * 0.68)) + 24, gg.xMax + 24), gg.xMin))

    cjk_adv = 900
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

    if ord("0") in reused:
        g0, _ = reused[ord("0")]
        gg = transform_simple_glyph(g0, fglyf, sx=1.18, sy=1.02, dx=145.0, dy=0.0, ref_x=0.0, ref_y=0.0)
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
            [(gdot, 95 - gdot.xMin, dy_mid), (gdot, 395 - gdot.xMin, dy_mid), (gdot, 695 - gdot.xMin, dy_mid)],
            fglyf,
        )
        reused[0x2026] = (g_ell, (cjk_adv, g_ell.xMin))

    if ord("-") in reused:
        gdash, _ = reused[ord("-")]
        cx_d = 0.5 * (gdash.xMin + gdash.xMax)
        gg_em = transform_simple_glyph(gdash, fglyf, sx=2.15, sy=1.0, dx=450.0 - cx_d, dy=20.0, ref_x=cx_d, ref_y=0.0)
        reused[0x2014] = (gg_em, (cjk_adv, gg_em.xMin))
        reused[0x2015] = (copy.deepcopy(gg_em), (cjk_adv, gg_em.xMin))
        reused[0x2212] = copy.deepcopy(reused[ord("-")])

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
            gg = extract_simple_glyph(mg, mglyf)
            if gg is not None:
                gg.program = ttProgram.Program()
                if is_bold:
                    thicken_simple_glyph(gg, mglyf, offset_units=9.5)
                gg.recalcBounds(mglyf)
                m_adv, _ = mhmtx[mgn]
                gh = gg.yMax - gg.yMin
                gw = gg.xMax - gg.xMin
                if cp >= 0x2190 and (gh > 640 or gg.yMin < -80):
                    s_fit = min(1.0, 620.0 / max(1.0, gh))
                    cx_m = 0.5 * (gg.xMin + gg.xMax)
                    cy_m = 0.5 * (gg.yMin + gg.yMax)
                    gg = transform_simple_glyph(
                        gg, mglyf, sx=s_fit, sy=s_fit, dx=0.0, dy=290.0 - cy_m, ref_x=cx_m, ref_y=cy_m
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
    cjk_tasks = []
    for idx, gname in enumerate(font.getGlyphOrder()):
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
        elif g.numberOfContours > 0 and not g.isComposite():
            seed = (min(cps) if cps else idx) * 131 + 17
            is_cjk_ideo = any(c >= 0x3400 for c in cps)
            cjk_tasks.append(
                (gname, list(g.coordinates), bytearray(g.flags), list(g.endPtsOfContours), seed, is_cjk_ideo, is_bold)
            )

    with Pool(cpu_count()) as pool:
        for gname, res in pool.imap_unordered(_cjk_anime_remodel_worker, cjk_tasks, chunksize=256):
            if res is not None:
                merged_coords, out_flags, out_ends, new_adv = res
                g = glyf[gname]
                g.numberOfContours = len(out_ends)
                g.coordinates = GlyphCoordinates(merged_coords)
                g.flags = out_flags
                g.endPtsOfContours = out_ends
                g.program = ttProgram.Program()
                g.recalcBounds(glyf)
                orig_adv, _ = hmtx[gname]
                hmtx[gname] = (new_adv if orig_adv >= 800 else orig_adv, g.xMin)

    if "post" in font:
        font["post"].formatType = 3.0

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
        f"Remodeled {font_path} in {dt:.2f}s (total_glyphs={len(font.getGlyphOrder()):,}, "
        f"synced_latin_math={synced_count:,}, cjk_glyphs={len(cjk_tasks):,})"
    )


if __name__ == "__main__":
    update_fei_font(os.path.join(FEI_DIR, "Fei-Regular.ttf"), "Regular", "Fei Regular", "Fei-Regular")
    update_fei_font(os.path.join(FEI_DIR, "Fei-Bold.ttf"), "Bold", "Fei Bold", "Fei-Bold")
