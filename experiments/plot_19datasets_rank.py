"""Generate the 19-dataset Average Rank scatter plot (Figure 10 reproduction).

Reads `results/studd_19datasets_summary.csv` using only the Python standard library,
computes exact Friedman/average ranks (with average tie-breaking) for both our
reproduction and the paper reference (Tables 3-4), and renders a side-by-side
2-panel figure at `results/fig_19datasets_rank_plot.png`.

Uses `matplotlib` when available and falls back to a high-resolution 2x-supersampled
Pillow (PIL) renderer so it runs cleanly in minimal environments.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DEFAULT_INPUT_CSV = RESULTS_DIR / "studd_19datasets_summary.csv"
DEFAULT_OUTPUT_PNG = RESULTS_DIR / "fig_19datasets_rank_plot.png"

METHODS: Tuple[str, ...] = (
    "BL-st",
    "BL-ret",
    "STUDD",
    "SS",
    "DSS",
    "WS",
    "DWS",
    "OS",
    "OF",
    "FF",
)

METHOD_STYLE: Dict[str, Dict[str, str]] = {
    "STUDD": {"group": "STUDD", "color": "#DC2626", "edge": "#7F1D1D", "shape": "star"},
    "SS": {"group": "Supervised", "color": "#2563EB", "edge": "#1E3A8A", "shape": "circle"},
    "DSS": {"group": "Supervised", "color": "#2563EB", "edge": "#1E3A8A", "shape": "circle"},
    "WS": {"group": "Supervised", "color": "#2563EB", "edge": "#1E3A8A", "shape": "circle"},
    "DWS": {"group": "Supervised", "color": "#2563EB", "edge": "#1E3A8A", "shape": "circle"},
    "OS": {"group": "Unsupervised", "color": "#059669", "edge": "#064E3B", "shape": "diamond"},
    "OF": {"group": "Unsupervised", "color": "#059669", "edge": "#064E3B", "shape": "diamond"},
    "FF": {"group": "Unsupervised", "color": "#059669", "edge": "#064E3B", "shape": "diamond"},
    "BL-st": {"group": "Baseline", "color": "#475569", "edge": "#0F172A", "shape": "square"},
    "BL-ret": {"group": "Baseline", "color": "#475569", "edge": "#0F172A", "shape": "square"},
}


def compute_19dataset_ranks(csv_path: Path) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Compute (avg_rank_perf, avg_rank_cost) per method for Our and Paper results."""
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    by_dataset: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_dataset[row["Dataset"]][row["Method"]] = row

    datasets = sorted(by_dataset.keys())
    if not datasets:
        raise ValueError(f"No rows found in {csv_path}")

    def _avg_ranks(metric_key: str, descending: bool) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        for dataset in datasets:
            m_dict = by_dataset[dataset]
            values = [(method, float(m_dict[method][metric_key])) for method in METHODS]
            values.sort(key=lambda item: item[1], reverse=descending)
            i = 0
            while i < len(values):
                j = i
                while j + 1 < len(values) and values[j + 1][1] == values[i][1]:
                    j += 1
                avg_rank = (i + 1 + j + 1) / 2.0
                for k in range(i, j + 1):
                    totals[values[k][0]] += avg_rank
                i = j + 1
        n_ds = float(len(datasets))
        return {method: totals[method] / n_ds for method in METHODS}

    our_perf = _avg_ranks("Our_Kappa", descending=True)
    our_cost = _avg_ranks("Our_Cost", descending=False)
    paper_perf = _avg_ranks("Paper_Kappa", descending=True)
    paper_cost = _avg_ranks("Paper_Cost", descending=False)

    our_points = {m: (our_perf[m], our_cost[m]) for m in METHODS}
    paper_points = {m: (paper_perf[m], paper_cost[m]) for m in METHODS}
    return our_points, paper_points


# ---------------------------------------------------------------------------
# Shared collision-free label layout helper (in pixel space)
# ---------------------------------------------------------------------------

def _rects_overlap_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], pad: float = 0.0) -> float:
    ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    if ox <= 0 or oy <= 0:
        return 0.0
    return ox * oy


def _point_to_rect_anchor(px: float, py: float, rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
    rx0, ry0, rx1, ry1 = rect
    cx = min(max(px, rx0), rx1)
    cy = min(max(py, ry0), ry1)
    return cx, cy


def _segment_dist_to_point(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def place_labels_without_overlap(
    pixel_points: Dict[str, Tuple[float, float]],
    label_sizes: Dict[str, Tuple[float, float]],
    plot_box: Tuple[float, float, float, float],
    forbidden_boxes: Sequence[Tuple[float, float, float, float]],
    scale: float,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Greedily place non-overlapping label bounding boxes inside `plot_box`."""
    x0, y0, x1, y1 = plot_box
    margin = 10.0 * scale

    marker_boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for m, (px, py) in pixel_points.items():
        r = (20.0 if m == "STUDD" else 13.0) * scale
        marker_boxes[m] = (px - r, py - r, px + r, py + r)

    # Order: STUDD first, then most crowded points first
    def _nearest_dist(method: str) -> float:
        px, py = pixel_points[method]
        return min(
            math.hypot(px - ox, py - oy)
            for other, (ox, oy) in pixel_points.items()
            if other != method
        )

    ordered = ["STUDD"] + sorted([m for m in METHODS if m != "STUDD"], key=_nearest_dist)

    placed: Dict[str, Tuple[float, float, float, float]] = {}
    angles_deg = [0, 30, -30, 60, -60, 150, -150, 180, 120, -120, 90, -90, 15, -15, 45, -45, 135, -135]
    radii = [22.0, 32.0, 44.0, 58.0, 74.0, 92.0, 114.0]

    for m in ordered:
        px, py = pixel_points[m]
        lw, lh = label_sizes[m]
        best_rect = (px + 18.0 * scale, py - lh / 2.0, px + 18.0 * scale + lw, py + lh / 2.0)
        best_cost = float("inf")

        for r_base in radii:
            r = r_base * scale
            for ang_idx, deg in enumerate(angles_deg):
                rad = math.radians(deg)
                ux, uy = math.cos(rad), math.sin(rad)
                # Anchor box edge near the ray endpoint so horizontal labels don't float unevenly
                cx = px + ux * (r + lw * 0.42)
                cy = py + uy * (r + lh * 0.48)
                rect = (cx - lw / 2.0, cy - lh / 2.0, cx + lw / 2.0, cy + lh / 2.0)

                # Hard bounds check
                if rect[0] < x0 + margin or rect[2] > x1 - margin or rect[1] < y0 + margin or rect[3] > y1 - margin:
                    continue

                cost = r_base * 1.4 + ang_idx * 0.25

                # Penalty for overlapping any marker
                for om, mbox in marker_boxes.items():
                    ov = _rects_overlap_area(rect, mbox, pad=5.0 * scale)
                    if ov > 0:
                        cost += 200_000.0 + ov

                # Penalty for overlapping already-placed labels
                for om, lbox in placed.items():
                    ov = _rects_overlap_area(rect, lbox, pad=7.0 * scale)
                    if ov > 0:
                        cost += 250_000.0 + ov

                # Penalty for overlapping forbidden badges
                for fbox in forbidden_boxes:
                    ov = _rects_overlap_area(rect, fbox, pad=6.0 * scale)
                    if ov > 0:
                        cost += 180_000.0 + ov

                # Avoid leader line crossing other markers
                ax, ay = _point_to_rect_anchor(px, py, rect)
                for om, (ox, oy) in pixel_points.items():
                    if om == m:
                        continue
                    d_seg = _segment_dist_to_point(px, py, ax, ay, ox, oy)
                    if d_seg < 15.0 * scale:
                        cost += 12_000.0 * (15.0 * scale - d_seg)

                if cost < best_cost:
                    best_cost = cost
                    best_rect = rect

        placed[m] = best_rect
    return placed


# ---------------------------------------------------------------------------
# PIL (Pillow) high-resolution renderer (2x supersampled + LANCZOS)
# ---------------------------------------------------------------------------

def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates_bold = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    candidates_reg = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in (candidates_bold if bold else candidates_reg):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_dashed_line(draw, x0: float, y0: float, x1: float, y1: float, fill: str, width: int, dash: float, gap: float):
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist == 0:
        return
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    pos = 0.0
    while pos < dist:
        end = min(dist, pos + dash)
        draw.line([(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)], fill=fill, width=width)
        pos += dash + gap


def _star_polygon(cx: float, cy: float, r_outer: float, r_inner: float, n_points: int = 5) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n_points * 2):
        angle = -math.pi / 2.0 + i * math.pi / n_points
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def _draw_marker(draw, cx: float, cy: float, shape: str, fill: str, edge: str, scale: float, highlight: bool = False):
    w = max(1, int(round(2.0 * scale)))
    if shape == "star":
        if highlight:
            halo_r = 25.0 * scale
            draw.ellipse([cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r], fill="#FEE2E2", outline="#FCA5A5", width=max(1, int(round(1.5 * scale))))
        poly = _star_polygon(cx, cy, 18.0 * scale, 7.8 * scale, n_points=5)
        draw.polygon(poly, fill=fill, outline=edge)
        # Re-stroke polygon outline cleanly for thicker edge
        draw.line(poly + [poly[0]], fill=edge, width=w, joint="curve")
    elif shape == "circle":
        r = 10.5 * scale
        draw.ellipse([cx - r - 2 * scale, cy - r - 2 * scale, cx + r + 2 * scale, cy + r + 2 * scale], fill="#FFFFFF")
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=edge, width=w)
    elif shape == "diamond":
        r = 12.2 * scale
        outer = [(cx, cy - r - 2 * scale), (cx + r + 2 * scale, cy), (cx, cy + r + 2 * scale), (cx - r - 2 * scale, cy)]
        draw.polygon(outer, fill="#FFFFFF")
        poly = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(poly, fill=fill, outline=edge)
        draw.line(poly + [poly[0]], fill=edge, width=w)
    elif shape == "square":
        r = 9.5 * scale
        draw.rectangle([cx - r - 2 * scale, cy - r - 2 * scale, cx + r + 2 * scale, cy + r + 2 * scale], fill="#FFFFFF")
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=edge, width=w)


def _draw_rotated_text(base_img, text: str, center_xy: Tuple[float, float], font, fill: str, angle: float = 90.0):
    from PIL import Image, ImageDraw

    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    ddraw = ImageDraw.Draw(dummy)
    bbox = ddraw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 16
    txt_img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(txt_img)
    tdraw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=fill)
    rotated = txt_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    rx, ry = int(round(center_xy[0] - rotated.width / 2.0)), int(round(center_xy[1] - rotated.height / 2.0))
    base_img.alpha_composite(rotated, (rx, ry))


def render_with_pil(
    our_points: Dict[str, Tuple[float, float]],
    paper_points: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    scale = 2.0
    width_1x, height_1x = 2200, 1080
    W, H = int(width_1x * scale), int(height_1x * scale)

    img = Image.new("RGBA", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_suptitle = _load_font(int(34 * scale), bold=True)
    f_subtitle = _load_font(int(20 * scale), bold=False)
    f_panel = _load_font(int(25 * scale), bold=True)
    f_axis = _load_font(int(21 * scale), bold=True)
    f_tick = _load_font(int(18 * scale), bold=False)
    f_label = _load_font(int(19 * scale), bold=True)
    f_studd = _load_font(int(20 * scale), bold=True)
    f_hint = _load_font(int(16 * scale), bold=False)
    f_legend = _load_font(int(19 * scale), bold=False)

    # Header
    title = "Average Rank: Predictive Performance vs. Labeling Cost (19 Datasets)"
    tb = draw.textbbox((0, 0), title, font=f_suptitle)
    draw.text(((W - (tb[2] - tb[0])) / 2.0, 26 * scale), title, font=f_suptitle, fill="#0F172A")

    subtitle = "Figure 10 Reproduction (Cerqueira et al., 2023)  —  Lower rank is better on both axes (1 = best, 10 = worst)"
    sb = draw.textbbox((0, 0), subtitle, font=f_subtitle)
    draw.text(((W - (sb[2] - sb[0])) / 2.0, 74 * scale), subtitle, font=f_subtitle, fill="#475569")

    panels = [
        ("Our 19-Dataset Reproduction", our_points, (165.0 * scale, 175.0 * scale, 1035.0 * scale, 885.0 * scale)),
        ("Paper Reference (Tables 3–4)", paper_points, (1245.0 * scale, 175.0 * scale, 2115.0 * scale, 885.0 * scale)),
    ]

    axis_min, axis_max = 1.0, 10.0

    for panel_title, points, plot_box in panels:
        x0, y0, x1, y1 = plot_box
        pw, ph = x1 - x0, y1 - y0

        # Panel title
        ptb = draw.textbbox((0, 0), panel_title, font=f_panel)
        draw.text((x0 + (pw - (ptb[2] - ptb[0])) / 2.0, 130 * scale), panel_title, font=f_panel, fill="#1E293B")

        # Plot background
        draw.rectangle([x0, y0, x1, y1], fill="#FAFBFD", outline="#334155", width=int(2.2 * scale))

        def to_px(rx: float, ry: float) -> Tuple[float, float]:
            px = x0 + (rx - axis_min) / (axis_max - axis_min) * pw
            py = y1 - (ry - axis_min) / (axis_max - axis_min) * ph
            return px, py

        # Gridlines and ticks
        for tick in range(1, 11):
            tx, ty = to_px(float(tick), float(tick))
            if 1 < tick < 10:
                _draw_dashed_line(draw, tx, y0, tx, y1, fill="#CBD5E1", width=int(1.2 * scale), dash=7 * scale, gap=6 * scale)
                _draw_dashed_line(draw, x0, ty, x1, ty, fill="#CBD5E1", width=int(1.2 * scale), dash=7 * scale, gap=6 * scale)

            # Tick marks
            draw.line([(tx, y1), (tx, y1 + 8 * scale)], fill="#334155", width=int(1.8 * scale))
            draw.line([(x0 - 8 * scale, ty), (x0, ty)], fill="#334155", width=int(1.8 * scale))

            # X tick text
            t_str = str(tick)
            xb = draw.textbbox((0, 0), t_str, font=f_tick)
            draw.text((tx - (xb[2] - xb[0]) / 2.0, y1 + 12 * scale), t_str, font=f_tick, fill="#334155")

            # Y tick text
            yb = draw.textbbox((0, 0), t_str, font=f_tick)
            draw.text((x0 - 15 * scale - (yb[2] - yb[0]), ty - (yb[3] - yb[1]) / 2.0 - yb[1]), t_str, font=f_tick, fill="#334155")

        # Subtle directional hint badge in empty bottom-left corner
        hint_text = "Lower-left = Better (higher Kappa, lower cost)"
        hb = draw.textbbox((0, 0), hint_text, font=f_hint)
        hw, hh = hb[2] - hb[0], hb[3] - hb[1]
        hx0 = x0 + 18 * scale
        hy1 = y1 - 16 * scale
        hint_rect = (hx0, hy1 - hh - 14 * scale, hx0 + hw + 22 * scale, hy1)
        draw.rounded_rectangle(hint_rect, radius=8 * scale, fill="#F1F5F9", outline="#CBD5E1", width=int(1.2 * scale))
        draw.text((hint_rect[0] + 11 * scale - hb[0], hint_rect[1] + 7 * scale - hb[1]), hint_text, font=f_hint, fill="#475569")

        # Compute pixel coordinates & label bounding box dimensions
        pixel_points = {m: to_px(*points[m]) for m in METHODS}
        label_texts: Dict[str, str] = {}
        label_sizes: Dict[str, Tuple[float, float]] = {}
        for m in METHODS:
            if m == "STUDD":
                rx, ry = points[m]
                txt = f"STUDD ({rx:.2f}, {ry:.2f})"
                bb = draw.textbbox((0, 0), txt, font=f_studd)
                pad_x, pad_y = 10 * scale, 6 * scale
            else:
                txt = m
                bb = draw.textbbox((0, 0), txt, font=f_label)
                pad_x, pad_y = 6 * scale, 4 * scale
            label_texts[m] = txt
            label_sizes[m] = ((bb[2] - bb[0]) + 2 * pad_x, (bb[3] - bb[1]) + 2 * pad_y)

        placed_labels = place_labels_without_overlap(
            pixel_points=pixel_points,
            label_sizes=label_sizes,
            plot_box=plot_box,
            forbidden_boxes=[hint_rect],
            scale=scale,
        )

        # Draw subtle leader lines first (behind markers and labels)
        for m in METHODS:
            px, py = pixel_points[m]
            rect = placed_labels[m]
            ax, ay = _point_to_rect_anchor(px, py, rect)
            dist = math.hypot(ax - px, ay - py)
            if dist > 15.0 * scale or m == "STUDD":
                line_color = "#DC2626" if m == "STUDD" else "#94A3B8"
                line_w = int((2.0 if m == "STUDD" else 1.4) * scale)
                draw.line([(px, py), (ax, ay)], fill=line_color, width=line_w)

        # Draw markers (non-STUDD first, STUDD on top)
        for m in [k for k in METHODS if k != "STUDD"] + ["STUDD"]:
            px, py = pixel_points[m]
            st = METHOD_STYLE[m]
            _draw_marker(draw, px, py, st["shape"], st["color"], st["edge"], scale, highlight=(m == "STUDD"))

        # Draw labels on top
        for m in [k for k in METHODS if k != "STUDD"] + ["STUDD"]:
            rect = placed_labels[m]
            txt = label_texts[m]
            st = METHOD_STYLE[m]
            if m == "STUDD":
                draw.rounded_rectangle(rect, radius=8 * scale, fill="#FEF2F2", outline="#DC2626", width=int(2.0 * scale))
                bb = draw.textbbox((0, 0), txt, font=f_studd)
                tx = rect[0] + (rect[2] - rect[0] - (bb[2] - bb[0])) / 2.0 - bb[0]
                ty = rect[1] + (rect[3] - rect[1] - (bb[3] - bb[1])) / 2.0 - bb[1]
                draw.text((tx, ty), txt, font=f_studd, fill="#B91C1C")
            else:
                draw.rounded_rectangle(rect, radius=5 * scale, fill=(255, 255, 255, 235), outline=(226, 232, 240, 235), width=int(1.0 * scale))
                bb = draw.textbbox((0, 0), txt, font=f_label)
                tx = rect[0] + (rect[2] - rect[0] - (bb[2] - bb[0])) / 2.0 - bb[0]
                ty = rect[1] + (rect[3] - rect[1] - (bb[3] - bb[1])) / 2.0 - bb[1]
                draw.text((tx, ty), txt, font=f_label, fill=st["color"])

        # X axis label
        x_title = "Average Rank in Predictive Performance (lower = better)"
        xtb = draw.textbbox((0, 0), x_title, font=f_axis)
        draw.text((x0 + (pw - (xtb[2] - xtb[0])) / 2.0, y1 + 48 * scale), x_title, font=f_axis, fill="#1E293B")

    # Y axis label (left of left panel and left of right panel)
    y_title = "Average Rank in Labeling Cost (lower = better)"
    _draw_rotated_text(img, y_title, (56 * scale, 530 * scale), f_axis, "#1E293B", angle=90.0)
    _draw_rotated_text(img, y_title, (1136 * scale, 530 * scale), f_axis, "#1E293B", angle=90.0)

    # Bottom shared legend bar
    legend_items = [
        ("star", "#DC2626", "#7F1D1D", "STUDD (Proposed, Unsupervised)"),
        ("circle", "#2563EB", "#1E3A8A", "Supervised (SS, DSS, WS, DWS)"),
        ("diamond", "#059669", "#064E3B", "Unsupervised Baselines (OS, OF, FF)"),
        ("square", "#475569", "#0F172A", "Static / Periodic Baselines (BL-st, BL-ret)"),
    ]
    item_widths = []
    for _, _, _, text in legend_items:
        bb = draw.textbbox((0, 0), text, font=f_legend)
        item_widths.append((bb[2] - bb[0]) + 42 * scale)
    gap_x = 46 * scale
    total_w = sum(item_widths) + gap_x * (len(legend_items) - 1)
    pad_leg_x = 34 * scale
    leg_y_mid = 1024 * scale
    leg_box = (
        (W - total_w) / 2.0 - pad_leg_x,
        leg_y_mid - 28 * scale,
        (W + total_w) / 2.0 + pad_leg_x,
        leg_y_mid + 28 * scale,
    )
    draw.rounded_rectangle(leg_box, radius=12 * scale, fill="#F8FAFC", outline="#CBD5E1", width=int(1.5 * scale))

    cursor_x = (W - total_w) / 2.0
    for (shape, fill, edge, text), item_w in zip(legend_items, item_widths):
        _draw_marker(draw, cursor_x + 14 * scale, leg_y_mid, shape, fill, edge, scale * 0.85, highlight=False)
        bb = draw.textbbox((0, 0), text, font=f_legend)
        draw.text((cursor_x + 36 * scale - bb[0], leg_y_mid - (bb[3] - bb[1]) / 2.0 - bb[1]), text, font=f_legend, fill="#1E293B")
        cursor_x += item_w + gap_x

    final_img = img.convert("RGB").resize((width_1x, height_1x), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(output_path, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Optional Matplotlib renderer (used if matplotlib is installed)
# ---------------------------------------------------------------------------

def render_with_matplotlib(
    our_points: Dict[str, Tuple[float, float]],
    paper_points: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0), dpi=300)
    fig.patch.set_facecolor("white")

    shape_map = {"star": "*", "circle": "o", "diamond": "D", "square": "s"}
    panels = [
        (axes[0], "Our 19-Dataset Reproduction", our_points),
        (axes[1], "Paper Reference (Tables 3–4)", paper_points),
    ]

    for ax, title, pts in panels:
        ax.set_facecolor("#FAFBFD")
        ax.set_xlim(1.0, 10.0)
        ax.set_ylim(1.0, 10.0)
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(1, 11))
        ax.grid(True, linestyle="--", alpha=0.45, color="#94A3B8")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1E293B", pad=10)
        ax.set_xlabel("Average Rank in Predictive Performance (lower = better)", fontsize=10.5, fontweight="bold", color="#1E293B")
        ax.set_ylabel("Average Rank in Labeling Cost (lower = better)", fontsize=10.5, fontweight="bold", color="#1E293B")

        # Compute collision-free offsets in normalized plot space [0..870, 0..710]
        pw, ph = 870.0, 710.0
        px_pts = {m: ((pts[m][0] - 1.0) / 9.0 * pw, (10.0 - pts[m][1]) / 9.0 * ph) for m in METHODS}
        lbl_sizes = {m: ((175.0, 28.0) if m == "STUDD" else (62.0, 24.0)) for m in METHODS}
        placed = place_labels_without_overlap(px_pts, lbl_sizes, (0.0, 0.0, pw, ph), [(12.0, ph - 44.0, 340.0, ph - 10.0)], scale=1.0)

        for m in [k for k in METHODS if k != "STUDD"] + ["STUDD"]:
            rx, ry = pts[m]
            st = METHOD_STYLE[m]
            is_studd = m == "STUDD"
            ax.scatter(
                rx,
                ry,
                s=340 if is_studd else 95,
                c=st["color"],
                marker=shape_map[st["shape"]],
                edgecolors=st["edge"],
                linewidths=1.2,
                zorder=6 if is_studd else 4,
            )
            rect = placed[m]
            cx = 1.0 + ((rect[0] + rect[2]) / 2.0) / pw * 9.0
            cy = 10.0 - ((rect[1] + rect[3]) / 2.0) / ph * 9.0
            label = f"STUDD ({rx:.2f}, {ry:.2f})" if is_studd else m
            bbox_kw = (
                dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", lw=1.2)
                if is_studd
                else dict(boxstyle="round,pad=0.2", fc="white", ec="#E2E8F0", alpha=0.92, lw=0.7)
            )
            ax.annotate(
                label,
                xy=(rx, ry),
                xytext=(cx, cy),
                ha="center",
                va="center",
                fontsize=9.5 if is_studd else 8.8,
                fontweight="bold",
                color="#B91C1C" if is_studd else st["color"],
                bbox=bbox_kw,
                arrowprops=dict(arrowstyle="-", color="#DC2626" if is_studd else "#94A3B8", lw=0.9),
                zorder=7 if is_studd else 5,
            )

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", label="STUDD (Proposed, Unsupervised)", markerfacecolor="#DC2626", markeredgecolor="#7F1D1D", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="Supervised (SS, DSS, WS, DWS)", markerfacecolor="#2563EB", markeredgecolor="#1E3A8A", markersize=9),
        Line2D([0], [0], marker="D", color="w", label="Unsupervised Baselines (OS, OF, FF)", markerfacecolor="#059669", markeredgecolor="#064E3B", markersize=8.5),
        Line2D([0], [0], marker="s", color="w", label="Static / Periodic Baselines (BL-st, BL-ret)", markerfacecolor="#475569", markeredgecolor="#0F172A", markersize=8.5),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(
        "Average Rank: Predictive Performance vs. Labeling Cost (19 Datasets — Figure 10 Reproduction)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )
    plt.tight_layout(rect=[0.0, 0.07, 1.0, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 19-dataset Average Rank scatter figure (Figure 10).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Path to studd_19datasets_summary.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PNG, help="Output PNG path")
    parser.add_argument(
        "--backend",
        choices=("auto", "matplotlib", "pil"),
        default="auto",
        help="Rendering backend (default: auto tries matplotlib then falls back to PIL).",
    )
    args = parser.parse_args()

    our_points, paper_points = compute_19dataset_ranks(args.input)

    backend = args.backend
    if backend == "auto":
        try:
            import matplotlib  # noqa: F401
            backend = "matplotlib"
        except ImportError:
            backend = "pil"

    if backend == "matplotlib":
        render_with_matplotlib(our_points, paper_points, args.output)
    else:
        render_with_pil(our_points, paper_points, args.output)

    print(f"Saved 19-dataset rank plot ({backend} backend) to {args.output}")


if __name__ == "__main__":
    main()
