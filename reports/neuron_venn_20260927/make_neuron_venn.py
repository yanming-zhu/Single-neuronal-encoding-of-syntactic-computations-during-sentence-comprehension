#!/usr/bin/env python3
"""Create a count-proportional Venn diagram from three existing unit screens."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn._venn3 import compute_venn3_regions


SOURCES = {
    "function_word": ROOT / "outputs/active_passive_corrected_timepoint_screen_20260722/corrected_unit_results.csv",
    "noun_position": ROOT / "outputs/noun_slot_corrected_timepoint_screen_20260723/corrected_noun_slot_unit_results.csv",
    "verb_context": ROOT / "outputs/fig3_verb_aligned_active_passive_20260805/verb_aligned_active_passive_unit_results.csv",
}
KEY = ["subject", "run", "unit_id"]
ORDER = ["100", "010", "110", "001", "101", "011", "111"]
COLORS = ["#4F85B5", "#55A28D", "#D19B55"]
INK = "#26323C"
GRAY = "#626B72"
WIDTH_MM = 85.0
HEIGHT_MM = 35.0
FONT_PT = 8.0
RASTER_DPI = 600


def load_membership() -> tuple[pd.DataFrame, dict]:
    frames = {}
    audit = {}
    reference = None
    for name, path in SOURCES.items():
        frame = pd.read_csv(path).rename(columns={"patient": "subject"})
        assert not frame.duplicated(KEY).any(), f"Duplicate unit keys in {name}"
        assert frame["has_qualifying_segment"].dtype == bool
        assert frame[KEY + ["ks_label"]].notna().all().all()
        assert (frame["pointwise_alpha"] == 0.025).all()
        identity = frame.set_index(KEY)["ks_label"].sort_index()
        if reference is None:
            reference = identity
        else:
            assert reference.equals(identity), f"Unit universe or labels differ: {name}"
        frames[name] = frame.set_index(KEY)["has_qualifying_segment"].rename(name)
        audit[name] = {
            "source": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_tested": len(frame),
            "n_positive": int(frame["has_qualifying_segment"].sum()),
            "quality_counts_positive": frame.loc[
                frame["has_qualifying_segment"], "ks_label"
            ].value_counts().to_dict(),
        }
    membership = pd.concat([reference.rename("ks_label"), *frames.values()], axis=1)
    membership = membership.reset_index().sort_values(KEY).reset_index(drop=True)
    membership["region"] = membership[list(SOURCES)].astype(int).astype(str).agg("".join, axis=1)
    return membership, audit


def main() -> None:
    membership, source_audit = load_membership()
    region_counts = membership["region"].value_counts().to_dict()
    totals = [int(membership[name].sum()) for name in SOURCES]
    counts = [int(region_counts.get(code, 0)) for code in ORDER]
    outside = int(region_counts.get("000", 0))
    union = sum(counts)
    assert totals == [172, 279, 148]
    assert counts == [72, 156, 49, 49, 25, 48, 26]
    assert union + outside == len(membership) == 1782
    assert membership["subject"].nunique() == 5 and membership["run"].nunique() == 8

    contract = {
        "claim": "Three syntax-related response screens identify partially overlapping recorded neuronal units.",
        "evidence": "One Venn panel: whole-circle areas encode set totals; exact labels encode seven mutually exclusive counts.",
        "archetype": "quantitative grid, single overlap panel",
        "backend": "Python/matplotlib",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "font_family": "Arial",
        "all_text_size_pt": FONT_PT,
        "exports": ["SVG with editable text", "PDF with embedded TrueType text", "PNG at 600 dpi"],
        "visible_content": "Only the circles, seven region counts, three response-type labels, comparison descriptions and three set totals, as selected by the user.",
        "counting_unit": "recording-run-specific Kilosort cluster (good + MUA); not cross-session tracked cells",
        "biological_replicates": 5,
        "sessions": 8,
        "selection": "Use the existing has_qualifying_segment flag, pointwise p < 0.025 for >= 6 consecutive 1-ms bins in 0-800 ms, without gap filling.",
        "tests": {
            "function_word": "two-group one-way ANOVA, active versus passive, function-word onset aligned",
            "noun_position": "paired same-noun comparison of sentence-initial versus post-function-word position, noun onset aligned",
            "verb_context": "paired identical-verb comparison, active versus passive, verb onset aligned",
        },
        "risks": [
            "Three circles generally cannot reproduce all seven region areas exactly; whole-circle areas are proportional, overlap shapes are approximate, labels are exact.",
            "Response-screen membership is descriptive and does not by itself isolate abstract syntax or demonstrate enrichment of overlaps.",
            "The three-condition verb analysis has a different 719-unit universe and is not used here; this figure uses the active-versus-passive verb screen.",
        ],
    }
    (OUT / "figure_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")

    font_files = {
        weight: findfont(FontProperties(family="Arial", weight=weight), fallback_to_default=False)
        for weight in ("normal", "bold")
    }
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FONT_PT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
    })
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([23 / WIDTH_MM, 1 / HEIGHT_MM, 39 / WIDTH_MM, 33 / HEIGHT_MM])
    diagram = venn3(counts, set_labels=None, set_colors=COLORS, alpha=0.29, ax=ax)
    circles = venn3_circles(counts, ax=ax, linewidth=0.6, color=INK)
    for circle, color in zip(circles, COLORS):
        circle.set_edgecolor(color)
    for code in ORDER:
        label = diagram.get_label_by_id(code)
        assert label is not None
        label.set_fontsize(FONT_PT)
        label.set_fontweight("normal")
        label.set_color(INK)

    # Side labels keep all text at 8 pt on the requested landscape page.
    def text_mm(x: float, y: float, content: str, **kwargs):
        return fig.text(x / WIDTH_MM, y / HEIGHT_MM, content,
                        ha="center", va="center", fontsize=FONT_PT,
                        fontfamily="Arial", **kwargs)

    label_specs = [
        (11.7, 24.2, ["Function-word", "responses"], ["Active vs passive"], 172, COLORS[0]),
        (73.2, 32.0, ["Noun-position", "responses"], ["Identical nouns", "across positions"], 279, COLORS[1]),
        (73.2, 15.6, ["Verb-context", "responses"], ["Identical verbs:", "active vs passive"], 148, COLORS[2]),
    ]
    line_step_mm = 3.1
    for x, y, title_lines, subtitle_lines, total, color in label_specs:
        for row, title in enumerate(title_lines):
            text_mm(x, y - line_step_mm * row, title, fontweight="bold", color=color)
        for row, subtitle in enumerate(subtitle_lines, start=len(title_lines)):
            text_mm(x, y - line_step_mm * row, subtitle, color=GRAY)
        text_mm(x, y - line_step_mm * (len(title_lines) + len(subtitle_lines)),
                f"n = {total}", fontweight="bold", color=color)

    ax.set_xlim(-0.70, 0.72)
    ax.set_ylim(-0.62, 0.60)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.canvas.draw()
    for index, angle_deg, label_edge_mm in [(0, 160, 22), (1, 25, 61.9), (2, -40, 61.9)]:
        center = np.asarray(diagram.get_circle_center(index))
        radius = diagram.get_circle_radius(index)
        angle = np.deg2rad(angle_deg)
        point = center + radius * np.array([np.cos(angle), np.sin(angle)])
        x, y = fig.transFigure.inverted().transform(ax.transData.transform(point))
        fig.add_artist(Line2D([x, label_edge_mm / WIDTH_MM], [y, y],
                             transform=fig.transFigure, color=COLORS[index], linewidth=0.55))

    visible_texts = [text for text in fig.findobj(matplotlib.text.Text)
                     if text.get_visible() and text.get_text().strip()]
    assert all(text.get_fontsize() == FONT_PT for text in visible_texts)
    assert all(text.get_fontfamily() == ["Arial"] for text in visible_texts)
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in visible_texts]
    assert all(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
               for box in boxes), "Text outside the specified page"
    overlapping_texts = [
        (visible_texts[i].get_text(), visible_texts[j].get_text())
        for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j])
    ]
    assert not overlapping_texts, overlapping_texts

    stem = OUT / "neuron_venn_area_proportional"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=RASTER_DPI)

    radii = np.array([diagram.get_circle_radius(i) for i in range(3)])
    areas = np.pi * radii ** 2
    np.testing.assert_allclose(areas / areas[0], np.array(totals) / totals[0], rtol=1e-12)
    centers = np.array([diagram.get_circle_center(i) for i in range(3)])
    regions = compute_venn3_regions(centers, radii)
    drawn_equivalent = [float(region.size() * union) for region in regions]
    area_check = pd.DataFrame({
        "region": ORDER,
        "exact_unit_count": counts,
        "drawn_area_equivalent_units_approximate": drawn_equivalent,
    })
    area_check.to_csv(OUT / "overlap_geometry_audit.csv", index=False)
    membership.to_csv(OUT / "unit_membership.csv", index=False)
    region_table = pd.DataFrame({
        "region": ORDER + ["000"],
        "description": [
            "Function word only", "Noun position only", "Function word + noun position only",
            "Verb context only", "Function word + verb context only", "Noun position + verb context only",
            "All three", "None of the three",
        ],
        "n_units": counts + [outside],
    })
    region_table.to_csv(OUT / "region_counts.csv", index=False)
    pd.DataFrame({"response_type": list(SOURCES), "n_units": totals}).to_csv(OUT / "set_totals.csv", index=False)
    membership.groupby("subject")[list(SOURCES)].sum().to_csv(OUT / "counts_by_participant.csv")
    qa = {
        "source_audit": source_audit,
        "unique_key": KEY,
        "identical_unit_universes_and_quality_labels": True,
        "n_units": len(membership),
        "set_totals": dict(zip(SOURCES, totals)),
        "exact_regions": dict(zip(ORDER, counts)),
        "n_union": union,
        "n_outside": outside,
        "circle_radii": radii.tolist(),
        "circle_area_ratios": (areas / areas[0]).tolist(),
        "count_ratios": (np.array(totals) / totals[0]).tolist(),
        "circle_area_proportionality_passed": True,
        "overlap_areas_exact": False,
        "matplotlib_version": matplotlib.__version__,
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "font_family": "Arial",
        "font_files": font_files,
        "all_text_size_pt": FONT_PT,
        "visible_text_count": len(visible_texts),
        "text_clipping_check_passed": True,
        "text_overlap_check_passed": True,
        "visual_review": "pending",
    }
    (OUT / "qa_report.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n")
    plt.close(fig)
    print(json.dumps({"figure": str(stem), "totals": totals, "regions": dict(zip(ORDER, counts)), "circle_area_proportionality": "passed"}, indent=2))


if __name__ == "__main__":
    main()
