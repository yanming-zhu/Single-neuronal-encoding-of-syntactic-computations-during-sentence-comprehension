#!/usr/bin/env python3
"""Stack corrected screen-positive unit heatmaps across five patients.

The two outputs separate units whose qualifying corrected-onset intervals are
active-dominant versus passive-dominant.  Within each patient, rows retain the
source population-heatmap order (run, then probe depth).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


SCREEN_PATH = ROOT / "outputs/active_passive_corrected_timepoint_screen_20260722/unit_crosswalk.csv"
OUTPUT_DIR = ROOT / "reports/active_passive_unit_deck_20260722/figures"
ORDER_PATH = (
    ROOT
    / "outputs/active_passive_corrected_timepoint_screen_20260722"
    / "stacked_significant_heatmap_unit_order.csv"
)
REMOVED_PATH = (
    ROOT
    / "outputs/active_passive_corrected_timepoint_screen_20260722"
    / "stacked_significant_heatmap_removed_noise_rows.csv"
)

PATIENTS = ("gz001", "gz002", "gz003", "gz004", "gz005")
PATIENT_COLORS = {
    "gz001": "#0072B2",
    "gz002": "#E69F00",
    "gz003": "#009E73",
    "gz004": "#CC79A7",
    "gz005": "#D55E00",
}
DIRECTIONS = {
    "active_gt_passive": {
        "relation": "Active > Passive",
        "stem": "Fig_corrected_significant_active_gt_passive_stacked_heatmap",
        "expected": 80,
    },
    "passive_gt_active": {
        "relation": "Active < Passive",
        "stem": "Fig_corrected_significant_active_lt_passive_stacked_heatmap",
        "expected": 92,
    },
}

INK = "#17322F"
GRAY = "#687673"
PAPER = "#FBFAF6"

# Display-only horizontal-stripe QC requested for these heatmaps.  It does not
# alter the corrected significance census (80/92); it removes persistently
# extreme rows from the visualization and records them in REMOVED_PATH.
LOW_FULL_WINDOW_MEAN_HZ = 2.0
HIGH_FULL_WINDOW_MEAN_HZ = 20.0


@dataclass
class DirectionPayload:
    direction: str
    time_sec: np.ndarray
    active_hz: np.ndarray
    passive_hz: np.ndarray
    difference_hz: np.ndarray
    metadata: pd.DataFrame


def filter_horizontal_noise_stripes(
    payload: DirectionPayload,
) -> tuple[DirectionPayload, pd.DataFrame]:
    active_mean = payload.active_hz.mean(axis=1)
    passive_mean = payload.passive_hz.mean(axis=1)
    persistent_low = (
        (active_mean < LOW_FULL_WINDOW_MEAN_HZ)
        & (passive_mean < LOW_FULL_WINDOW_MEAN_HZ)
    )
    persistent_high = (
        (active_mean > HIGH_FULL_WINDOW_MEAN_HZ)
        | (passive_mean > HIGH_FULL_WINDOW_MEAN_HZ)
    )
    remove = persistent_low | persistent_high

    metadata = payload.metadata.copy()
    metadata["source_stacked_row"] = metadata["stacked_row"].astype(int)
    metadata["active_full_window_mean_hz"] = active_mean
    metadata["passive_full_window_mean_hz"] = passive_mean
    metadata["noise_stripe_reason"] = np.select(
        [persistent_high, persistent_low],
        ["persistent_high", "persistent_low"],
        default="retained",
    )
    removed = metadata.loc[remove].reset_index(drop=True)
    kept_metadata = metadata.loc[~remove].reset_index(drop=True)
    kept_metadata["stacked_row"] = np.arange(len(kept_metadata), dtype=int)
    filtered = DirectionPayload(
        direction=payload.direction,
        time_sec=payload.time_sec,
        active_hz=payload.active_hz[~remove],
        passive_hz=payload.passive_hz[~remove],
        difference_hz=payload.difference_hz[~remove],
        metadata=kept_metadata,
    )
    present_patients = tuple(
        patient for patient in PATIENTS if patient in set(kept_metadata["patient"])
    )
    if present_patients != PATIENTS:
        raise ValueError(
            f"{payload.direction}: stripe QC removed every row for at least one patient"
        )
    return filtered, removed


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def heatmap_npz_path(patient: str) -> Path:
    return (
        ROOT
        / f"neuropixel2025/data/{patient}/processed/ActPasSynSentence"
        / "event_related_active_passive"
        / f"paper_figures_{patient}"
        / "fig03_function_word_aligned_population_heatmaps_data.npz"
    )


def load_direction(screen: pd.DataFrame, direction: str) -> DirectionPayload:
    selected = screen.loc[
        screen["corrected_contiguous_gt5ms"].astype(bool)
        & screen["corrected_direction"].eq(direction),
        ["subject", "run", "unit_id", "ks_label", "corrected_direction"],
    ].copy()
    selected_keys = set(
        zip(
            selected["subject"].astype(str),
            selected["run"].astype(str),
            selected["unit_id"].astype(int),
        )
    )

    active_parts: list[np.ndarray] = []
    passive_parts: list[np.ndarray] = []
    difference_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    common_time: np.ndarray | None = None
    global_offset = 0

    for patient_index, patient in enumerate(PATIENTS):
        path = heatmap_npz_path(patient)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as archive:
            time_sec = archive["time_sec"].astype(float)
            if common_time is None:
                common_time = time_sec
            elif not np.array_equal(common_time, time_sec):
                raise ValueError(f"{patient}: time axis differs from the other patients")

            runs = archive["runs"].astype(str)
            unit_ids = archive["unit_ids"].astype(int)
            indices = np.asarray(
                [
                    index
                    for index, (run, unit_id) in enumerate(zip(runs, unit_ids))
                    if (patient, str(run), int(unit_id)) in selected_keys
                ],
                dtype=int,
            )
            if len(indices) == 0:
                raise ValueError(f"{patient}: no selected units for {direction}")

            active_parts.append(archive["active_hz"][indices].astype(np.float32))
            passive_parts.append(archive["passive_hz"][indices].astype(np.float32))
            difference_parts.append(archive["active_minus_passive_hz"][indices].astype(np.float32))

            patient_meta = pd.DataFrame(
                {
                    "direction": direction,
                    "patient": patient,
                    "patient_index": patient_index,
                    "run": runs[indices],
                    "unit_id": unit_ids[indices],
                    "depth_um": archive["depth_um"][indices].astype(float),
                    "source_row": indices,
                    "stacked_row": np.arange(global_offset, global_offset + len(indices)),
                }
            )
            metadata_parts.append(patient_meta)
            global_offset += len(indices)

    if common_time is None:
        raise RuntimeError("No heatmap data were loaded")
    payload = DirectionPayload(
        direction=direction,
        time_sec=common_time,
        active_hz=np.vstack(active_parts),
        passive_hz=np.vstack(passive_parts),
        difference_hz=np.vstack(difference_parts),
        metadata=pd.concat(metadata_parts, ignore_index=True),
    )
    expected = int(DIRECTIONS[direction]["expected"])
    if len(payload.metadata) != expected:
        raise ValueError(f"{direction}: expected {expected} units, found {len(payload.metadata)}")
    return payload


def contiguous_blocks(values: np.ndarray) -> list[tuple[str, int, int]]:
    values = np.asarray(values).astype(str)
    if len(values) == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, len(values)]
    return [(str(values[start]), int(start), int(stop)) for start, stop in zip(starts, stops)]


def draw_boundaries(ax: plt.Axes, metadata: pd.DataFrame, *, strip: bool = False) -> None:
    run_blocks = contiguous_blocks(metadata["run"].to_numpy())
    patient_blocks = contiguous_blocks(metadata["patient"].to_numpy())
    for _, _, stop in run_blocks[:-1]:
        ax.axhline(stop, color="white", lw=1.0 if not strip else 0.8, alpha=0.92, zorder=5)
    for _, _, stop in patient_blocks[:-1]:
        ax.axhline(stop, color="white", lw=2.8 if not strip else 1.6, zorder=6)
        ax.axhline(stop, color=INK, lw=0.65, zorder=7)


def add_time_markers(ax: plt.Axes) -> None:
    for value, style in ((0.0, "--"), (0.8, ":")):
        ax.axvline(value, color="white", lw=1.7, ls=style, alpha=0.95, zorder=6)
        ax.axvline(value, color=INK, lw=0.65, ls=style, alpha=0.95, zorder=7)


def plot_direction(
    payload: DirectionPayload,
    rate_vmax: float,
    difference_vmax: float,
) -> Path:
    configure_style()
    settings = DIRECTIONS[payload.direction]
    n_units = len(payload.metadata)
    patient_blocks = contiguous_blocks(payload.metadata["patient"].to_numpy())
    patient_codes = payload.metadata["patient_index"].to_numpy(int)[:, None]

    fig = plt.figure(figsize=(13.8, 8.2))
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[0.075, 1.0, 1.0, 0.035, 1.0, 0.035],
        wspace=0.16,
    )
    ax_strip = fig.add_subplot(grid[0, 0])
    ax_active = fig.add_subplot(grid[0, 1], sharey=ax_strip)
    ax_passive = fig.add_subplot(grid[0, 2], sharex=ax_active, sharey=ax_strip)
    ax_rate_cbar = fig.add_subplot(grid[0, 3])
    ax_difference = fig.add_subplot(grid[0, 4], sharex=ax_active, sharey=ax_strip)
    ax_difference_cbar = fig.add_subplot(grid[0, 5])
    heatmap_axes = [ax_active, ax_passive, ax_difference]

    patient_cmap = ListedColormap([PATIENT_COLORS[patient] for patient in PATIENTS])
    ax_strip.imshow(
        patient_codes,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[0, 1, n_units, 0],
        cmap=patient_cmap,
        vmin=-0.5,
        vmax=len(PATIENTS) - 0.5,
    )
    centers = [(start + stop) / 2 for _, start, stop in patient_blocks]
    labels = [f"{patient}  n={stop - start}" for patient, start, stop in patient_blocks]
    ax_strip.set_yticks(centers, labels)
    for tick_label, (patient, _, _) in zip(ax_strip.get_yticklabels(), patient_blocks):
        tick_label.set_color(PATIENT_COLORS[patient])
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(10.5)
    ax_strip.set_xticks([])
    ax_strip.tick_params(axis="y", length=0, pad=7)
    ax_strip.set_title("Patient", fontsize=10.5, fontweight="bold", pad=12)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    draw_boundaries(ax_strip, payload.metadata, strip=True)

    extent = [float(payload.time_sec[0]), float(payload.time_sec[-1]), n_units, 0]
    active_image = ax_active.imshow(
        payload.active_hz,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=rate_vmax,
        rasterized=True,
    )
    ax_passive.imshow(
        payload.passive_hz,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=rate_vmax,
        rasterized=True,
    )
    difference_image = ax_difference.imshow(
        payload.difference_hz,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        cmap="RdBu_r",
        vmin=-difference_vmax,
        vmax=difference_vmax,
        rasterized=True,
    )

    for ax, title in zip(
        heatmap_axes,
        ("Active trials", "Passive trials", "Active − passive"),
    ):
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlim(float(payload.time_sec[0]), float(payload.time_sec[-1]))
        ax.set_ylim(n_units, 0)
        ax.set_xticks([-1.0, 0.0, 0.8, 2.0])
        ax.set_xlabel("Time from corrected function-word onset (s)")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        draw_boundaries(ax, payload.metadata)
        add_time_markers(ax)

    rate_cbar = fig.colorbar(active_image, cax=ax_rate_cbar)
    rate_cbar.set_label("Mean firing rate (Hz)", rotation=270, labelpad=15)
    rate_cbar.outline.set_linewidth(0.6)
    difference_cbar = fig.colorbar(difference_image, cax=ax_difference_cbar)
    difference_cbar.set_label("Active − passive (Hz)", rotation=270, labelpad=15)
    difference_cbar.outline.set_linewidth(0.6)

    fig.suptitle(
        f"Corrected screen-positive units: {settings['relation']}",
        fontsize=19,
        fontweight="bold",
        color=INK,
        y=0.978,
    )
    fig.text(
        0.54,
        0.938,
        f"Corrected-significant units after stripe QC (n={n_units} of {settings['expected']}) · "
        "patients stacked gz001→gz005 · "
        "within patient: run then depth",
        ha="center",
        fontsize=11.2,
        color=GRAY,
    )
    fig.text(
        0.54,
        0.018,
        "Selection: corrected-onset pointwise ANOVA p<0.025 for ≥6 consecutive 1-ms bins in 0–0.8 s. "
        "Display QC removes horizontal stripes with either condition's full-window mean >20 Hz or both means <2 Hz.\n"
        "Condition means use Gaussian smoothing σ=35 ms; dashed line=0 s, dotted line=0.8 s.",
        ha="center",
        fontsize=9.2,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.13, right=0.965, top=0.895, bottom=0.12)

    output_stem = OUTPUT_DIR / str(settings["stem"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.png", dpi=320, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight", facecolor=PAPER)
    fig.savefig(f"{output_stem}.svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return Path(f"{output_stem}.png")


def main() -> None:
    screen = pd.read_csv(SCREEN_PATH)
    required = {
        "subject",
        "run",
        "unit_id",
        "ks_label",
        "corrected_contiguous_gt5ms",
        "corrected_direction",
    }
    missing = sorted(required - set(screen.columns))
    if missing:
        raise ValueError(f"Screen table lacks columns: {missing}")
    if int(screen["corrected_contiguous_gt5ms"].astype(bool).sum()) != 172:
        raise ValueError("Expected 172 corrected screen-positive units")

    raw_payloads = {
        direction: load_direction(screen, direction)
        for direction in DIRECTIONS
    }
    payloads: dict[str, DirectionPayload] = {}
    removed_parts: list[pd.DataFrame] = []
    for direction, payload in raw_payloads.items():
        filtered, removed = filter_horizontal_noise_stripes(payload)
        payloads[direction] = filtered
        removed_parts.append(removed)
    rate_values = np.concatenate(
        [
            matrix.ravel()
            for payload in payloads.values()
            for matrix in (payload.active_hz, payload.passive_hz)
        ]
    )
    difference_values = np.concatenate(
        [payload.difference_hz.ravel() for payload in payloads.values()]
    )
    rate_vmax = max(1.0, float(np.nanpercentile(rate_values, 98)))
    difference_vmax = max(1.0, float(np.nanpercentile(np.abs(difference_values), 98)))

    order = pd.concat(
        [payload.metadata for payload in payloads.values()],
        ignore_index=True,
    )
    order["rate_vmax_hz_shared"] = rate_vmax
    order["difference_abs_vmax_hz_shared"] = difference_vmax
    ORDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    order.to_csv(ORDER_PATH, index=False)
    removed_rows = pd.concat(removed_parts, ignore_index=True)
    removed_rows.to_csv(REMOVED_PATH, index=False)

    for payload in payloads.values():
        path = plot_direction(payload, rate_vmax, difference_vmax)
        counts = payload.metadata["patient"].value_counts(sort=False).to_dict()
        print(f"{path} | counts={counts}")
    print(
        f"removed horizontal noise stripes: {len(removed_rows)} rows "
        f"({removed_rows['noise_stripe_reason'].value_counts().to_dict()})"
    )
    print(f"shared scales: firing rate 0–{rate_vmax:.2f} Hz; difference ±{difference_vmax:.2f} Hz")


if __name__ == "__main__":
    main()
