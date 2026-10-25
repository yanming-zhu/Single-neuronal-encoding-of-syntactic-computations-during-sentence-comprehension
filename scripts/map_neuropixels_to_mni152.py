#!/usr/bin/env python3
"""Plot Neuropixels electrode locations on a shared MNI152 brain.

Input CSV can contain either contact-level MNI coordinates:

    subject,contact,mni_x,mni_y,mni_z

or trajectory-level entry/tip coordinates:

    subject,entry_x,entry_y,entry_z,tip_x,tip_y,tip_z

Rows with native_x/native_y/native_z plus transform_path are also supported
for affine native-to-MNI transforms. Nonlinear transforms should be applied
upstream with ANTs/FSL, then supplied as MNI coordinates here.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets


CONTACT_COLUMNS = ("mni_x", "mni_y", "mni_z")
NATIVE_COLUMNS = ("native_x", "native_y", "native_z")
TRAJECTORY_COLUMNS = (
    "entry_x",
    "entry_y",
    "entry_z",
    "tip_x",
    "tip_y",
    "tip_z",
)


@dataclass(frozen=True)
class Trajectory:
    subject: str
    entry: np.ndarray
    tip: np.ndarray
    source: str = ""
    confidence: str = ""
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map Neuropixels contacts or probe trajectories to MNI152."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "neuropixel2025/data/electrode_mni/"
            "neuropixels_mni152_approx_from_screenshots.csv"
        ),
        help="CSV with MNI contact coordinates or entry/tip trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for figure and derived point table.",
    )
    parser.add_argument(
        "--prefix",
        default="neuropixels_mni152_map",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--samples-per-trajectory",
        type=int,
        default=24,
        help="Number of displayed sample points along each entry-tip line.",
    )
    parser.add_argument(
        "--title",
        default="Neuropixels electrode locations on MNI152",
        help="Figure title.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Annotate subjects next to their trajectory tips/points.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats. Supported: png,pdf.",
    )
    return parser.parse_args()


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(str(path_text))
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_affine(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine matrix in {path}, got {arr.shape}")
    return arr


def apply_affine_contacts(df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    """Fill missing mni_x/y/z from native coordinates and transform_path."""
    if "transform_path" not in df.columns:
        return df

    native_available = all(col in df.columns for col in NATIVE_COLUMNS)
    if not native_available:
        return df

    for col in CONTACT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    affine_cache: dict[Path, np.ndarray] = {}
    for idx, row in df.iterrows():
        if all(np.isfinite(pd.to_numeric(row.get(col), errors="coerce")) for col in CONTACT_COLUMNS):
            continue
        native = np.array(
            [pd.to_numeric(row.get(col), errors="coerce") for col in NATIVE_COLUMNS],
            dtype=float,
        )
        if not np.all(np.isfinite(native)) or pd.isna(row.get("transform_path")):
            continue

        affine_path = resolve_path(str(row["transform_path"]), input_dir)
        if affine_path not in affine_cache:
            affine_cache[affine_path] = load_affine(affine_path)
        mni = nib.affines.apply_affine(affine_cache[affine_path], native)
        for col, value in zip(CONTACT_COLUMNS, mni):
            df.at[idx, col] = value

    return df


def load_rows(input_csv: Path) -> tuple[pd.DataFrame, list[Trajectory]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "subject" not in df.columns:
        raise ValueError("Input CSV must contain a 'subject' column")

    df = apply_affine_contacts(df, input_csv.parent)

    trajectories: list[Trajectory] = []
    if all(col in df.columns for col in TRAJECTORY_COLUMNS):
        for _, row in df.iterrows():
            values = [pd.to_numeric(row.get(col), errors="coerce") for col in TRAJECTORY_COLUMNS]
            if not np.all(np.isfinite(values)):
                continue
            entry = np.array(values[:3], dtype=float)
            tip = np.array(values[3:], dtype=float)
            trajectories.append(
                Trajectory(
                    subject=str(row["subject"]),
                    entry=entry,
                    tip=tip,
                    source=str(row.get("source", "")),
                    confidence=str(row.get("confidence", "")),
                    note=str(row.get("note", "")),
                )
            )

    if all(col in df.columns for col in CONTACT_COLUMNS):
        point_mask = np.logical_and.reduce(
            [np.isfinite(numeric_series(df, col).to_numpy(dtype=float)) for col in CONTACT_COLUMNS]
        )
        points = df.loc[point_mask].copy()
        for col in CONTACT_COLUMNS:
            points[col] = pd.to_numeric(points[col], errors="coerce")
    else:
        points = pd.DataFrame(columns=list(df.columns) + list(CONTACT_COLUMNS))

    if points.empty and not trajectories:
        raise ValueError(
            "No plottable coordinates found. Provide mni_x/mni_y/mni_z rows "
            "or entry_x/entry_y/entry_z/tip_x/tip_y/tip_z rows."
        )

    return points, trajectories


def mni_axis_values(img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = img.shape[:3]
    affine = img.affine
    x = affine[0, 0] * np.arange(shape[0]) + affine[0, 3]
    y = affine[1, 1] * np.arange(shape[1]) + affine[1, 3]
    z = affine[2, 2] * np.arange(shape[2]) + affine[2, 3]
    return x, y, z


def normalized_template_data() -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    img = datasets.load_mni152_template(resolution=2)
    mask = datasets.load_mni152_brain_mask(resolution=2).get_fdata().astype(bool)
    data = img.get_fdata()
    data = np.where(mask, data, np.nan)
    lo, hi = np.nanpercentile(data, [2, 99.5])
    data = np.clip((data - lo) / (hi - lo), 0, 1)
    return img, data, mask


def projection_background(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    view: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], str, str]:
    def safe_nanmax(array: np.ndarray, axis: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            projection = np.nanmax(array, axis=axis)
        return np.nan_to_num(projection, nan=0.0)

    if view == "sagittal_left":
        hemi = x <= 0
        proj = safe_nanmax(data[hemi, :, :], axis=0).T
        extent = (float(y.min()), float(y.max()), float(z.min()), float(z.max()))
        return proj, extent, "MNI y", "MNI z"
    if view == "axial":
        proj = safe_nanmax(data, axis=2).T
        extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
        return proj, extent, "MNI x", "MNI y"
    if view == "coronal":
        proj = safe_nanmax(data, axis=1).T
        extent = (float(x.min()), float(x.max()), float(z.min()), float(z.max()))
        return proj, extent, "MNI x", "MNI z"
    raise ValueError(f"Unknown view: {view}")


def sort_contacts(group: pd.DataFrame) -> pd.DataFrame:
    for column in ("contact", "channel", "depth_um", "depth_mm"):
        if column in group.columns:
            return group.assign(_sort=pd.to_numeric(group[column], errors="coerce")).sort_values(
                "_sort", na_position="last"
            )
    return group


def sample_trajectory(traj: Trajectory, samples: int) -> pd.DataFrame:
    samples = max(2, samples)
    rows = []
    for idx, t in enumerate(np.linspace(0, 1, samples)):
        xyz = traj.entry * (1 - t) + traj.tip * t
        rows.append(
            {
                "subject": traj.subject,
                "sample_index": idx,
                "sample_fraction_entry_to_tip": t,
                "mni_x": xyz[0],
                "mni_y": xyz[1],
                "mni_z": xyz[2],
                "source": traj.source,
                "confidence": traj.confidence,
                "note": traj.note,
            }
        )
    return pd.DataFrame(rows)


def projection_xy(points: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    if view == "sagittal_left":
        return points[:, 1], points[:, 2]
    if view == "axial":
        return points[:, 0], points[:, 1]
    if view == "coronal":
        return points[:, 0], points[:, 2]
    raise ValueError(f"Unknown view: {view}")


def finite_xyz(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 3), dtype=float)
    return df.loc[:, CONTACT_COLUMNS].to_numpy(dtype=float)


def subject_order(points: pd.DataFrame, trajectories: Iterable[Trajectory]) -> list[str]:
    subjects = []
    for subject in points.get("subject", []):
        subject = str(subject)
        if subject not in subjects:
            subjects.append(subject)
    for traj in trajectories:
        if traj.subject not in subjects:
            subjects.append(traj.subject)
    return subjects


def plot_map(
    points: pd.DataFrame,
    trajectories: list[Trajectory],
    output_dir: Path,
    prefix: str,
    samples_per_trajectory: int,
    title: str,
    show_labels: bool,
    formats: Iterable[str],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    img, data, _ = normalized_template_data()
    x, y, z = mni_axis_values(img)

    subjects = subject_order(points, trajectories)
    cmap = plt.get_cmap("tab10")
    colors = {subject: cmap(idx % 10) for idx, subject in enumerate(subjects)}

    fig = plt.figure(figsize=(13.5, 8.2), dpi=150)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0), wspace=0.13, hspace=0.2)
    axes = {
        "sagittal_left": fig.add_subplot(grid[:, 0]),
        "axial": fig.add_subplot(grid[0, 1]),
        "coronal": fig.add_subplot(grid[1, 1]),
    }
    labels = {
        "sagittal_left": "Left hemisphere projection",
        "axial": "Axial projection",
        "coronal": "Coronal projection",
    }

    for view, ax in axes.items():
        bg, extent, xlabel, ylabel = projection_background(data, x, y, z, view)
        ax.imshow(
            bg,
            extent=extent,
            origin="lower",
            cmap="gray",
            alpha=0.82,
            interpolation="bilinear",
        )
        ax.set_title(labels[view], fontsize=11, pad=7)
        ax.set_xlabel(f"{xlabel} (mm)", fontsize=9)
        ax.set_ylabel(f"{ylabel} (mm)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(color="white", alpha=0.16, linewidth=0.5)
        ax.tick_params(labelsize=8)

    sampled_frames: list[pd.DataFrame] = []
    for traj in trajectories:
        sampled = sample_trajectory(traj, samples_per_trajectory)
        sampled_frames.append(sampled)
        xyz = sampled.loc[:, CONTACT_COLUMNS].to_numpy(dtype=float)
        color = colors[traj.subject]
        for view, ax in axes.items():
            px, py = projection_xy(xyz, view)
            ax.plot(px, py, color=color, linewidth=2.2, alpha=0.95, solid_capstyle="round")
            ax.scatter(px, py, s=13, color=color, alpha=0.86, edgecolors="none")
            entry_xy = projection_xy(traj.entry[None, :], view)
            tip_xy = projection_xy(traj.tip[None, :], view)
            ax.scatter(
                entry_xy[0],
                entry_xy[1],
                s=48,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.7,
                marker="o",
            )
            ax.scatter(
                tip_xy[0],
                tip_xy[1],
                s=54,
                color=[color],
                marker="X",
                linewidths=0.6,
                edgecolors="black",
            )
            if show_labels and view == "sagittal_left":
                ax.text(
                    float(tip_xy[0][0]) + 1.5,
                    float(tip_xy[1][0]) + 1.5,
                    traj.subject,
                    color=color,
                    fontsize=8.5,
                    weight="bold",
                )

    if not points.empty:
        for subject, group in points.groupby(points["subject"].astype(str), sort=False):
            group = sort_contacts(group)
            xyz = finite_xyz(group)
            color = colors.get(subject, "tab:blue")
            for view, ax in axes.items():
                px, py = projection_xy(xyz, view)
                if len(px) > 1:
                    ax.plot(px, py, color=color, linewidth=1.6, alpha=0.78)
                ax.scatter(
                    px,
                    py,
                    s=24,
                    color=color,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.25,
                    label=subject,
                )
                if show_labels and view == "sagittal_left":
                    label_x, label_y = px[-1], py[-1]
                    ax.text(label_x + 1.5, label_y + 1.5, subject, color=color, fontsize=8.5)

    axes["sagittal_left"].set_xlim(-95, 60)
    axes["sagittal_left"].set_ylim(-50, 85)
    axes["axial"].set_xlim(-85, 25)
    axes["axial"].set_ylim(-85, 45)
    axes["coronal"].set_xlim(-85, 25)
    axes["coronal"].set_ylim(-50, 85)

    handles = [
        plt.Line2D([0], [0], color=colors[subject], lw=2.6, marker="o", markersize=5)
        for subject in subjects
    ]
    fig.legend(
        handles,
        subjects,
        loc="lower center",
        ncol=min(len(subjects), 5),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(title, fontsize=15, y=0.975)
    fig.text(
        0.5,
        0.012,
        "Background: nilearn MNI152 2 mm template maximum-intensity projections. "
        "Open circles mark entries; X marks tips. Verify source coordinates before scientific use.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#3d3d3d",
    )

    requested_formats = {fmt.strip().lower() for fmt in formats if fmt.strip()}
    unsupported = requested_formats.difference({"png", "pdf"})
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {', '.join(sorted(unsupported))}")

    png_path = output_dir / f"{prefix}.png"
    pdf_path = output_dir / f"{prefix}.pdf"
    points_path = output_dir / f"{prefix}_display_points.csv"
    if "png" in requested_formats:
        fig.savefig(png_path, bbox_inches="tight")
    if "pdf" in requested_formats:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    derived_frames = []
    if not points.empty:
        point_out = points.copy()
        point_out["sample_index"] = np.nan
        derived_frames.append(point_out)
    derived_frames.extend(sampled_frames)
    if derived_frames:
        pd.concat(derived_frames, ignore_index=True, sort=False).to_csv(points_path, index=False)
    else:
        pd.DataFrame().to_csv(points_path, index=False)

    return png_path, pdf_path, points_path


def main() -> None:
    args = parse_args()
    points, trajectories = load_rows(args.input)
    png_path, pdf_path, points_path = plot_map(
        points=points,
        trajectories=trajectories,
        output_dir=args.output_dir,
        prefix=args.prefix,
        samples_per_trajectory=args.samples_per_trajectory,
        title=args.title,
        show_labels=args.show_labels,
        formats=args.formats.split(","),
    )
    for fmt in {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}:
        if fmt == "png":
            print(f"Wrote {png_path}")
        elif fmt == "pdf":
            print(f"Wrote {pdf_path}")
    print(f"Wrote {points_path}")


if __name__ == "__main__":
    main()
