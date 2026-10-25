#!/usr/bin/env python3
"""Plot Neuropixels surface entry locations on an fsaverage cortical surface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nilearn import datasets, surface
from scipy.spatial import cKDTree

from map_neuropixels_to_mni152 import CONTACT_COLUMNS, TRAJECTORY_COLUMNS, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Neuropixels surface locations on an fsaverage cortical surface."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "neuropixel2025/data/electrode_mni/"
            "neuropixels_mni152_approx_from_screenshots.csv"
        ),
        help="CSV with trajectory entry/tip coordinates or contact MNI coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for output figure and projected coordinate table.",
    )
    parser.add_argument(
        "--prefix",
        default="neuropixels_mni152_surface_locations",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--title",
        default="Neuropixels surface locations on cortical surface",
        help="Figure title.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats. Supported: png,pdf.",
    )
    parser.add_argument(
        "--mesh",
        default="fsaverage5",
        help="Nilearn fsaverage mesh name, e.g. fsaverage5.",
    )
    parser.add_argument(
        "--surface-kind",
        choices=("pial", "infl", "white"),
        default="pial",
        help="Cortical surface geometry to draw.",
    )
    parser.add_argument(
        "--view",
        choices=("lateral", "oblique"),
        default="lateral",
        help="Camera view for the left hemisphere.",
    )
    parser.add_argument(
        "--light-background",
        action="store_true",
        help="Use a white background instead of the default black figure style.",
    )
    parser.add_argument(
        "--marker-offset-mm",
        type=float,
        default=5.0,
        help="Move markers outward along left-lateral x for visibility only.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw subject labels next to each marker.",
    )
    return parser.parse_args()


def supported_formats(formats: str) -> set[str]:
    requested = {fmt.strip().lower() for fmt in formats.split(",") if fmt.strip()}
    unsupported = requested.difference({"png", "pdf"})
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {', '.join(sorted(unsupported))}")
    return requested


def make_cortical_surface(mesh: str, surface_kind: str) -> tuple[np.ndarray, np.ndarray]:
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    key = {
        "pial": "pial_left",
        "infl": "infl_left",
        "white": "white_left",
    }[surface_kind]
    surf_mesh = surface.load_surf_mesh(fsaverage[key])
    return surf_mesh.coordinates.astype(float), surf_mesh.faces.astype(int)


def collect_surface_points(points: pd.DataFrame, trajectories) -> pd.DataFrame:
    rows = []
    for traj in trajectories:
        rows.append(
            {
                "subject": traj.subject,
                "source_x": traj.entry[0],
                "source_y": traj.entry[1],
                "source_z": traj.entry[2],
                "coordinate_source": "entry",
                "source": traj.source,
                "confidence": traj.confidence,
                "note": traj.note,
            }
        )

    if not trajectories and not points.empty:
        for _, row in points.iterrows():
            rows.append(
                {
                    "subject": str(row["subject"]),
                    "source_x": row["mni_x"],
                    "source_y": row["mni_y"],
                    "source_z": row["mni_z"],
                    "coordinate_source": "mni_contact",
                    "source": str(row.get("source", "")),
                    "confidence": str(row.get("confidence", "")),
                    "note": str(row.get("note", "")),
                }
            )

    if not rows:
        raise ValueError(
            "No surface points found. Provide trajectory entry coordinates "
            f"({', '.join(TRAJECTORY_COLUMNS[:3])}) or contact coordinates "
            f"({', '.join(CONTACT_COLUMNS)})."
        )
    return pd.DataFrame(rows)


def project_to_surface(
    surface_points: pd.DataFrame,
    verts: np.ndarray,
    marker_offset_mm: float = 5.0,
) -> pd.DataFrame:
    tree = cKDTree(verts)
    source_xyz = surface_points.loc[:, ["source_x", "source_y", "source_z"]].to_numpy(dtype=float)
    distances, indices = tree.query(source_xyz)
    projected = surface_points.copy()
    projected[["surface_x", "surface_y", "surface_z"]] = verts[indices]
    projected["surface_projection_distance_mm"] = distances
    projected["display_x"] = projected["surface_x"] - marker_offset_mm
    projected["display_y"] = projected["surface_y"]
    projected["display_z"] = projected["surface_z"]
    projected["display_marker_offset_mm"] = marker_offset_mm
    return projected


def set_equal_axes(ax, verts: np.ndarray, zoom: float = 0.72) -> None:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = float(np.max(maxs - mins) * zoom / 2)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def surface_facecolors(
    verts: np.ndarray,
    faces: np.ndarray,
    dark_background: bool,
) -> np.ndarray:
    triangles = verts[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9

    light = np.array([-0.65, -0.3, 0.7], dtype=float)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normals @ light, 0, 1)
    shade = 0.42 + 0.58 * diffuse
    base = np.array([0.82, 0.84, 0.84]) if dark_background else np.array([0.74, 0.75, 0.74])
    colors = np.clip(base[None, :] * shade[:, None] + 0.08, 0, 1)
    alpha = np.full((colors.shape[0], 1), 1.0)
    return np.hstack([colors, alpha])


def view_angles(view: str) -> tuple[float, float, float]:
    if view == "lateral":
        return 0.0, 180.0, 0.0
    return 10.0, 210.0, 0.0


def draw_surface(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    projected: pd.DataFrame,
    view: str,
    dark_background: bool,
    show_labels: bool,
) -> None:
    facecolors = surface_facecolors(verts, faces, dark_background=dark_background)
    mesh = Poly3DCollection(
        verts[faces],
        facecolors=facecolors,
        edgecolors=(0.5, 0.5, 0.5, 0.04),
        linewidth=0.04,
        zsort="average",
    )
    ax.add_collection3d(mesh)

    subjects = list(projected["subject"].astype(str))
    cmap = plt.get_cmap("tab10")
    colors = {subject: cmap(idx % 10) for idx, subject in enumerate(subjects)}

    label_offsets_yz = [
        (0.0, 10.0),
        (-7.0, 7.5),
        (8.0, 3.0),
        (5.5, -7.0),
        (-9.0, -5.5),
    ]
    for idx, row in projected.reset_index(drop=True).iterrows():
        subject = str(row["subject"])
        color = colors[subject]
        label_dy, label_dz = label_offsets_yz[idx % len(label_offsets_yz)]
        marker_x = row["display_x"]
        marker_y = row["display_y"]
        marker_z = row["display_z"]
        label_x = marker_x - 2.0
        label_y = marker_y + label_dy
        label_z = marker_z + label_dz
        ax.scatter(
            marker_x,
            marker_y,
            marker_z,
            s=140,
            color=color,
            edgecolors="black",
            linewidths=0.9,
            depthshade=False,
            label=subject,
        )
        ax.plot(
            [row["surface_x"], marker_x],
            [row["surface_y"], marker_y],
            [row["surface_z"], marker_z],
            color=color,
            linewidth=0.8,
            alpha=0.72,
        )
        if show_labels:
            ax.plot(
                [marker_x, label_x],
                [marker_y, label_y],
                [marker_z, label_z],
                color=color,
                linewidth=0.8,
                alpha=0.72,
            )
            ax.text(
                label_x,
                label_y,
                label_z,
                subject,
                color=color,
                fontsize=10.5,
                weight="bold",
                ha="center",
                va="bottom",
                path_effects=[
                    pe.withStroke(
                        linewidth=2.5,
                        foreground="#111111" if dark_background else "#ffffff",
                    )
                ],
            )

    elev, azim, roll = view_angles(view)
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    set_equal_axes(ax, verts)
    ax.set_axis_off()
    ax.set_facecolor("#171717" if dark_background else "#ffffff")


def plot_surface_locations(
    projected: pd.DataFrame,
    verts: np.ndarray,
    faces: np.ndarray,
    output_dir: Path,
    prefix: str,
    title: str,
    formats: Iterable[str],
    view: str,
    dark_background: bool,
    show_labels: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(formats)

    bg = "#171717" if dark_background else "#ffffff"
    text = "#eeeeee" if dark_background else "#222222"
    fig = plt.figure(figsize=(12.4, 8.4), dpi=170, facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    draw_surface(
        ax,
        verts,
        faces,
        projected,
        view=view,
        dark_background=dark_background,
        show_labels=show_labels,
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.93)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        frameon=False,
        fontsize=10,
        labelcolor=text,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.suptitle(title, fontsize=16, y=0.965, color=text)
    fig.text(
        0.5,
        0.018,
        "Surface: nilearn fsaverage5 left pial cortical mesh. "
        "Markers are entry/source points projected to the nearest pial vertex.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#cfcfcf" if dark_background else "#3d3d3d",
    )

    png_path = output_dir / f"{prefix}.png"
    pdf_path = output_dir / f"{prefix}.pdf"
    csv_path = output_dir / f"{prefix}_projected_surface_points.csv"
    if "png" in requested:
        fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    if "pdf" in requested:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    projected.to_csv(csv_path, index=False)
    return png_path, pdf_path, csv_path


def main() -> None:
    args = parse_args()
    requested = supported_formats(args.formats)
    points, trajectories = load_rows(args.input)
    surface_points = collect_surface_points(points, trajectories)
    verts, faces = make_cortical_surface(args.mesh, args.surface_kind)
    projected = project_to_surface(
        surface_points,
        verts,
        marker_offset_mm=args.marker_offset_mm,
    )
    png_path, pdf_path, csv_path = plot_surface_locations(
        projected=projected,
        verts=verts,
        faces=faces,
        output_dir=args.output_dir,
        prefix=args.prefix,
        title=args.title,
        formats=requested,
        view=args.view,
        dark_background=not args.light_background,
        show_labels=args.show_labels,
    )
    if "png" in requested:
        print(f"Wrote {png_path}")
    if "pdf" in requested:
        print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
