#!/usr/bin/env python3
"""Conservative multi-feature probe tissue-boundary candidate analysis.

The analysis detects stable transitions in probe coordinates.  It deliberately
does not assign pia, white matter, cortical layer, or cell type without an
independent anatomical trajectory anchor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/probe_tissue_boundary_candidates_20260730"
FIGURES = OUTPUT / "figures"
SOURCE = OUTPUT / "source_data"
MAPS = OUTPUT / "boundary_maps"
QA = OUTPUT / "qa"
CSD_OUTPUT = ROOT / "reports/csd_spike_l5_candidates_20260730"
UNITS_PATH = (
    ROOT
    / "reports/multirun_grouped_go_20260713/nfo/formal/"
    "392b8a2dbb2018ae415f2d279a0c7f653bfcc76816ffcd8a0241aeeb40626ea6/"
    "delivery/source_data/nfo_units.csv"
)

N_LF_WINDOWS = 16
LF_WINDOW_SEC = 1.0
N_AP_WINDOWS = 12
AP_WINDOW_SEC = 0.50
N_BOOTSTRAP = 240
SEED = 20260730
KERNEL_UM = 120.0
MIN_PEAK_SEPARATION_UM = 600.0
MATCH_RADIUS_UM = 400.0
SUPPORT_RADIUS_UM = 200.0
MIN_EDGE_DISTANCE_UM = 200.0
MIN_FONT_PT = 10.0

DARK = "#20242A"
GREY = "#747A80"
LIGHT = "#D8DDE2"
FIELD = "#4C78A8"
SPIKING = "#E07A5F"
UNIT = "#7A5195"
CSD = "#C94C3B"
ENCODED = "#0A8F6A"
GOLD = "#D9A326"

CANDIDATE_COLUMNS = [
    "candidate",
    "probe_y_um",
    "relative_probe_coordinate",
    "composite_score",
    "bootstrap_support_within_200um",
    "bootstrap_ci_low_um",
    "bootstrap_ci_high_um",
    "bootstrap_ci_width_um",
    "feature_group_agreement_out_of_3",
    "cross_boundary_level_contrast",
    "edge_distance_um",
    "status",
    "anatomical_assignment",
]


@dataclass(frozen=True)
class RunConfig:
    patient: str
    run: str
    label: str

    @property
    def task_dir(self) -> Path:
        return ROOT / f"neuropixel2025/data/{self.patient}/processed/ActPasSynSentence"

    @property
    def run_dir(self) -> Path:
        return self.task_dir / self.run

    @property
    def ks_dir(self) -> Path:
        return self.run_dir / "kilosort4"


RUNS = (
    RunConfig("gz001", "gz001t_g0_actpas_run1", "gz001 run1"),
    RunConfig("gz002", "gz002_g0_actpas_run2", "gz002 run2"),
    RunConfig("gz003", "gz003t2_g0_actpas_run1", "gz003 run1"),
    RunConfig("gz003", "gz003t2_g0_actpas_run4", "gz003 run4"),
    RunConfig("gz004", "gz004t1_g0_actpas_run0", "gz004 run0"),
    RunConfig("gz004", "gz004t1_g3_actpas_run3", "gz004 run3"),
    RunConfig("gz005", "gz005t0_g0_actpas_run0", "gz005 run0"),
    RunConfig("gz005", "gz005t0_g3_actpas_run3", "gz005 run3"),
)
PATIENTS = tuple(dict.fromkeys(run.patient for run in RUNS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": MIN_FONT_PT,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_meta(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(errors="ignore").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def binary_paths(config: RunConfig, band: str) -> tuple[Path, Path]:
    meta = next(config.run_dir.glob(f"*imec0/*.{band}.meta"))
    binary = meta.with_suffix(".bin")
    if not binary.exists():
        candidate = ROOT / parse_meta(meta).get("fileName", "")
        if candidate.exists():
            binary = candidate
    if not binary.exists():
        raise FileNotFoundError(binary)
    return binary, meta


def uv_per_bit(meta: dict[str, str], band: str) -> float:
    gain = float(meta[f"imChan0{band}Gain"])
    return (
        float(meta.get("imAiRangeMax", "0.6"))
        / float(meta.get("imMaxInt", "512"))
        / gain
        * 1e6
    )


def window_starts(n_samples: int, fs: float, duration: float, n_windows: int) -> np.ndarray:
    width = int(round(duration * fs))
    margin = max(width, int(round(2.0 * fs)))
    low = margin
    high = n_samples - margin - width
    if high <= low:
        raise RuntimeError("recording is too short for deterministic window sampling")
    return np.rint(np.linspace(low, high, n_windows)).astype(np.int64)


def depth_index(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths, inverse = np.unique(positions[:, 1].astype(float), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    return depths, inverse, counts


def aggregate_channels(values: np.ndarray, inverse: np.ndarray, counts: np.ndarray) -> np.ndarray:
    result = np.zeros(len(counts), dtype=float)
    np.add.at(result, inverse, np.asarray(values, dtype=float))
    return result / counts


def average_signal_by_depth(
    channel_time: np.ndarray, inverse: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(counts), channel_time.shape[1]), dtype=np.float32)
    for channel, depth_bin in enumerate(inverse):
        result[depth_bin] += channel_time[channel]
    result /= counts[:, None]
    return result


def local_depth_correlation(signal: np.ndarray) -> np.ndarray:
    centered = signal - signal.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
    normalized = centered / np.maximum(scale, 1e-12)
    pair = np.sum(normalized[:-1] * normalized[1:], axis=1)
    output = np.empty(signal.shape[0], dtype=float)
    output[0] = pair[0]
    output[-1] = pair[-1]
    if len(output) > 2:
        output[1:-1] = 0.5 * (pair[:-1] + pair[1:])
    return np.clip(output, -1.0, 1.0)


def sample_lf(config: RunConfig, positions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    binary, meta_path = binary_paths(config, "lf")
    meta = parse_meta(meta_path)
    fs = float(meta["imSampRate"])
    n_saved = int(meta["nSavedChans"])
    n_signal = int(meta.get("snsApLfSy", "0,384,1").split(",")[1])
    n_samples = binary.stat().st_size // (2 * n_saved)
    data = np.memmap(binary, dtype=np.int16, mode="r", shape=(n_samples, n_saved))
    depths, inverse, counts = depth_index(positions[:n_signal])
    starts = window_starts(n_samples, fs, LF_WINDOW_SEC, N_LF_WINDOWS)
    width = int(round(LF_WINDOW_SEC * fs))
    taper = np.hanning(width).astype(np.float32)
    frequencies = np.fft.rfftfreq(width, 1.0 / fs)
    bands = {
        "lf_low_power": (frequencies >= 1.0) & (frequencies < 30.0),
        "lf_gamma_power": (frequencies >= 30.0) & (frequencies <= 100.0),
    }
    features = {
        "lf_rms": [],
        "lf_low_power": [],
        "lf_gamma_power": [],
        "lf_local_correlation": [],
    }
    scale = uv_per_bit(meta, "lf")
    for start in starts:
        chunk = data[start : start + width, :n_signal].astype(np.float32) * scale
        chunk -= np.median(chunk, axis=1, keepdims=True)
        depth_signal = average_signal_by_depth(chunk.T, inverse, counts)
        depth_signal -= depth_signal.mean(axis=1, keepdims=True)
        features["lf_rms"].append(np.sqrt(np.mean(depth_signal**2, axis=1)))
        spectrum = np.fft.rfft(depth_signal * taper[None, :], axis=1)
        power = np.abs(spectrum) ** 2
        for name, mask in bands.items():
            features[name].append(np.mean(power[:, mask], axis=1))
        features["lf_local_correlation"].append(local_depth_correlation(depth_signal))
    del data
    return depths, {name: np.asarray(values, dtype=np.float32) for name, values in features.items()}


def sample_ap(config: RunConfig, positions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    binary, meta_path = binary_paths(config, "ap")
    meta = parse_meta(meta_path)
    fs = float(meta["imSampRate"])
    n_saved = int(meta["nSavedChans"])
    n_signal = int(meta.get("snsApLfSy", "384,0,1").split(",")[0])
    n_samples = binary.stat().st_size // (2 * n_saved)
    duration = n_samples / fs
    data = np.memmap(binary, dtype=np.int16, mode="r", shape=(n_samples, n_saved))
    depths, inverse, counts = depth_index(positions[:n_signal])
    starts = window_starts(n_samples, fs, AP_WINDOW_SEC, N_AP_WINDOWS)
    width = int(round(AP_WINDOW_SEC * fs))
    features = {"ap_rms": [], "ap_threshold_crossing_rate": []}
    scale = uv_per_bit(meta, "ap")
    for start in starts:
        chunk = data[start : start + width, :n_signal].astype(np.float32) * scale
        chunk -= np.median(chunk, axis=1, keepdims=True)
        channel_time = chunk.T
        channel_time -= np.median(channel_time, axis=1, keepdims=True)
        rms = np.sqrt(np.mean(channel_time**2, axis=1))
        noise = 1.4826 * np.median(np.abs(channel_time), axis=1)
        threshold = -4.0 * np.maximum(noise, 1e-6)
        crossings = np.sum(
            (channel_time[:, 1:] < threshold[:, None])
            & (channel_time[:, :-1] >= threshold[:, None]),
            axis=1,
        ) / AP_WINDOW_SEC
        features["ap_rms"].append(aggregate_channels(rms, inverse, counts))
        features["ap_threshold_crossing_rate"].append(
            aggregate_channels(crossings, inverse, counts)
        )
    del data
    return depths, {name: np.asarray(values, dtype=np.float32) for name, values in features.items()}, duration


def gaussian_depth_profile(
    target_depths: np.ndarray,
    unit_depths: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(unit_depths), dtype=float)
    step = float(np.median(np.diff(target_depths)))
    edges = np.r_[target_depths - step / 2.0, target_depths[-1] + step / 2.0]
    values, _ = np.histogram(unit_depths, bins=edges, weights=weights)
    return gaussian_filter1d(values.astype(float), KERNEL_UM / step, mode="nearest")


def waveform_metrics(config: RunConfig, units: pd.DataFrame, ap_fs: float) -> pd.DataFrame:
    templates = np.load(config.ks_dir / "templates.npy", mmap_mode="r")
    rows: list[dict[str, float | int]] = []
    for unit_id in units["unit_id"].astype(int):
        if unit_id < 0 or unit_id >= templates.shape[0]:
            rows.append({"unit_id": unit_id, "waveform_trough_to_peak_ms": np.nan, "template_peak_to_peak": np.nan})
            continue
        template = np.asarray(templates[unit_id], dtype=float)
        channel = int(np.argmax(np.ptp(template, axis=0)))
        waveform = template[:, channel]
        trough = int(np.argmin(waveform))
        if trough >= len(waveform) - 2:
            duration_ms = np.nan
        else:
            peak = trough + int(np.argmax(waveform[trough:]))
            duration_ms = (peak - trough) / ap_fs * 1000.0
        rows.append(
            {
                "unit_id": unit_id,
                "waveform_trough_to_peak_ms": duration_ms,
                "template_peak_to_peak": float(np.ptp(waveform)),
            }
        )
    return pd.DataFrame(rows)


def unit_features(
    config: RunConfig,
    units: pd.DataFrame,
    depths: np.ndarray,
    duration_sec: float,
    ap_fs: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    clusters = np.load(config.ks_dir / "spike_clusters.npy", mmap_mode="r")
    ids, counts = np.unique(np.asarray(clusters, dtype=np.int64), return_counts=True)
    count_table = pd.DataFrame({"unit_id": ids, "n_spikes": counts})
    amplitudes = pd.read_csv(config.ks_dir / "cluster_Amplitude.tsv", sep="\t").rename(
        columns={"cluster_id": "unit_id", "Amplitude": "amplitude_uv"}
    )
    waveforms = waveform_metrics(config, units, ap_fs)
    table = (
        units.merge(count_table, on="unit_id", how="left")
        .merge(amplitudes[["unit_id", "amplitude_uv"]], on="unit_id", how="left")
        .merge(waveforms, on="unit_id", how="left")
    )
    if table[["n_spikes", "amplitude_uv"]].isna().any().any():
        raise RuntimeError(f"missing spike metrics for {config.run}")
    table["firing_rate_hz"] = table["n_spikes"] / duration_sec
    table["condition_sensitive_unit"] = table["primary_degree"].astype(int) > 0
    unit_depths = table["absolute_depth_um"].to_numpy(float)
    features = {
        "unit_density": gaussian_depth_profile(depths, unit_depths),
        "spike_rate_density": gaussian_depth_profile(
            depths, unit_depths, table["firing_rate_hz"].to_numpy(float)
        ),
        "encoded_unit_density": gaussian_depth_profile(
            depths, unit_depths, table["condition_sensitive_unit"].to_numpy(float)
        ),
        "amplitude_weighted_density": gaussian_depth_profile(
            depths, unit_depths, table["amplitude_uv"].to_numpy(float)
        ),
    }
    return table, features


def robust_level(values: np.ndarray, log_transform: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if log_transform:
        array = np.log1p(np.maximum(array, 0.0))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    low, high = np.percentile(finite, [5.0, 95.0])
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def change_score(values: np.ndarray, sigma: float = 2.0, log_transform: bool = True) -> np.ndarray:
    level = robust_level(values, log_transform=log_transform)
    smooth = gaussian_filter1d(level, sigma, mode="nearest")
    derivative = np.abs(np.gradient(smooth))
    derivative = gaussian_filter1d(derivative, 1.0, mode="nearest")
    maximum = float(np.nanmax(derivative))
    return derivative / maximum if maximum > 0 else np.zeros_like(derivative)


def group_scores(
    lf_profiles: dict[str, np.ndarray],
    ap_profiles: dict[str, np.ndarray],
    unit_profiles: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    lf_median = {name: np.median(values, axis=0) for name, values in lf_profiles.items()}
    ap_median = {name: np.median(values, axis=0) for name, values in ap_profiles.items()}
    field_level = np.mean(
        [
            robust_level(lf_median["lf_rms"]),
            robust_level(lf_median["lf_low_power"]),
            robust_level(lf_median["lf_gamma_power"]),
            robust_level(lf_median["lf_local_correlation"], log_transform=False),
        ],
        axis=0,
    )
    spiking_level = np.mean(
        [
            robust_level(ap_median["ap_rms"]),
            robust_level(ap_median["ap_threshold_crossing_rate"]),
        ],
        axis=0,
    )
    unit_level = np.mean(
        [
            robust_level(unit_profiles["unit_density"]),
            robust_level(unit_profiles["spike_rate_density"]),
        ],
        axis=0,
    )
    changes = {
        "field_change_score": np.mean(
            [
                change_score(lf_median["lf_rms"]),
                change_score(lf_median["lf_low_power"]),
                change_score(lf_median["lf_gamma_power"]),
                change_score(lf_median["lf_local_correlation"], log_transform=False),
            ],
            axis=0,
        ),
        "spiking_change_score": np.mean(
            [
                change_score(ap_median["ap_rms"]),
                change_score(ap_median["ap_threshold_crossing_rate"]),
            ],
            axis=0,
        ),
        "unit_change_score": np.mean(
            [
                change_score(unit_profiles["unit_density"]),
                change_score(unit_profiles["spike_rate_density"]),
            ],
            axis=0,
        ),
    }
    composite = np.mean(list(changes.values()), axis=0)
    composite /= max(float(np.max(composite)), 1e-12)
    changes["composite_change_score"] = composite
    levels = {
        "field_level": field_level,
        "spiking_level": spiking_level,
        "unit_level": unit_level,
        "encoded_unit_density_level": robust_level(unit_profiles["encoded_unit_density"]),
        "amplitude_level": robust_level(unit_profiles["amplitude_weighted_density"]),
    }
    return levels, changes


def interior_peak_indices(score: np.ndarray, depths: np.ndarray, maximum: int = 2) -> np.ndarray:
    step = float(np.median(np.diff(depths)))
    edge_bins = max(1, int(np.ceil(MIN_EDGE_DISTANCE_UM / step)))
    distance_bins = max(1, int(np.ceil(MIN_PEAK_SEPARATION_UM / step)))
    interior = score.copy()
    interior[:edge_bins] = 0.0
    interior[-edge_bins:] = 0.0
    peaks, properties = find_peaks(interior, distance=distance_bins, prominence=0.06)
    if len(peaks) == 0:
        return np.array([], dtype=int)
    quality = interior[peaks] + properties["prominences"]
    order = np.argsort(quality)[::-1][:maximum]
    return np.sort(peaks[order])


def side_contrast(level: np.ndarray, index: int, radius_bins: int) -> float:
    left = level[max(0, index - 2 * radius_bins) : max(0, index - radius_bins)]
    right = level[min(len(level), index + radius_bins) : min(len(level), index + 2 * radius_bins)]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    return float(abs(np.median(left) - np.median(right)))


def bootstrap_boundaries(
    depths: np.ndarray,
    lf_profiles: dict[str, np.ndarray],
    ap_profiles: dict[str, np.ndarray],
    unit_table: pd.DataFrame,
    full_levels: dict[str, np.ndarray],
    full_changes: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    full_candidates = interior_peak_indices(full_changes["composite_change_score"], depths)
    probability_counts = np.zeros(len(depths), dtype=float)
    matched: dict[int, list[float]] = {int(index): [] for index in full_candidates}
    step = float(np.median(np.diff(depths)))
    match_bins = max(1, int(round(MATCH_RADIUS_UM / step)))
    contrast_bins = max(1, int(round(200.0 / step)))
    unit_depths_all = unit_table["absolute_depth_um"].to_numpy(float)
    rates_all = unit_table["firing_rate_hz"].to_numpy(float)
    encoded_all = unit_table["condition_sensitive_unit"].to_numpy(float)
    amplitudes_all = unit_table["amplitude_uv"].to_numpy(float)
    for _ in range(N_BOOTSTRAP):
        lf_idx = rng.integers(0, len(next(iter(lf_profiles.values()))), size=len(next(iter(lf_profiles.values()))))
        ap_idx = rng.integers(0, len(next(iter(ap_profiles.values()))), size=len(next(iter(ap_profiles.values()))))
        unit_idx = rng.integers(0, len(unit_table), size=len(unit_table))
        boot_lf = {name: values[lf_idx] for name, values in lf_profiles.items()}
        boot_ap = {name: values[ap_idx] for name, values in ap_profiles.items()}
        boot_depths = unit_depths_all[unit_idx]
        boot_units = {
            "unit_density": gaussian_depth_profile(depths, boot_depths),
            "spike_rate_density": gaussian_depth_profile(depths, boot_depths, rates_all[unit_idx]),
            "encoded_unit_density": gaussian_depth_profile(
                depths, boot_depths, encoded_all[unit_idx]
            ),
            "amplitude_weighted_density": gaussian_depth_profile(
                depths, boot_depths, amplitudes_all[unit_idx]
            ),
        }
        _, changes = group_scores(boot_lf, boot_ap, boot_units)
        peaks = interior_peak_indices(changes["composite_change_score"], depths)
        possible_matches = sorted(
            (
                (abs(int(peak) - int(candidate)), int(candidate), int(peak))
                for candidate in full_candidates
                for peak in peaks
                if abs(int(peak) - int(candidate)) <= match_bins
            ),
            key=lambda item: item[0],
        )
        used_candidates: set[int] = set()
        used_peaks: set[int] = set()
        for _, candidate, peak in possible_matches:
            if candidate in used_candidates or peak in used_peaks:
                continue
            used_candidates.add(candidate)
            used_peaks.add(peak)
            matched[candidate].append(float(depths[peak]))
            probability_counts[peak] += 1.0 / N_BOOTSTRAP
    probability = gaussian_filter1d(probability_counts, 1.5, mode="nearest")
    rows: list[dict[str, object]] = []
    group_names = ("field_change_score", "spiking_change_score", "unit_change_score")
    group_peaks = {
        name: interior_peak_indices(full_changes[name], depths, maximum=4) for name in group_names
    }
    for ordinal, candidate in enumerate(full_candidates, start=1):
        locations = np.asarray(matched[int(candidate)], dtype=float)
        support = float(
            np.sum(np.abs(locations - depths[candidate]) <= SUPPORT_RADIUS_UM)
            / N_BOOTSTRAP
        )
        ci_low, ci_high = (
            np.percentile(locations, [2.5, 97.5]) if len(locations) >= 10 else (np.nan, np.nan)
        )
        agreement = sum(
            bool(len(group_peaks[name]))
            and float(np.min(np.abs(depths[group_peaks[name]] - depths[candidate]))) <= 300.0
            for name in group_names
        )
        contrast = float(
            np.mean(
                [
                    side_contrast(full_levels["field_level"], int(candidate), contrast_bins),
                    side_contrast(full_levels["spiking_level"], int(candidate), contrast_bins),
                    side_contrast(full_levels["unit_level"], int(candidate), contrast_bins),
                ]
            )
        )
        ci_width = float(ci_high - ci_low) if np.isfinite(ci_low) else np.inf
        score = float(full_changes["composite_change_score"][candidate])
        edge_distance = float(min(depths[candidate] - depths.min(), depths.max() - depths[candidate]))
        if support >= 0.60 and agreement >= 2 and contrast >= 0.15 and ci_width <= 600.0 and score >= 0.45 and edge_distance >= MIN_EDGE_DISTANCE_UM:
            status = "supported_probe_boundary"
        elif support >= 0.40 and agreement >= 2 and score >= 0.40:
            status = "tentative_probe_boundary"
        else:
            status = "unsupported_transition"
        rows.append(
            {
                "candidate": f"B{ordinal}",
                "probe_y_um": float(depths[candidate]),
                "relative_probe_coordinate": float(
                    (depths[candidate] - depths.min()) / max(np.ptp(depths), 1.0)
                ),
                "composite_score": score,
                "bootstrap_support_within_200um": support,
                "bootstrap_ci_low_um": float(ci_low),
                "bootstrap_ci_high_um": float(ci_high),
                "bootstrap_ci_width_um": ci_width,
                "feature_group_agreement_out_of_3": int(agreement),
                "cross_boundary_level_contrast": contrast,
                "edge_distance_um": edge_distance,
                "status": status,
                "anatomical_assignment": "unassigned_without_trajectory_anchor",
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), probability


def load_csd_profile(config: RunConfig, depths: np.ndarray) -> np.ndarray:
    table = pd.read_csv(CSD_OUTPUT / "source_data/run_depth_profiles.csv")
    row = table.loc[table["patient"].eq(config.patient) & table["run"].eq(config.run)]
    if row.empty:
        return np.zeros(len(depths), dtype=float)
    values = np.interp(
        depths,
        row["probe_depth_um"].to_numpy(float),
        row["absolute_csd_energy"].to_numpy(float),
        left=np.nan,
        right=np.nan,
    )
    finite = np.isfinite(values)
    if finite.any():
        values[~finite] = np.nanmedian(values[finite])
    return robust_level(values)


def analyze_run(
    config: RunConfig, registry: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"sampling {config.label}", flush=True)
    units = registry.loc[
        registry["patient"].eq(config.patient) & registry["run"].eq(config.run)
    ].copy()
    positions = np.load(config.ks_dir / "channel_positions.npy").astype(float)
    lf_depths, lf_profiles = sample_lf(config, positions)
    ap_depths, ap_profiles, duration_sec = sample_ap(config, positions)
    if not np.array_equal(lf_depths, ap_depths):
        raise RuntimeError(f"AP/LF depth coordinates differ for {config.run}")
    _, ap_meta_path = binary_paths(config, "ap")
    ap_fs = float(parse_meta(ap_meta_path)["imSampRate"])
    unit_table, static_features = unit_features(
        config, units, lf_depths, duration_sec, ap_fs
    )
    levels, changes = group_scores(lf_profiles, ap_profiles, static_features)
    candidates, probability = bootstrap_boundaries(
        lf_depths, lf_profiles, ap_profiles, unit_table, levels, changes, rng
    )
    candidates.insert(0, "run_label", config.label)
    candidates.insert(0, "run", config.run)
    candidates.insert(0, "patient", config.patient)
    supported = candidates.loc[candidates["status"].eq("supported_probe_boundary")]
    can_normalize = len(supported) == 2
    lower = float(supported["probe_y_um"].min()) if can_normalize else np.nan
    upper = float(supported["probe_y_um"].max()) if can_normalize else np.nan
    unit_table.insert(0, "run_label", config.label)
    unit_table["boundary_coordinate_available"] = can_normalize
    unit_table["boundary_fraction_low_to_high_probe_y"] = (
        (unit_table["absolute_depth_um"] - lower) / (upper - lower)
        if can_normalize
        else np.nan
    )
    unit_table["inside_supported_boundary_span"] = (
        unit_table["absolute_depth_um"].between(lower, upper) if can_normalize else False
    )
    if len(candidates):
        candidate_depths = candidates["probe_y_um"].to_numpy(float)
        unit_table["distance_to_nearest_probe_boundary_candidate_um"] = np.min(
            np.abs(unit_table["absolute_depth_um"].to_numpy(float)[:, None] - candidate_depths[None, :]),
            axis=1,
        )
    else:
        unit_table["distance_to_nearest_probe_boundary_candidate_um"] = np.nan
    csd_level = load_csd_profile(config, lf_depths)
    profile = pd.DataFrame(
        {
            "patient": config.patient,
            "run": config.run,
            "run_label": config.label,
            "probe_y_um": lf_depths,
            **levels,
            **changes,
            "boundary_bootstrap_localization_probability": probability,
            "csd_absolute_energy_level": csd_level,
        }
    )
    np.savez_compressed(
        MAPS / f"{config.patient}_{config.run}_probe_boundary_map.npz",
        probe_y_um=lf_depths,
        boundary_bootstrap_localization_probability=probability,
        **levels,
        **changes,
    )
    statuses = candidates["status"].value_counts().to_dict()
    summary = {
        "patient": config.patient,
        "run": config.run,
        "run_label": config.label,
        "n_registered_units": int(len(unit_table)),
        "n_condition_sensitive_units": int(unit_table["condition_sensitive_unit"].sum()),
        "probe_y_min_um": float(lf_depths.min()),
        "probe_y_max_um": float(lf_depths.max()),
        "n_candidates": int(len(candidates)),
        "n_supported_probe_boundaries": int(statuses.get("supported_probe_boundary", 0)),
        "n_tentative_probe_boundaries": int(statuses.get("tentative_probe_boundary", 0)),
        "two_boundary_coordinate_available": bool(can_normalize),
        "histological_pia_identifiable": False,
        "histological_white_matter_identifiable": False,
        "cortical_layer_identifiable": False,
    }
    print(
        f"{config.label}: supported={summary['n_supported_probe_boundaries']}, "
        f"tentative={summary['n_tentative_probe_boundaries']}, "
        f"two-boundary={can_normalize}",
        flush=True,
    )
    return summary, candidates, profile, unit_table


def plot_patient(
    patient: str,
    profiles: pd.DataFrame,
    candidates: pd.DataFrame,
    units: pd.DataFrame,
) -> plt.Figure:
    run_labels = profiles.loc[profiles["patient"].eq(patient), "run_label"].drop_duplicates().tolist()
    n_rows = len(run_labels)
    height_mm = 120.0 if n_rows == 1 else 185.0
    fig = plt.figure(figsize=(180 / 25.4, height_mm / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        n_rows,
        3,
        left=0.10,
        right=0.97,
        bottom=0.20 if n_rows == 1 else 0.15,
        top=0.66 if n_rows == 1 else 0.70,
        width_ratios=(1.0, 0.88, 1.08),
        hspace=0.48,
        wspace=0.36,
    )
    fig.text(0.05, 0.965, f"{patient} | Probe-boundary candidates", fontsize=13, fontweight="bold", va="top", color=DARK)
    fig.text(
        0.05,
        0.905,
        "Stable physiological transitions in probe coordinates; anatomical identity remains unassigned.\n"
        "Probe y=0 is tipward; increasing y is baseward.",
        fontsize=10,
        color=GREY,
        va="top",
    )
    shared_handles = [
        Line2D([0], [0], color=FIELD, lw=2, label="LF field"),
        Line2D([0], [0], color=SPIKING, lw=2, label="AP/MUA"),
        Line2D([0], [0], color=UNIT, lw=2, label="unit/spike rate"),
        Line2D([0], [0], color=CSD, lw=1.5, ls=":", label="CSD energy"),
    ]
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.790),
        ncol=4,
        fontsize=MIN_FONT_PT,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    for row_index, run_label in enumerate(run_labels):
        profile = profiles.loc[profiles["run_label"].eq(run_label)].sort_values("probe_y_um")
        run_units = units.loc[units["run_label"].eq(run_label)]
        run_candidates = candidates.loc[candidates["run_label"].eq(run_label)]
        depth = profile["probe_y_um"].to_numpy(float)
        ax_levels = fig.add_subplot(grid[row_index, 0])
        ax_boundary = fig.add_subplot(grid[row_index, 1], sharey=ax_levels)
        ax_units = fig.add_subplot(grid[row_index, 2], sharey=ax_levels)
        ax_levels.plot(profile["field_level"], depth, color=FIELD, lw=1.8, label="LF field")
        ax_levels.plot(profile["spiking_level"], depth, color=SPIKING, lw=1.8, label="AP/MUA")
        ax_levels.plot(profile["unit_level"], depth, color=UNIT, lw=1.8, label="unit/spike rate")
        ax_levels.plot(profile["csd_absolute_energy_level"], depth, color=CSD, lw=1.3, ls=":", label="CSD energy")
        ax_levels.set_xlim(0, 1.05)
        ax_levels.set_ylim(depth.max(), depth.min())
        ax_levels.set_xlabel("Normalized level")
        ax_levels.set_ylabel("Probe y-coordinate (µm)")
        short_run = run_label.split()[-1]
        first_panel = chr(ord("a") + 3 * row_index)
        ax_levels.set_title(f"{first_panel}  {short_run} | signals", loc="left", fontweight="bold")
        ax_boundary.plot(profile["composite_change_score"], depth, color=DARK, lw=2.0, label="change score")
        ax_boundary.fill_betweenx(depth, 0, profile["boundary_bootstrap_localization_probability"], color=GOLD, alpha=0.38, lw=0, label="bootstrap density")
        for candidate in run_candidates.itertuples(index=False):
            if candidate.status == "supported_probe_boundary":
                color, linestyle, alpha = ENCODED, "-", 0.14
            elif candidate.status == "tentative_probe_boundary":
                color, linestyle, alpha = GOLD, "--", 0.10
            else:
                color, linestyle, alpha = GREY, ":", 0.06
            if np.isfinite(candidate.bootstrap_ci_low_um):
                ax_boundary.axhspan(candidate.bootstrap_ci_low_um, candidate.bootstrap_ci_high_um, color=color, alpha=alpha, lw=0)
            ax_boundary.axhline(candidate.probe_y_um, color=color, lw=1.5, ls=linestyle)
            ax_boundary.text(
                1.14,
                candidate.probe_y_um,
                f"{candidate.candidate}\n{candidate.status.split('_')[0]}",
                fontsize=MIN_FONT_PT,
                va="center",
                ha="right",
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
            )
        ax_boundary.set_xlim(0, 1.18)
        ax_boundary.set_xlabel("Change / bootstrap")
        ax_boundary.tick_params(axis="y", labelleft=False)
        ax_boundary.set_title(f"{chr(ord(first_panel) + 1)}  boundary evidence", loc="left", fontweight="bold")
        encoded = run_units["condition_sensitive_unit"].astype(bool).to_numpy()
        amplitude = run_units["amplitude_uv"].to_numpy(float)
        size = 8.0 + 28.0 * robust_level(amplitude)
        waveform = run_units["waveform_trough_to_peak_ms"].to_numpy(float)
        ax_units.scatter(waveform[~encoded], run_units.loc[~encoded, "absolute_depth_um"], s=size[~encoded], color=LIGHT, edgecolor=DARK, linewidth=0.25, alpha=0.65, label="other registered unit")
        ax_units.scatter(waveform[encoded], run_units.loc[encoded, "absolute_depth_um"], s=size[encoded], color=ENCODED, edgecolor="white", linewidth=0.25, alpha=0.80, label="condition-sensitive unit")
        finite_waveform = waveform[np.isfinite(waveform)]
        waveform_limit = (
            float(np.percentile(finite_waveform, 99)) + 0.05
            if finite_waveform.size
            else 1.2
        )
        ax_units.set_xlim(-0.02, max(1.2, waveform_limit))
        ax_units.set_xlabel("Waveform trough-to-peak (ms)")
        ax_units.tick_params(axis="y", labelleft=False)
        ax_units.set_title(f"{chr(ord(first_panel) + 2)}  units | n={len(run_units)}", loc="left", fontweight="bold")
    fig.text(0.05, 0.062, "Green: condition-sensitive (>5 ms; p < 0.05/K, K=conditions); size: Kilosort amplitude.", fontsize=MIN_FONT_PT, color=GREY, ha="left")
    fig.text(0.05, 0.025, "Condition-to-condition only; waveform is descriptive—not a baseline, cell-type or layer test.", fontsize=MIN_FONT_PT, color=GREY, ha="left")
    return fig


def plot_overview(
    run_summary: pd.DataFrame,
    profiles: pd.DataFrame,
    candidates: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(180 / 25.4, 145 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.22, right=0.96, top=0.70, bottom=0.18)
    rows = run_summary.reset_index(drop=True)
    y_positions = np.arange(len(rows))[::-1]
    for y, row in zip(y_positions, rows.itertuples(index=False)):
        profile = profiles.loc[profiles["run"].eq(row.run)].sort_values("probe_y_um")
        x = (profile["probe_y_um"].to_numpy(float) - row.probe_y_min_um) / max(row.probe_y_max_um - row.probe_y_min_um, 1.0)
        score = profile["composite_change_score"].to_numpy(float)
        ax.fill_between(x, y - 0.24, y - 0.24 + 0.42 * score, color=LIGHT, alpha=0.85, lw=0)
        ax.plot(x, y - 0.24 + 0.42 * score, color=DARK, lw=0.9)
        subset = candidates.loc[candidates["run"].eq(row.run)]
        for candidate in subset.itertuples(index=False):
            color = ENCODED if candidate.status == "supported_probe_boundary" else GOLD if candidate.status == "tentative_probe_boundary" else GREY
            marker = "o" if candidate.status == "supported_probe_boundary" else "D" if candidate.status == "tentative_probe_boundary" else "x"
            ax.plot(candidate.relative_probe_coordinate, y, marker=marker, ms=7, color=color, markeredgecolor=DARK if marker != "x" else color, markeredgewidth=0.4, zorder=4)
        ax.text(1.03, y, f"supported {row.n_supported_probe_boundaries}; tentative {row.n_tentative_probe_boundaries}", va="center", fontsize=MIN_FONT_PT, clip_on=False)
    labels = [f"{row.patient} · {row.run_label.split()[-1]}" for row in rows.itertuples(index=False)]
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(-0.02, 1.43)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel("Relative probe y-coordinate (0=tipward contact; 1=baseward contact)")
    fig.text(0.12, 0.965, "Patient-resolved probe-boundary evidence", fontsize=13, fontweight="bold", va="top", color=DARK)
    fig.text(
        0.12,
        0.895,
        "Physiological transitions do not identify pia or white matter without a registered trajectory.\n"
        "All runs therefore retain an anatomically unassigned status.",
        fontsize=MIN_FONT_PT,
        color=GREY,
        va="top",
    )
    handles = [
        Line2D([0], [0], color=DARK, lw=2, label="multi-feature change score"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ENCODED, markeredgecolor=DARK, label="supported"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GOLD, markeredgecolor=DARK, label="tentative"),
        Line2D([0], [0], marker="x", color=GREY, lw=0, label="unsupported"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.785), ncol=4, fontsize=MIN_FONT_PT, columnspacing=1.2, handletextpad=0.45)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    fig.text(0.05, 0.055, "Fixed criteria: bootstrap support, 2/3 feature agreement, level contrast, CI width and edge exclusion.", fontsize=MIN_FONT_PT, color=GREY)
    fig.text(0.05, 0.022, "Exploratory probe registration only; not histological layer or pyramidal-cell classification.", fontsize=MIN_FONT_PT, color=GREY)
    return fig


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    paths = [base.with_suffix(f".{suffix}") for suffix in ("svg", "pdf", "png", "tiff")]
    fig.savefig(paths[0], format="svg")
    fig.savefig(paths[1], format="pdf")
    fig.savefig(paths[2], format="png", dpi=600)
    temporary = paths[3].with_suffix(".raw.tiff")
    fig.savefig(temporary, format="tiff", dpi=600)
    with Image.open(temporary) as image:
        image.save(paths[3], compression="tiff_lzw", dpi=(600, 600))
    temporary.unlink()
    plt.close(fig)
    return paths


def svg_qa(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    sizes = [float(value) for value in re.findall(r"font-size:\s*([0-9.]+)px", text)]
    width = re.search(r'width="([0-9.]+)pt"', text)
    height = re.search(r'height="([0-9.]+)pt"', text)
    return {
        "width_mm": float(width.group(1)) * 25.4 / 72.0 if width else np.nan,
        "height_mm": float(height.group(1)) * 25.4 / 72.0 if height else np.nan,
        "minimum_font_pt": min(sizes) if sizes else 0.0,
        "editable_text_nodes": len(re.findall(r"<text\b", text)),
        "embedded_raster": "<image" in text.lower(),
    }


def raster_qa(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
        return {
            "pixel_dimensions": list(image.size),
            "dpi": list(dpi),
            "dpi_approximately_600": all(abs(value - 600.0) < 1.0 for value in dpi),
            "compression": image.info.get("compression", ""),
        }


def write_contract() -> None:
    payload = {
        "schema_version": "probe-boundary-candidate-contract-v1",
        "core_conclusion": "Only reproducible multi-feature transitions can be called probe-boundary candidates; anatomical layer assignment requires an independent registered trajectory.",
        "archetype": "quantitative grid",
        "backend": "python",
        "scope": {"patients": 5, "runs": 8, "registered_units": 1782},
        "features": ["LF RMS", "LF 1-30 Hz power", "LF 30-100 Hz power", "LF local correlation", "AP RMS", "AP threshold-crossing rate", "unit density", "spike-rate density"],
        "adjunct_overlays": ["task-locked CSD energy", "condition-sensitive unit density", "waveform trough-to-peak duration", "Kilosort amplitude"],
        "condition_sensitive_unit_definition": "condition-to-condition only; significant duration >5 ms and p < 0.05 / number of conditions; no condition-versus-baseline inference",
        "fixed_candidate_criteria": {"bootstrap_support_within_200um": 0.60, "feature_group_agreement_out_of_3": 2, "cross_boundary_level_contrast": 0.15, "bootstrap_ci_width_um": 600.0, "composite_score": 0.45, "minimum_edge_distance_um": MIN_EDGE_DISTANCE_UM},
        "forbidden_calls_without_anatomy": ["pia", "white matter", "cortical layer", "layer 5", "pyramidal cell"],
        "export": {"formats": ["svg", "pdf", "png", "tiff"], "minimum_font_pt": MIN_FONT_PT, "raster_dpi": 600},
    }
    (OUTPUT / "analysis_contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_interpretation(
    run_summary: pd.DataFrame,
    patient_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    units: pd.DataFrame,
) -> None:
    n_runs_supported = int((run_summary["n_supported_probe_boundaries"] > 0).sum())
    n_runs_two_boundary = int(run_summary["two_boundary_coordinate_available"].sum())
    lines = [
        "# 探针组织边界候选与植入深度辅助定位",
        "",
        "本分析融合LF、AP/MUA、unit/spike-rate深度剖面，用固定的change-point和bootstrap标准寻找probe坐标中稳定的生理转折。",
        "CSD、语法/条件敏感unit和spike波形只用于叠加解释，不用于强制指定pia或白质。",
        "条件敏感unit沿用condition-to-condition标准：显著时间>5 ms且p<0.05/condition数量；不使用condition-versus-baseline推断。",
        "",
        "## 核心结论",
        "",
        f"8个run中有{n_runs_supported}个至少包含1个supported probe-coordinate transition，但只有{n_runs_two_boundary}个run同时包含2个supported边界。",
        "因此，当前数据可以识别部分稳定的探针坐标生理转折，但不能识别pia、白质、L5或任何特定细胞类型。",
        "",
        "## Run级结果",
        "",
    ]
    for row in run_summary.itertuples(index=False):
        lines.append(f"- {row.run_label}: supported={row.n_supported_probe_boundaries}, tentative={row.n_tentative_probe_boundaries}, two-boundary coordinate={'yes' if row.two_boundary_coordinate_available else 'no'}.")
        subset = candidates.loc[
            candidates["run"].eq(row.run)
            & candidates["status"].isin(
                ["supported_probe_boundary", "tentative_probe_boundary"]
            )
        ]
        for candidate in subset.itertuples(index=False):
            ci = (
                f"95% bootstrap interval {candidate.bootstrap_ci_low_um:.0f}–"
                f"{candidate.bootstrap_ci_high_um:.0f} µm"
                if np.isfinite(candidate.bootstrap_ci_low_um)
                else "95% bootstrap interval unavailable"
            )
            lines.append(
                f"  - {candidate.candidate}: y={candidate.probe_y_um:.0f} µm, "
                f"support={candidate.bootstrap_support_within_200um:.3f}, {ci}, "
                f"agreement={candidate.feature_group_agreement_out_of_3}/3, "
                f"contrast={candidate.cross_boundary_level_contrast:.3f}, "
                f"status={candidate.status}."
            )
    lines.extend(["", "## 患者级结果", ""])
    for row in patient_summary.itertuples(index=False):
        lines.append(f"- {row.patient}: {row.n_runs} run(s), {row.n_runs_with_two_boundary_coordinate} run(s) support a two-boundary probe coordinate; histological cortical depth remains unidentified.")
    lines.extend(
        [
            "",
            "- gz003的两个run均出现一个靠近tipward端的supported转折（500与520 µm，相差20 µm），提示存在可重复的患者内probe-coordinate生理转折；其组织学身份仍未知。",
            "- gz003 run4另有1740 µm的supported转折，因此只有该run可建立0–1的双边界归一化probe坐标。gz003 run1的第二个转折位于3320 µm且仅为tentative，故双边界结构没有获得患者内重复。",
            "- gz005两个run的supported转折分别位于5980与3860 µm，相差2120 µm，不支持稳定的患者内同一边界。",
            "- gz001和gz002各只有一个run，无法检验患者内重现性；gz004两个run均无supported边界。",
            "",
            "## 条件敏感unit是否聚集在某一候选层段",
            "",
        ]
    )
    normalized = units.loc[units["boundary_coordinate_available"]].copy()
    sensitive = normalized["condition_sensitive_unit"].astype(bool)
    n_inside = int(normalized["inside_supported_boundary_span"].sum())
    n_sensitive_inside = int(
        normalized.loc[sensitive, "inside_supported_boundary_span"].sum()
    )
    n_other_inside = int(
        normalized.loc[~sensitive, "inside_supported_boundary_span"].sum()
    )
    sensitive_total = int(sensitive.sum())
    other_total = int((~sensitive).sum())
    lines.extend(
        [
            f"只有gz003 run4的{len(normalized)}个unit可以获得严格的双边界归一化坐标；其余{len(units) - len(normalized)}个unit仍保留在总表中，但不赋予未经支持的层坐标。",
            f"gz003 run4中，{n_inside}/{len(normalized)}个unit位于两个supported边界之间。条件敏感unit为{n_sensitive_inside}/{sensitive_total}（{100*n_sensitive_inside/sensitive_total:.1f}%），其他unit为{n_other_inside}/{other_total}（{100*n_other_inside/other_total:.1f}%）。",
            "条件敏感unit在该边界区间内没有表现出富集；当前结果不支持‘语法/条件编码集中在某一候选层段’。这只是单run的描述性结果，不能升级为患者级层次结论。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "probe y=0是靠近探针尖端的contact，y增大方向是靠近探针基部；它不能在没有轨迹配准时直接翻译为浅层到深层。",
            "只有两个候选边界同时通过预先固定的支持标准时，才计算low-to-high probe-y归一化坐标；该坐标仍不是组织学皮层深度。",
            "确定pia、白质与L5仍需要每位患者的native MRI/navigation entry-tip轨迹、插入角和brain-shift校正。",
        ]
    )
    (OUTPUT / "interpretation_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (OUTPUT, FIGURES, SOURCE, MAPS, QA):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    registry = pd.read_csv(UNITS_PATH)
    if (
        len(registry) != 1782
        or int((registry["primary_degree"] > 0).sum()) != 695
        or registry[["patient", "run", "unit_id"]].duplicated().any()
    ):
        raise RuntimeError("registered-unit scope changed")
    rng = np.random.default_rng(SEED)
    summaries: list[dict[str, object]] = []
    candidate_tables: list[pd.DataFrame] = []
    profile_tables: list[pd.DataFrame] = []
    unit_tables: list[pd.DataFrame] = []
    for config in RUNS:
        summary, candidates, profiles, units = analyze_run(config, registry, rng)
        summaries.append(summary)
        candidate_tables.append(candidates)
        profile_tables.append(profiles)
        unit_tables.append(units)
    run_summary = pd.DataFrame(summaries)
    candidates = pd.concat(candidate_tables, ignore_index=True)
    profiles = pd.concat(profile_tables, ignore_index=True)
    units = pd.concat(unit_tables, ignore_index=True)
    if len(units) != 1782:
        raise RuntimeError("unit table lost registered units")
    patient_summary = (
        run_summary.groupby("patient", sort=False)
        .agg(
            n_runs=("run", "size"),
            n_supported_probe_boundaries=("n_supported_probe_boundaries", "sum"),
            n_tentative_probe_boundaries=("n_tentative_probe_boundaries", "sum"),
            n_runs_with_two_boundary_coordinate=("two_boundary_coordinate_available", "sum"),
            n_registered_units=("n_registered_units", "sum"),
            n_condition_sensitive_units=("n_condition_sensitive_units", "sum"),
        )
        .reset_index()
    )
    patient_summary["histological_cortical_depth_identifiable"] = False
    run_summary.to_csv(SOURCE / "run_probe_boundary_summary.csv", index=False)
    patient_summary.to_csv(SOURCE / "patient_probe_boundary_summary.csv", index=False)
    candidates.to_csv(SOURCE / "run_probe_boundary_candidates.csv", index=False)
    profiles.to_csv(SOURCE / "run_depth_feature_profiles.csv", index=False)
    units.to_csv(SOURCE / "all_registered_units_boundary_coordinates.csv", index=False)
    figure_paths: list[Path] = []
    for patient in PATIENTS:
        figure_paths.extend(save_figure(plot_patient(patient, profiles, candidates, units), FIGURES / f"Fig_probe_boundary_candidates_{patient}"))
    figure_paths.extend(save_figure(plot_overview(run_summary, profiles, candidates), FIGURES / "Fig_probe_boundary_candidates_all_patients"))
    write_contract()
    write_interpretation(run_summary, patient_summary, candidates, units)
    figure_qa: dict[str, object] = {}
    for svg in sorted(FIGURES.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        png = svg.with_suffix(".png")
        tiff = svg.with_suffix(".tiff")
        svg_info = svg_qa(svg)
        pdf_bytes = pdf.read_bytes()
        png_info = raster_qa(png)
        tiff_info = raster_qa(tiff)
        passed = bool(svg_info["minimum_font_pt"] >= MIN_FONT_PT and svg_info["editable_text_nodes"] > 0 and not svg_info["embedded_raster"] and pdf_bytes.count(b"/Subtype /Image") == 0 and png_info["dpi_approximately_600"] and tiff_info["dpi_approximately_600"] and tiff_info["compression"] == "tiff_lzw")
        figure_qa[svg.stem] = {"passed": passed, "svg": svg_info, "pdf_embedded_image_xobjects": pdf_bytes.count(b"/Subtype /Image"), "png": png_info, "tiff": tiff_info}
    scientific = {
        "patients": int(run_summary["patient"].nunique()),
        "runs": int(len(run_summary)),
        "registered_units": int(len(units)),
        "condition_sensitive_units": int(units["condition_sensitive_unit"].sum()),
        "histological_pia_calls": int(run_summary["histological_pia_identifiable"].sum()),
        "histological_white_matter_calls": int(run_summary["histological_white_matter_identifiable"].sum()),
        "cortical_layer_calls": int(run_summary["cortical_layer_identifiable"].sum()),
    }
    qa_payload = {"schema_version": "probe-boundary-candidate-qa-v1", "passed": all(item["passed"] for item in figure_qa.values()) and scientific["registered_units"] == 1782 and scientific["condition_sensitive_units"] == 695 and scientific["histological_pia_calls"] == 0 and scientific["histological_white_matter_calls"] == 0 and scientific["cortical_layer_calls"] == 0, "figures": figure_qa, "scientific_assertions": scientific}
    (QA / "visual_and_scientific_QA.json").write_text(json.dumps(qa_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not qa_payload["passed"]:
        raise RuntimeError("QA failed")
    inventory = [*figure_paths, *sorted(SOURCE.glob("*.csv")), *sorted(MAPS.glob("*.npz")), OUTPUT / "analysis_contract.json", OUTPUT / "interpretation_zh.md", QA / "visual_and_scientific_QA.json"]
    manifest = {"schema_version": "probe-boundary-candidate-manifest-v1", "created_by": str(Path(__file__).relative_to(ROOT)), "input_units_sha256": sha256(UNITS_PATH), "outputs": {str(path.relative_to(OUTPUT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(inventory)}}
    (OUTPUT / "delivery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT, flush=True)
    print(run_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
