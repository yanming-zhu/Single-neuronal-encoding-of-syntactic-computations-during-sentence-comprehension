#!/usr/bin/env python3
"""Run Kilosort4 for gz004 ActPasSynSentence runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["HOME"] = str(Path.cwd())
Path.home().joinpath(".kilosort").mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

from kilosort import run_kilosort
from kilosort.io import save_probe


ROOT = Path("neuropixel2025/data/gz004/processed/ActPasSynSentence")
RUNS = ["gz004t1_g0_actpas_run0", "gz004t1_g3_actpas_run3"]
KILOSORT_REQUIRED_OUTPUTS = (
    "spike_times.npy",
    "spike_clusters.npy",
    "cluster_KSLabel.tsv",
)


def read_meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def parse_geom_map(meta: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    geom = meta["~snsGeomMap"]
    coords = re.findall(r"\(\d+:(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?):[01]\)", geom)
    if len(coords) < 384:
        raise ValueError(f"Expected at least 384 geometry entries, found {len(coords)}")
    arr = np.array(coords[:384], dtype=np.float32)
    return arr[:, 0], arr[:, 1]


def make_probe(meta_path: Path, probe_path: Path) -> dict[str, np.ndarray | int]:
    meta = read_meta(meta_path)
    xc, yc = parse_geom_map(meta)
    probe = {
        "chanMap": np.arange(384, dtype=np.int32),
        "xc": xc.astype(np.float32),
        "yc": yc.astype(np.float32),
        "kcoords": np.ones(384, dtype=np.float32),
        "n_chan": 384,
    }
    save_probe(probe, probe_path)
    return probe


def run_one(run_name: str, overwrite: bool = False) -> dict[str, str]:
    run_dir = ROOT / run_name
    imec_dir = run_dir / f"{run_name}_imec0"
    ap_bin = imec_dir / f"{run_name}_t0.imec0.ap.bin"
    ap_meta = imec_dir / f"{run_name}_t0.imec0.ap.meta"
    if not ap_bin.exists():
        raise FileNotFoundError(ap_bin)
    if not ap_meta.exists():
        raise FileNotFoundError(ap_meta)

    meta = read_meta(ap_meta)
    fs = float(meta["imSampRate"])
    n_chan_bin = int(meta["nSavedChans"])
    probe_path = run_dir / "np1015_384ch_from_spikeglx.json"
    probe = make_probe(ap_meta, probe_path)

    results_dir = run_dir / "kilosort4"
    results_complete = results_dir.exists() and all((results_dir / name).exists() for name in KILOSORT_REQUIRED_OUTPUTS)
    if results_complete and not overwrite:
        print(f"Skipping {run_name}; results already exist at {results_dir}")
        return {
            "run_name": run_name,
            "status": "skipped_existing",
            "results_dir": str(results_dir),
            "probe_path": str(probe_path),
        }
    if results_dir.exists() and any(results_dir.iterdir()) and not results_complete and not overwrite:
        raise RuntimeError(
            f"Found incomplete Kilosort output at {results_dir}; rerun with --overwrite to replace it."
        )
    if results_dir.exists() and any(results_dir.iterdir()) and overwrite:
        shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "n_chan_bin": n_chan_bin,
        "fs": fs,
        "tmin": 0,
        "tmax": np.inf,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_thread_lim = max(1, min(8, os.cpu_count() or 1))
    if device == torch.device("cpu"):
        settings.update(
            {
                "nblocks": 0,
                "templates_from_data": False,
            }
        )
        torch_thread_lim = 1
    print(f"Running Kilosort4 for {run_name} on {device}; ap={ap_bin}")
    run_kilosort(
        settings=settings,
        probe=probe,
        filename=ap_bin,
        results_dir=results_dir,
        data_dtype="int16",
        device=device,
        do_CAR=True,
        invert_sign=False,
        save_extra_vars=False,
        clear_cache=(device != torch.device("cpu")),
        verbose_console=True,
        torch_thread_lim=torch_thread_lim,
    )
    return {
        "run_name": run_name,
        "status": "sorted",
        "results_dir": str(results_dir),
        "probe_path": str(probe_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS + ["all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    runs = RUNS if args.run == "all" else [args.run]
    results = [run_one(r, overwrite=args.overwrite) for r in runs]
    out = ROOT / "kilosort4_run_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
