#!/usr/bin/env python3
"""Estimate function-word onsets for ActPas active/passive sentence audio."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "neuropixel2025/files/实验设计/程序设计/neuropixel_m255_202511/ActPasSynSentence/scripts"
OUT_DIR = ROOT / "neuropixel2025/data/gz003/processed/ActPasSynSentence/event_related_active_passive"
SR = 16_000

STIMULI = {
    "stim_actpas/act001.mp3": ("active", "虫子把树叶啃光了", "把"),
    "stim_actpas/act002.mp3": ("active", "交警给汽车拦住了", "给"),
    "stim_actpas/act003.mp3": ("active", "小狗将骨头叼走了", "将"),
    "stim_actpas/act004.mp3": ("active", "妈妈把衣服晾好了", "把"),
    "stim_actpas/act005.mp3": ("active", "球员给足球踢飞了", "给"),
    "stim_actpas/act006.mp3": ("active", "兔子把菜叶吃光了", "把"),
    "stim_actpas/act007.mp3": ("active", "爸爸将爆竹点着了", "将"),
    "stim_actpas/act008.mp3": ("active", "婴儿把牛奶喝完了", "把"),
    "stim_actpas/act009.mp3": ("active", "奶奶将西瓜切开了", "将"),
    "stim_actpas/act010.mp3": ("active", "警察给道路封锁了", "给"),
    "stim_actpas/pass001.mp3": ("passive", "树叶被虫子啃光了", "被"),
    "stim_actpas/pass002.mp3": ("passive", "汽车让交警拦住了", "让"),
    "stim_actpas/pass003.mp3": ("passive", "骨头叫小狗叼走了", "叫"),
    "stim_actpas/pass004.mp3": ("passive", "衣服被妈妈晾好了", "被"),
    "stim_actpas/pass005.mp3": ("passive", "足球让球员踢飞了", "让"),
    "stim_actpas/pass006.mp3": ("passive", "菜叶被兔子吃光了", "被"),
    "stim_actpas/pass007.mp3": ("passive", "爆竹叫爸爸点着了", "叫"),
    "stim_actpas/pass008.mp3": ("passive", "牛奶被婴儿喝完了", "被"),
    "stim_actpas/pass009.mp3": ("passive", "西瓜叫奶奶切开了", "叫"),
    "stim_actpas/pass010.mp3": ("passive", "道路让警察封锁了", "让"),
}

# The function word is the third syllable in this stimulus set. A few envelope
# detections split/merge syllables and pick the following content word; these
# corrections were selected from the QC envelope plots.
MANUAL_TARGET_ONSET_AUDIO_SEC = {
    "stim_actpas/act008.mp3": 0.620,
    "stim_actpas/pass004.mp3": 0.655,
}


def decode_mp3(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-",
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same")


def estimate_target_onset(audio: np.ndarray) -> dict[str, object]:
    hop = int(0.005 * SR)
    frame = int(0.025 * SR)
    starts = np.arange(0, max(1, audio.size - frame), hop)
    rms = np.array([np.sqrt(np.mean(audio[s : s + frame] ** 2)) for s in starts])
    env = moving_average(rms, 9)
    times = starts / SR

    active_thr = max(0.01, float(np.percentile(env, 95) * 0.08))
    active = np.flatnonzero(env > active_thr)
    if active.size == 0:
        raise RuntimeError("No active audio detected")
    speech_start = max(0, int(active[0]) - 2)
    speech_start_sec = float(times[speech_start])

    search_start = speech_start + int(0.08 / 0.005)
    min_peak_gap = int(0.11 / 0.005)
    peak_thr = max(float(np.percentile(env, 75)), float(env.max() * 0.22))
    candidates: list[int] = []
    last = -10_000
    for i in range(search_start + 1, len(env) - 1):
        if env[i] >= peak_thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if i - last >= min_peak_gap:
                candidates.append(i)
                last = i
            elif candidates and env[i] > env[candidates[-1]]:
                candidates[-1] = i
                last = i

    if len(candidates) < 3:
        early_end = min(len(env), speech_start + int(1.4 / 0.005))
        order = np.argsort(env[search_start:early_end])[::-1] + search_start
        candidates = []
        for idx in order:
            if all(abs(int(idx) - j) >= min_peak_gap for j in candidates):
                candidates.append(int(idx))
            if len(candidates) >= 6:
                break
        candidates.sort()

    if len(candidates) < 3:
        raise RuntimeError(f"Could not identify three syllable peaks; found {len(candidates)}")

    target_peak = int(candidates[2])
    left = target_peak
    valley_limit = speech_start + int(0.12 / 0.005)
    while left > valley_limit and env[left - 1] <= env[left]:
        left -= 1
    onset_thr = env[left] + 0.25 * (env[target_peak] - env[left])
    onset = left
    while onset < target_peak and env[onset] < onset_thr:
        onset += 1

    return {
        "speech_start_sec": speech_start_sec,
        "target_onset_sec": float(times[onset]),
        "target_peak_sec": float(times[target_peak]),
        "peak_times_sec": [float(times[i]) for i in candidates[:8]],
        "env_times_sec": times,
        "env": env,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(10, 2, figsize=(13, 23), sharex=False)
    for ax, (sound_file, (condition, sentence, target_word)) in zip(axes.flat, STIMULI.items()):
        est = estimate_target_onset(decode_mp3(AUDIO_DIR / sound_file))
        onset_source = "audio-envelope"
        target_onset_sec = float(est["target_onset_sec"])
        if sound_file in MANUAL_TARGET_ONSET_AUDIO_SEC:
            target_onset_sec = MANUAL_TARGET_ONSET_AUDIO_SEC[sound_file]
            onset_source = "manual-envelope-correction"
        rel_target = target_onset_sec - float(est["speech_start_sec"])

        rows.append(
            {
                "sound_file": sound_file,
                "condition": condition,
                "sentence": sentence,
                "target_word": target_word,
                "speech_start_sec_in_audio": f"{float(est['speech_start_sec']):.6f}",
                "target_onset_sec_in_audio": f"{target_onset_sec:.6f}",
                "target_peak_sec_in_audio": f"{float(est['target_peak_sec']):.6f}",
                "target_onset_sec_from_sentence_onset": f"{rel_target:.6f}",
                "onset_source": onset_source,
                "detected_peak_times_sec_in_audio": json.dumps(est["peak_times_sec"], ensure_ascii=False),
            }
        )

        times = est["env_times_sec"]
        env = est["env"]
        ax.plot(times, env, color="#333333", linewidth=1.1)
        ax.axvline(float(est["speech_start_sec"]), color="#4c78a8", linewidth=1)
        ax.axvline(target_onset_sec, color="#d95f02", linewidth=1.5)
        ax.axvline(float(est["target_peak_sec"]), color="#d95f02", linewidth=0.9, linestyle="--")
        ax.set_title(f"{Path(sound_file).stem} {target_word}: {sentence}", fontsize=9)
        ax.set_xlim(0, min(float(times[-1]), 2.25))
        ax.set_ylabel("RMS")
    for ax in axes[-1]:
        ax.set_xlabel("Audio time (s)")
    fig.suptitle("Estimated active/passive function-word onset from audio envelope", y=0.997)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "active_passive_function_word_audio_onset_qc.png", dpi=170)
    plt.close(fig)

    out_csv = OUT_DIR / "active_passive_function_word_audio_onsets.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    print(f"Wrote {OUT_DIR / 'active_passive_function_word_audio_onset_qc.png'}")


if __name__ == "__main__":
    main()
