#!/usr/bin/env python3
"""Ilustračné obrázky do správy: spektrogramy pravého hlasu a všetkých typov voice changera
pre tú istú vetu, s vyznačeným priebehom F0."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vcdetect.audio import SR, load
from vcdetect.changers import CHANGER_TYPES, apply_changer
from vcdetect.features import HOP, yin_f0

TYPE_SK = {"pitch_shift": "posun výšky", "pitch_formant": "výška + zachované formanty",
           "resample": "prevzorkovanie", "phase_vocoder": "fázový vokóder", "formant_shift": "posun formantov",
           "robot": "robot", "fx_combo": "výška + efekt"}


def spec(ax, x, title):
    n, hop = 512, 128
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6)
    S -= S.max()
    ax.imshow(S.T, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
              extent=[0, len(x) / SR, 0, SR / 2000])
    f0, _ = yin_f0(x.astype(float))
    t = np.arange(len(f0)) * HOP / SR
    f = np.where(f0 > 0, f0 / 1000, np.nan)
    ax.plot(t, f * 10, color="cyan", lw=1, label="F0 × 10")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("kHz")
    ax.grid(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/generated/clean/tts_awb_002.wav")
    ap.add_argument("--out", default="results/spectrograms.png")
    a = ap.parse_args()
    x = load(a.src)
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(4, 2, figsize=(10, 9), sharex=True)
    axs = axs.ravel()
    spec(axs[0], x, "pravý hlas")
    axs[0].legend(loc="upper right", fontsize=7)
    for ax, t in zip(axs[1:], CHANGER_TYPES):
        y, m = apply_changer(x, t, rng)
        extra = f" ({m['semitones']:+.1f} pt)" if m["semitones"] else (
            f" (formanty ×{m['formant_factor']:.2f})" if t == "formant_shift" else "")
        spec(ax, y, TYPE_SK[t] + extra)
    for ax in axs[-2:]:
        ax.set_xlabel("čas [s]")
    fig.tight_layout()
    Path(a.out).parent.mkdir(exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
