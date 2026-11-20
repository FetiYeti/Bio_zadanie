#!/usr/bin/env python3
"""Ukážka pre video / prezentáciu.

Prehrá pravý hlas, hlas zmenený voice changerom a obnovený hlas a ku každému vypíše
výsledok detektora. Na konci zobrazí spektrogramy (pravý / zmenený / obnovený).

  python demo.py                  # celá ukážka
  python demo.py --pause          # pred každým krokom čaká na Enter (pohodlné pri nahrávaní)
  python demo.py --no-audio       # bez prehrávania zvuku
  python demo.py --file moj.wav   # analyzovať vlastnú nahrávku
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from vcdetect.audio import load, save
from vcdetect.restore import estimate_with_reference, restore, f0_median_st
from detect import analyze

warnings.filterwarnings("ignore")
if os.name == "nt":
    os.system("")                       # zapne ANSI farby vo Windows termináli
G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
EX = Path(__file__).parent / "examples"
OUT = Path(__file__).parent / "demo_out"


def play(path, enabled):
    if not enabled:
        return
    path = str(path)
    try:
        if os.name == "nt":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        elif shutil.which("ffplay"):
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        elif shutil.which("afplay"):
            subprocess.run(["afplay", path])
        elif shutil.which("aplay"):
            subprocess.run(["aplay", "-q", path])
    except Exception as e:
        print(f"  (zvuk sa nepodarilo prehrať: {e})")


def step(title, pause):
    print(f"\n{B}{C}{'─' * 70}\n  {title}\n{'─' * 70}{X}")
    if pause:
        input(f"  {Y}[Enter] pokračovať{X}")


def show(r):
    vc = r["verdikt"] == "VOICE CHANGER"
    col = R if vc else G
    bar = "█" * int(round(r["pravdepodobnost_voice_changer"] * 30))
    print(f"  Verdikt:          {B}{col}{r['verdikt']}{X}")
    print(f"  Pravdepodobnosť:  {col}{bar:<30}{X} {r['pravdepodobnost_voice_changer'] * 100:5.1f} %")
    if vc and "typ" in r:
        print(f"  Typ úpravy:       {B}{r['typ_sk']}{X}")
        if r.get("typ") != "robot":
            print(f"  Posun výšky:      {r['posun_vysky_poltony']:+.1f} poltónov   [{r['odhad_rezim']}]")
            print(f"  Posun formantov:  {r['posun_formantov_poltony']:+.1f} poltónov")
    if "poznamka" in r:
        print(f"  {Y}Poznámka: {r['poznamka']}{X}")


def spectro_plot(files, titles):
    import matplotlib.pyplot as plt
    from vcdetect.features import yin_f0, HOP
    fig, axs = plt.subplots(len(files), 1, figsize=(10, 2.6 * len(files)), sharex=True)
    for ax, f, t in zip(axs, files, titles):
        x = load(f)
        n, hop = 512, 128
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
        S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6)
        S -= S.max()
        ax.imshow(S.T, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=[0, len(x) / 16000, 0, 8])
        f0, _ = yin_f0(x.astype(float))
        ax.plot(np.arange(len(f0)) * HOP / 16000, np.where(f0 > 0, f0 / 100, np.nan), color="cyan", lw=1.5)
        ax.set_title(f"{t}   (medián F0: {100 * 2 ** (f0_median_st(x) / 12):.0f} Hz)")
        ax.set_ylabel("kHz")
    axs[-1].set_xlabel("čas [s]   (azúrová = F0 × 10)")
    fig.suptitle("Detekcia a odhalenie voice changera", fontweight="bold")
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--file", help="vlastná nahrávka na analýzu")
    a = ap.parse_args()
    audio = not a.no_audio
    OUT.mkdir(exist_ok=True)

    print(f"\n{B}  DETEKCIA A ODHALENIE VOICE CHANGERA V REČI{X}")
    print("  Biometrické systémy bezpečnosti – TUKE FEI\n  Načítavam model ...")
    t = time.time()
    model = joblib.load(Path(__file__).parent / "models" / "detector.joblib")
    print(f"  Model načítaný ({time.time() - t:.1f} s), detektor: {model['detector_name']}")

    if a.file:
        step(f"Analýza nahrávky: {a.file}", a.pause)
        play(a.file, audio)
        out = OUT / (Path(a.file).stem + "_obnovene.wav")
        r = analyze(a.file, model, restore_to=out)
        show(r)
        if r["verdikt"] == "VOICE CHANGER" and r.get("typ") != "robot":
            print(f"\n  {B}Obnovený hlas:{X} {out}")
            play(out, audio)
        return

    refs = [load(p) for p in sorted(EX.glob("ref_rms_*.wav"))]

    step("1/4  Pravý hlas", a.pause)
    play(EX / "1_pravy_hlas.wav", audio)
    show(analyze(EX / "1_pravy_hlas.wav", model))

    step("2/4  Ten istý človek cez VOICE CHANGER", a.pause)
    play(EX / "2_voice_changer_+7pt.wav", audio)
    show(analyze(EX / "2_voice_changer_+7pt.wav", model))

    step("3/4  ODHALENIE – obnova pôvodného hlasu (s registračnou vzorkou)", a.pause)
    x = load(EX / "2_voice_changer_+7pt.wav")
    s, f = estimate_with_reference(x, refs)
    print(f"  Odhadnutý posun výšky:     {s:+.1f} poltónov")
    print(f"  Odhadnutý posun formantov: {f:+.1f} poltónov")
    print("  Aplikujem inverznú transformáciu ...")
    restored = OUT / "obnoveny_hlas.wav"
    save(restored, restore(x, s, f))
    fo, fc, fr = (f0_median_st(load(p)) for p in (EX / "1_pravy_hlas.wav", EX / "2_voice_changer_+7pt.wav", restored))
    hz = lambda st: 100 * 2 ** (st / 12)
    print(f"  Medián F0:  originál {G}{hz(fo):.0f} Hz{X}  →  voice changer {R}{hz(fc):.0f} Hz{X}  →  obnovený {G}{hz(fr):.0f} Hz{X}")
    play(restored, audio)

    step("4/4  Robotický voice changer", a.pause)
    play(EX / "4_robot.wav", audio)
    show(analyze(EX / "4_robot.wav", model))

    print(f"\n{B}{G}  Hotovo.{X} Obnovená nahrávka: {restored}\n")
    if not a.no_plot:
        spectro_plot([EX / "1_pravy_hlas.wav", EX / "2_voice_changer_+7pt.wav", restored],
                     ["Pravý hlas", "Voice changer (+7 poltónov)", "Obnovený hlas"])


if __name__ == "__main__":
    main()
