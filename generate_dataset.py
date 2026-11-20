#!/usr/bin/env python3
"""Vygeneruje dataset pravých a zmenených (voice changer) nahrávok.

Zdroje pravej reči:
  1) syntetické hlasy flite (kal, kal16, awb, rms, slt) – vždy k dispozícii, bez sťahovania
  2) VLASTNÉ nahrávky:  data/own/<meno_hovoriaceho>/*.wav|mp3|m4a|ogg   (odporúčané!)
     Každý podpriečinok = jeden hovoriaci. Na tieto nahrávky sa aplikujú rovnaké voice changery.
  3) Nahrávky zo SKUTOČNÝCH aplikácií (Voicemod, MorphVOX, Clownfish...) na čisté testovanie:
     data/real_app/<názov_aplikácie_a_efektu>/*.wav  -> label=1, subset=external_test
     data/real_app/genuine/*.wav                       -> label=0, subset=external_test

Pre každú pravú nahrávku vzniknú 2 pravé verzie (rôzny kanál) a 2 zmenené verzie
(dva rôzne typy voice changera, každá cez náhodný kanál) -> vyvážené triedy.

Použitie:
  python generate_dataset.py --out data/generated --seed 42
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vcdetect.audio import SR, load, save, normalize
from vcdetect.changers import CHANGER_TYPES, CHANNELS, apply_changer, apply_channel

FLITE_VOICES = {"kal": "kal", "kal16": "kal", "awb": "awb", "rms": "rms", "slt": "slt"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".webm"}


def tts(text, voice, path):
    tf = str(path) + ".txt"
    Path(tf).write_text(text)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"flite=textfile={tf}:voice={voice}", "-ar", str(SR), "-ac", "1", str(path)],
                   check=True)
    os.remove(tf)


def trim_silence(x, top_db=40):
    hop = 160
    n = len(x) // hop
    e = 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    idx = np.where(e > e.max() - top_db)[0]
    if len(idx) == 0:
        return x
    a = max(0, (idx[0] - 10) * hop)
    b = min(len(x), (idx[-1] + 10) * hop)
    return x[a:b]


def process_source(src_id, x, speaker, origin, out, seed, job_idx):
    rng = np.random.default_rng([seed, job_idx])
    x = normalize(trim_silence(x)).astype(np.float32)
    clean_path = out / "clean" / f"{src_id}.wav"
    save(clean_path, x)
    rows = []
    chans = rng.choice(CHANNELS, 2, replace=False)
    for k, ch in enumerate(chans):
        y = apply_channel(x, ch, rng)
        p = out / "audio" / f"{src_id}_gen{k}.wav"
        save(p, y)
        rows.append(dict(file=str(p), label=0, speaker=speaker, origin=origin, source=src_id,
                         source_clean=str(clean_path), ctype="genuine", semitones=0.0,
                         formant_factor=1.0, channel=ch, subset="main"))
    # dva rôzne typy voice changera, vyvážene cez všetky nahrávky
    t1 = CHANGER_TYPES[job_idx % len(CHANGER_TYPES)]
    t2 = CHANGER_TYPES[(job_idx + 3) % len(CHANGER_TYPES)]
    for k, ct in enumerate([t1, t2]):
        y, meta = apply_changer(x, ct, rng)
        ch = rng.choice(CHANNELS)
        y = apply_channel(y, ch, rng)
        p = out / "audio" / f"{src_id}_vc{k}_{ct}.wav"
        save(p, y)
        rows.append(dict(file=str(p), label=1, speaker=speaker, origin=origin, source=src_id,
                         source_clean=str(clean_path), ctype=ct, semitones=meta["semitones"],
                         formant_factor=meta["formant_factor"], channel=ch, subset="main"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--texts", default="data/texts.txt")
    ap.add_argument("--own", default="data/own")
    ap.add_argument("--real_app", default="data/real_app")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_tts", action="store_true", help="nepoužiť syntetické hlasy (len vlastné nahrávky)")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    out = Path(a.out)
    for d in ("audio", "clean", "tts"):
        (out / d).mkdir(parents=True, exist_ok=True)

    sources = []   # (src_id, loader, speaker, origin)
    if not a.no_tts:
        texts = [t.strip() for t in Path(a.texts).read_text().splitlines() if t.strip()]
        print(f"TTS: {len(texts)} viet x {len(FLITE_VOICES)} hlasov")

        def mk(i, t, v):
            p = out / "tts" / f"{v}_{i:03d}.wav"
            if not p.exists():
                tts(t, v, p)
            return p
        paths = Parallel(n_jobs=a.jobs)(delayed(mk)(i, t, v) for v in FLITE_VOICES for i, t in enumerate(texts))
        k = 0
        for v in FLITE_VOICES:
            for i in range(len(texts)):
                sources.append((f"tts_{v}_{i:03d}", paths[k], FLITE_VOICES[v], "tts"))
                k += 1

    own = Path(a.own)
    if own.exists():
        for spk in sorted(p for p in own.iterdir() if p.is_dir()):
            files = sorted(f for f in spk.rglob("*") if f.suffix.lower() in AUDIO_EXT)
            print(f"Vlastné nahrávky: {spk.name}: {len(files)} súborov")
            for f in files:
                sources.append((f"own_{spk.name}_{f.stem}", f, f"own_{spk.name}", "own"))

    print(f"Spracúvam {len(sources)} zdrojových nahrávok ...")
    res = Parallel(n_jobs=a.jobs, verbose=0)(
        delayed(process_source)(sid, load(p), spk, org, out, a.seed, i)
        for i, (sid, p, spk, org) in enumerate(sources))
    rows = [r for rr in res for r in rr]

    # externé testovacie nahrávky zo skutočných aplikácií
    ra = Path(a.real_app)
    if ra.exists():
        for d in sorted(p for p in ra.iterdir() if p.is_dir()):
            for f in sorted(f for f in d.rglob("*") if f.suffix.lower() in AUDIO_EXT):
                g = d.name.lower() == "genuine"
                rows.append(dict(file=str(f), label=0 if g else 1, speaker="real_app", origin="real_app",
                                 source=f.stem, source_clean="", ctype="genuine" if g else f"app:{d.name}",
                                 semitones=np.nan, formant_factor=np.nan, channel="unknown",
                                 subset="external_test"))

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    print(df.groupby(["subset", "ctype"]).size().to_string())
    print(f"Spolu: {len(df)} nahrávok -> {out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
