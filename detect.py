#!/usr/bin/env python3
"""Detekcia voice changera v nahrávke a jeho odhalenie.

Príklady:
  python detect.py nahravka.wav
  python detect.py nahravka.mp3 --restore obnovena.wav
  python detect.py nahravka.wav --reference moj_hlas1.wav moj_hlas2.wav --restore obnovena.wav
  python detect.py priecinok_s_nahravkami/ --csv vysledky.csv

Výstup: verdikt (pravý hlas / voice changer), pravdepodobnosť, odhadnutý typ voice changera,
odhad posunu výšky a formantov v poltónoch a voliteľne obnovená nahrávka.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

from vcdetect.audio import load, save
from vcdetect.features import extract
from vcdetect.restore import estimate_with_reference, restore

warnings.filterwarnings("ignore")
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".webm"}


def analyze(path, model, refs=None, restore_to=None, threshold=None):
    x = load(path)
    if len(x) < 16000 * 0.8:
        raise ValueError("nahrávka je príliš krátka (min. ~1 s reči)")
    f = extract(x)
    thr = model["threshold"] if threshold is None else threshold
    p = float(model["detector"].predict_proba(f[model["detector_mask"]][None])[0, 1])
    res = {"subor": str(path), "pravdepodobnost_voice_changer": round(p, 4),
           "verdikt": "VOICE CHANGER" if p >= thr else "pravý hlas", "prah": thr}
    if p >= thr or restore_to:
        proba = model["type_clf"].predict_proba(f[None])[0]
        classes = model["type_clf"].classes_
        changed = [(c, q) for c, q in zip(classes, proba) if c != "genuine"]
        ctype, q = max(changed, key=lambda t: t[1])
        res["typ"] = ctype
        res["typ_sk"] = model["type_sk"][ctype]
        res["typ_istota"] = round(float(q / (sum(q_ for _, q_ in changed) + 1e-9)), 3)
        if refs:
            s, fs = estimate_with_reference(x, refs)
            res["odhad_rezim"] = "s referenciou"
        else:
            s = float(model["shift_reg"].predict(f[None])[0])
            fs = float(model["formant_reg"].predict(f[None])[0])
            res["odhad_rezim"] = "naslepo"
        res["posun_vysky_poltony"] = round(s, 2)
        res["posun_formantov_poltony"] = round(fs, 2)
        if ctype == "robot":
            res["poznamka"] = "robotický efekt nie je možné spätne obnoviť (priebeh F0 je stratený)"
        if restore_to:
            save(restore_to, restore(x, s, fs))
            res["obnovena_nahravka"] = str(restore_to)
    return res


def pretty(r):
    lines = [f"\n=== {r['subor']}",
             f"  Verdikt:        {r['verdikt']}   (P = {r['pravdepodobnost_voice_changer']:.3f}, prah {r['prah']:.2f})"]
    if "typ" in r:
        lines.append(f"  Typ zmeny:      {r['typ_sk']} ({r['typ']}), istota {r['typ_istota']:.2f}")
        d = "vyššie" if r["posun_vysky_poltony"] > 0 else "nižšie"
        lines.append(f"  Posun výšky:    {r['posun_vysky_poltony']:+.1f} poltónov (hlas je {d} ako pôvodný) [{r['odhad_rezim']}]")
        lines.append(f"  Posun formantov:{r['posun_formantov_poltony']:+.1f} poltónov")
    if "poznamka" in r:
        lines.append(f"  Poznámka:       {r['poznamka']}")
    if "obnovena_nahravka" in r:
        lines.append(f"  Obnovený hlas:  {r['obnovena_nahravka']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Detekcia a odhalenie voice changera")
    ap.add_argument("inputs", nargs="+", help="audio súbory alebo priečinky")
    ap.add_argument("--model", default="models/detector.joblib")
    ap.add_argument("--reference", nargs="*", help="registračné nahrávky pravého hlasu osoby (zlepší odhad a obnovu)")
    ap.add_argument("--restore", help="kam uložiť obnovenú nahrávku (pri priečinku: priečinok)")
    ap.add_argument("--threshold", type=float, help="rozhodovací prah (predvolene 0.5)")
    ap.add_argument("--json", action="store_true", help="výstup ako JSON")
    ap.add_argument("--csv", help="uložiť výsledky do CSV")
    a = ap.parse_args()

    model = joblib.load(a.model)
    refs = [load(r) for r in a.reference] if a.reference else None
    files = []
    for i in a.inputs:
        p = Path(i)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT) if p.is_dir() else [p]
    results = []
    for f in files:
        out = None
        if a.restore:
            out = Path(a.restore) / f"{f.stem}_obnovene.wav" if len(files) > 1 else Path(a.restore)
            out.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = analyze(f, model, refs, out, a.threshold)
        except Exception as e:
            r = {"subor": str(f), "chyba": str(e)}
        results.append(r)
        if not a.json:
            print(pretty(r) if "chyba" not in r else f"\n=== {f}\n  CHYBA: {r['chyba']}")
    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    if a.csv:
        import pandas as pd
        pd.DataFrame(results).to_csv(a.csv, index=False)
        print(f"\nUložené: {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
