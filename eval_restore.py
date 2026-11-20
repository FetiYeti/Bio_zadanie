#!/usr/bin/env python3
"""Vyhodnotenie "odhalenia" – spätnej obnovy pôvodného hlasu.

Režimy:
  bez obnovy     zmenená nahrávka tak, ako prišla
  naslepo        posuny odhadnuté regresiou z príznakov (modely trénované LOSO – hovoriaceho nepoznajú)
  s referenciou  posuny odhadnuté porovnaním s 5 registračnými nahrávkami toho istého človeka
  orákulum       skutočné parametre voice changera (horná hranica metódy)

Metriky:
  * chyba mediánu F0 voči originálu [poltóny]
  * identifikácia hovoriaceho (1:N, uzavretá množina) z príznakov identity hlasu (F0, formanty,
    priemerné spektrum): klasifikátor natrénovaný na pravých
    nahrávkach (párne vety), testovaný na zmenených / obnovených nahrávkach (nepárne vety)

Použitie:  python eval_restore.py
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vcdetect.audio import load
from vcdetect.features import SPEAKER_FEATS, extract, feature_names
from vcdetect.restore import estimate_with_reference, f0_median_st, restore

warnings.filterwarnings("ignore")
TYPE_SK = {"pitch_shift": "posun výšky", "pitch_formant": "výška + zach. formanty", "resample": "prevzorkovanie",
           "phase_vocoder": "fázový vokóder", "formant_shift": "posun formantov", "fx_combo": "výška + efekt"}
MODES = ["bez obnovy", "naslepo", "s referenciou", "orákulum"]


def work(row, s_blind, f_blind, ref_paths):
    x = load(row.file)
    orig = load(row.source_clean)
    refs = [load(p) for p in ref_paths]
    s_ref, f_ref = estimate_with_reference(x, refs)
    s_true, f_true = row.semitones, 12 * np.log2(row.formant_factor)
    outs = {"bez obnovy": x,
            "naslepo": restore(x, s_blind, f_blind),
            "s referenciou": restore(x, s_ref, f_ref),
            "orákulum": restore(x, s_true, f_true)}
    f0o = f0_median_st(orig)
    res = dict(s_true=s_true, f_true=f_true, s_blind=s_blind, f_blind=f_blind, s_ref=s_ref, f_ref=f_ref)
    feats = {}
    for m, y in outs.items():
        res[f"f0err|{m}"] = abs(f0_median_st(y) - f0o)
        feats[m] = extract(y)
    return res, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/generated")
    ap.add_argument("--results", default="results")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    R = Path(a.results); R.mkdir(exist_ok=True)
    df = pd.read_csv(Path(a.data) / "metadata.csv")
    X = np.load(Path(a.data) / "features.npz")["X"]
    main_m = (df.subset == "main").values
    df, X = df[main_m].reset_index(drop=True), X[main_m]
    df["text"] = df.source.str[-3:].astype(int)
    spk = df.speaker.values
    chg = ((df.label == 1) & (df.ctype != "robot")).values

    # --- naslepo: LOSO regresia posunu F0 a formantov
    reg = make_pipeline(StandardScaler(), ExtraTreesRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=0))
    sb, fb = np.zeros(len(df)), np.zeros(len(df))
    ys, yf = df.semitones.values, 12 * np.log2(df.formant_factor.values)
    for s in sorted(set(spk)):
        tr, te = chg & (spk != s), chg & (spk == s)
        sb[te] = clone(reg).fit(X[tr], ys[tr]).predict(X[te])
        fb[te] = clone(reg).fit(X[tr], yf[tr]).predict(X[te])

    test = np.where(chg & (df.text.values % 2 == 1))[0]
    clean_dir = Path(a.data) / "clean"

    def refs_for(row):     # 5 registračných nahrávok rovnakého hovoriaceho, iné (párne) vety
        cand = sorted(p for p in clean_dir.glob("*.wav")
                      if df.loc[df.source_clean == str(p), "speaker"].iloc[0] == row.speaker
                      and int(p.stem[-3:]) % 2 == 0 and p.stem != row.source)
        return cand[:5]
    print(f"Obnovujem {len(test)} zmenených nahrávok v 4 režimoch ...")
    out = Parallel(n_jobs=a.jobs)(delayed(work)(df.loc[i], sb[i], fb[i], refs_for(df.loc[i])) for i in test)
    res = pd.DataFrame([o[0] for o in out])
    res["ctype"] = df.ctype.values[test]
    res["speaker"] = spk[test]

    # --- identifikácia hovoriaceho (1:N)
    gen_tr = (df.label.values == 0) & (df.text.values % 2 == 0)
    sm = np.array([n.startswith(SPEAKER_FEATS) for n in feature_names()])   # príznaky identity hlasu
    sid = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(X[gen_tr][:, sm], spk[gen_tr])
    gen_te = (df.label.values == 0) & (df.text.values % 2 == 1)
    acc_gen = (sid.predict(X[gen_te][:, sm]) == spk[gen_te]).mean()
    for m in MODES:
        F = np.vstack([o[1][m] for o in out])[:, sm]
        res[f"sid|{m}"] = sid.predict(F) == res.speaker.values
    res.to_csv(R / "restore_details.csv", index=False)

    rows = []
    for m in MODES:
        rows.append({"režim": m, "chyba F0 [pt] (medián)": res[f"f0err|{m}"].median(),
                     "chyba F0 [pt] (priemer)": res[f"f0err|{m}"].mean(),
                     "identifikácia hovoriaceho [%]": res[f"sid|{m}"].mean() * 100})
    summ = pd.DataFrame(rows)
    summ.to_csv(R / "restore_summary.csv", index=False)
    by_t = res.groupby("ctype")[[f"sid|{m}" for m in MODES]].mean().mul(100).round(1)
    by_t.columns = MODES
    by_t.index = [TYPE_SK[t] for t in by_t.index]
    est = pd.DataFrame({
        "MAE posunu F0 [pt]": [np.mean(np.abs(res.s_blind - res.s_true)), np.mean(np.abs(res.s_ref - res.s_true))],
        "MAE posunu formantov [pt]": [np.mean(np.abs(res.f_blind - res.f_true)), np.mean(np.abs(res.f_ref - res.f_true))]},
        index=["naslepo", "s referenciou"])
    print(summ.round(3).to_string(index=False))
    print(by_t.to_string())
    print(est.round(2).to_string())

    md = ["# Odhalenie – obnova pôvodného hlasu\n",
          f"Testovaných zmenených nahrávok: {len(res)} (bez robota; nepárne vety). "
          f"Identifikácia hovoriaceho na PRAVÝCH nahrávkach: {acc_gen * 100:.1f} %.\n",
          summ.round(3).to_markdown(index=False),
          "\n\n## Presnosť odhadu parametrov voice changera\n", est.round(3).to_markdown(),
          "\n\n## Identifikácia hovoriaceho podľa typu voice changera [%]\n", by_t.to_markdown(), "\n"]
    (R / "restore_summary.md").write_text("\n".join(md))

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    c = ["#9e9e9e", "#ef6c00", "#1565c0", "#2e7d32"]
    axs[0].bar(MODES, summ["identifikácia hovoriaceho [%]"], color=c)
    axs[0].axhline(acc_gen * 100, color="k", ls="--", lw=1, label=f"pravé nahrávky ({acc_gen * 100:.0f} %)")
    axs[0].axhline(100 / len(set(spk)), color="r", ls=":", lw=1, label="náhodný tip")
    axs[0].set(ylabel="správne identifikovaný hovoriaci [%]", title="Identifikácia hovoriaceho (1:N)", ylim=(0, 105))
    axs[0].legend(fontsize=7)
    bp = [res[f"f0err|{m}"].dropna().clip(upper=15) for m in MODES]
    axs[1].boxplot(bp, tick_labels=MODES, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#e3f2fd"))
    axs[1].set(ylabel="|ΔF0| voči originálu [poltóny]", title="Chyba výšky hlasu po obnove")
    fig.tight_layout(); fig.savefig(R / "restore.png"); plt.close(fig)
    print(f"-> {R}/restore_summary.md")


if __name__ == "__main__":
    main()
