#!/usr/bin/env python3
"""Tréning a vyhodnotenie detektora voice changera.

Protokoly:
  P1  Leave-one-speaker-out (LOSO): test vždy na hovoriacom, ktorého model nevidel
      -> EER, AUC, presnosť pre 4 klasifikátory; rozpad podľa typu útoku a kanála
  P2  Leave-one-attack-out (LOAO): typ voice changera chýba v tréningu (neznámy útok)
  P3  Ablácia skupín príznakov
  P4  Klasifikácia typu voice changera (8 tried) – LOSO, matica zámen
  P5  Odhad posunu výšky v poltónoch (regresia) – LOSO
  +   Externý test na nahrávkach zo skutočných aplikácií (ak existujú v data/real_app)

Výstupy: results/*.csv, results/*.png, results/summary.md, models/detector.joblib
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

TYPE_SK = {"genuine": "pravý hlas", "pitch_shift": "posun výšky", "pitch_formant": "výška + zach. formanty",
           "resample": "prevzorkovanie", "phase_vocoder": "fázový vokóder", "formant_shift": "posun formantov",
           "robot": "robot", "fx_combo": "výška + efekt"}
SHIFT_TYPES = ["pitch_shift", "pitch_formant", "resample", "phase_vocoder", "fx_combo"]
from vcdetect.features import SPEAKER_FEATS   # vyradené z detektora, pozri P3


def classifiers(seed=0):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=4, gamma="scale", probability=True, random_state=seed)),
        "RandomForest": RandomForestClassifier(400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=seed),
    }


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def group_of(names, prefixes):
    return np.array([any(n.startswith(p) for p in prefixes) for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/generated")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=["artifact", "all"], default="artifact",
                    help="artifact = bez príznakov identity hovoriaceho (predvolené)")
    a = ap.parse_args()
    R = Path(a.results); R.mkdir(exist_ok=True)
    M = Path(a.models); M.mkdir(exist_ok=True)

    df = pd.read_csv(Path(a.data) / "metadata.csv")
    z = np.load(Path(a.data) / "features.npz")
    X, names = z["X"], list(z["names"])
    X_full, names_full = X, names
    det_mask = np.array([not n.startswith(SPEAKER_FEATS) for n in names]) if a.features == "artifact" \
        else np.ones(len(names), bool)
    X, names = X[:, det_mask], [n for n, k in zip(names, det_mask) if k]
    main_m = (df.subset == "main").values
    dfm, Xm = df[main_m].reset_index(drop=True), X[main_m]
    y = dfm.label.values
    spk = dfm.speaker.values
    speakers = sorted(set(spk))
    summary = ["# Výsledky – detekcia voice changera\n",
               f"Nahrávok: {len(dfm)} (pravé {int((y == 0).sum())}, zmenené {int((y == 1).sum())}), "
               f"hovoriaci: {', '.join(speakers)}, príznakov: {X.shape[1]}\n"]

    # ------------------------------------------------- P0 známy hovoriaci --- #
    print("P0: známy hovoriaci (verifikácia 1:1) ...")
    txt = dfm.source.str[-3:].values          # index vety -> rozdelenie podľa obsahu
    folds = pd.factorize(txt)[0] % 5
    rows = []
    for name, clf in classifiers(a.seed).items():
        s0 = np.zeros(len(y))
        for f in range(5):
            te = folds == f
            s0[te] = clone(clf).fit(Xm[~te], y[~te]).predict_proba(Xm[te])[:, 1]
        rows.append(dict(klasifikator=name, EER=eer(y, s0)[0], AUC=roc_auc_score(y, s0),
                         presnost=accuracy_score(y, s0 > 0.5)))
    p0 = pd.DataFrame(rows)
    p0.to_csv(R / "p0_seen_speaker.csv", index=False)
    print(p0.round(4).to_string())
    summary += ["\n## P0 – známy hovoriaci (5-násobná CV podľa vety; scenár verifikácie 1:1)\n",
                p0.assign(EER=lambda d: d.EER * 100).round(4).to_markdown(index=False), "\n"]

    # ------------------------------------------------------------- P1 LOSO --- #
    print("P1: leave-one-speaker-out ...")
    scores = {n: np.zeros(len(y)) for n in classifiers()}
    rows = []
    for name, clf in classifiers(a.seed).items():
        for s in speakers:
            te = spk == s
            m = clone(clf).fit(Xm[~te], y[~te])
            scores[name][te] = m.predict_proba(Xm[te])[:, 1]
        for s in speakers + ["SPOLU"]:
            te = spk == s if s != "SPOLU" else np.ones(len(y), bool)
            e, _ = eer(y[te], scores[name][te])
            rows.append(dict(klasifikator=name, test_hovoriaci=s, EER=e, AUC=roc_auc_score(y[te], scores[name][te]),
                             presnost=accuracy_score(y[te], scores[name][te] > 0.5)))
    p1 = pd.DataFrame(rows)
    p1.to_csv(R / "p1_loso.csv", index=False)
    piv = p1.pivot(index="klasifikator", columns="test_hovoriaci", values="EER")
    print((piv * 100).round(2).to_string())
    pooled = p1[p1.test_hovoriaci == "SPOLU"].set_index("klasifikator")
    best = pooled.EER.idxmin()
    summary += ["\n## P1 – neznámy hovoriaci (LOSO)\n", "EER [%] podľa testovaného hovoriaceho:\n",
                (piv * 100).round(2).to_markdown(), "\n\nSúhrn (všetky foldy spolu):\n",
                pooled[["EER", "AUC", "presnost"]].assign(EER=lambda d: d.EER * 100).round(4).to_markdown(),
                f"\n\nNajlepší klasifikátor: **{best}**\n"]

    # ROC
    fig, ax = plt.subplots(figsize=(4.6, 4))
    for n, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr * 100, tpr * 100, label=f"{n} (EER {eer(y, s)[0] * 100:.1f} %)")
    ax.plot([0, 100], [0, 100], "k:", lw=0.8)
    ax.set(xlabel="FAR – falošne prijaté zmenené hlasy [%]", ylabel="detegované zmenené hlasy [%]",
           title="ROC – neznámy hovoriaci (LOSO)", xlim=(0, 40), ylim=(60, 100.5))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(R / "roc_loso.png"); plt.close(fig)

    # rozpad podľa typu útoku a kanála (najlepší klasifikátor)
    sb = scores[best]
    _, thr_best = eer(y, sb)
    dt = []
    for t in dfm.ctype.unique():
        m = dfm.ctype.values == t
        rate = (sb[m] >= thr_best).mean() if t != "genuine" else (sb[m] < thr_best).mean()
        mm = m | (y == 0)
        dt.append(dict(typ=t, typ_sk=TYPE_SK[t], n=int(m.sum()),
                       uspesnost=rate, EER_vs_pravy=eer(y[mm], sb[mm])[0] if t != "genuine" else np.nan))
    dt = pd.DataFrame(dt).sort_values("uspesnost")
    dt.to_csv(R / "p1_by_type.csv", index=False)
    dc = []
    for c in sorted(dfm.channel.unique()):
        m = dfm.channel.values == c
        dc.append(dict(kanal=c, n=int(m.sum()), EER=eer(y[m], sb[m])[0], AUC=roc_auc_score(y[m], sb[m])))
    dc = pd.DataFrame(dc)
    dc.to_csv(R / "p1_by_channel.csv", index=False)
    summary += [f"\nÚspešnosť podľa typu ({best}, prah = EER prah {thr_best:.3f}; pri pravom hlase = správne prijaté):\n",
                dt[["typ_sk", "n", "uspesnost", "EER_vs_pravy"]].round(4).to_markdown(index=False),
                "\n\nPodľa prenosového kanála:\n", dc.round(4).to_markdown(index=False), "\n"]

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    axs[0].barh(dt.typ_sk, dt.uspesnost * 100, color=["#2e7d32" if t == "genuine" else "#c62828" for t in dt.typ])
    axs[0].set(xlabel="správne rozhodnutia [%]", title=f"Úspešnosť podľa typu ({best})", xlim=(min(50, dt.uspesnost.min() * 100 - 5), 100))
    for i, v in enumerate(dt.uspesnost):
        axs[0].text(v * 100 - 0.5, i, f"{v * 100:.1f}", va="center", ha="right", color="white", fontsize=7)
    axs[1].bar(dc.kanal, dc.EER * 100, color="#1565c0")
    axs[1].set(ylabel="EER [%]", title="EER podľa kanála")
    fig.tight_layout(); fig.savefig(R / "by_type_channel.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(sb[y == 0], 40, alpha=0.7, label="pravý hlas", color="#2e7d32")
    ax.hist(sb[y == 1], 40, alpha=0.7, label="voice changer", color="#c62828")
    ax.axvline(thr_best, color="k", ls="--", lw=1, label="prah (EER)")
    ax.set(xlabel="skóre detektora", ylabel="počet", title=f"Rozdelenie skóre ({best}, LOSO)", yscale="log")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(R / "score_hist.png"); plt.close(fig)

    # ------------------------------------------------------------- P2 LOAO --- #
    print("P2: leave-one-attack-out ...")
    rows = []
    best_clf = classifiers(a.seed)[best]
    for t in [c for c in TYPE_SK if c != "genuine"]:
        for s in speakers:
            te_s = spk == s
            tr = (~te_s) & (dfm.ctype.values != t)
            te = te_s & ((dfm.ctype.values == t) | (y == 0))
            m = clone(best_clf).fit(Xm[tr], y[tr])
            ptr = m.predict_proba(Xm[tr])[:, 1]
            _, th = eer(y[tr], ptr)
            p = m.predict_proba(Xm[te])[:, 1]
            rows.append(dict(neznamy_typ=t, s=s, eer=eer(y[te], p)[0], auc=roc_auc_score(y[te], p),
                             det=(p[y[te] == 1] >= 0.5).mean()))
    p2 = pd.DataFrame(rows).groupby("neznamy_typ")[["eer", "auc", "det"]].mean().reset_index()
    p2["typ_sk"] = p2.neznamy_typ.map(TYPE_SK)
    p2.to_csv(R / "p2_loao.csv", index=False)
    print((p2.set_index("typ_sk")[["eer", "det"]] * 100).round(1).to_string())
    summary += ["\n## P2 – neznámy typ voice changera (leave-one-attack-out, zároveň neznámy hovoriaci)\n",
                p2[["typ_sk", "eer", "auc", "det"]].rename(columns={"eer": "EER", "auc": "AUC", "det": "detegované (prah 0.5)"})
                .round(4).to_markdown(index=False), "\n"]

    # ----------------------------------------------------------- P3 ablácia --- #
    print("P3: ablácia príznakov ...")
    groups = {
        "kepstrálne (MFCC+LFCC)": ["mfcc", "dmfcc", "lfcc", "dlfcc"],
        "spektrálne + LTAS": ["centroid", "bandwidth", "rolloff", "logflat", "flux", "loghi", "logvhi", "ltas"],
        "prozódia (F0, jitter, periodicita)": ["f0_mean", "f0_std", "f0_range", "f0_median", "jitter", "shimmer", "periodicity", "voiced"],
        "formanty + F0/formanty": ["F1", "F2", "F3", "formant", "vtl", "f0_vs"],
        "fáza": ["phase"],
        "modulácie": ["mod_"],
        "harmonická štruktúra (HNR, CPP, cutoff)": ["hnr", "cpp", "cutoff", "tilt"],
    }
    rows = []
    Xf = X_full[main_m]
    spk_cols = np.array([n.startswith(SPEAKER_FEATS) for n in names_full])
    for g, pre in list(groups.items()) + [("detektor (bez príznakov identity)", None),
                                          ("VŠETKY vrátane príznakov identity", [""])]:
        cols = ~spk_cols if pre is None else group_of(names_full, pre)
        s_ = np.zeros(len(y))
        for s in speakers:
            te = spk == s
            m = clone(best_clf).fit(Xf[~te][:, cols], y[~te])
            s_[te] = m.predict_proba(Xf[te][:, cols])[:, 1]
        rows.append(dict(skupina=g, pocet=int(cols.sum()), EER=eer(y, s_)[0], AUC=roc_auc_score(y, s_),
                         **{f"EER_{sp}": eer(y[spk == sp], s_[spk == sp])[0] for sp in speakers}))
    p3 = pd.DataFrame(rows)
    p3.to_csv(R / "p3_ablation.csv", index=False)
    print(p3.round(4).to_string())
    summary += ["\n## P3 – ablácia skupín príznakov (LOSO, " + best + ")\n", p3.round(4).to_markdown(index=False), "\n"]

    # ----------------------------------------------------- P4 typ útoku ----- #
    print("P4: klasifikácia typu ...")
    yt = dfm.ctype.values
    pt = np.empty(len(yt), dtype=object)
    type_clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, random_state=a.seed)
    for s in speakers:
        te = spk == s
        pt[te] = clone(type_clf).fit(Xf[~te], yt[~te]).predict(Xf[te])
    labels = list(TYPE_SK)
    cm = confusion_matrix(yt, pt, labels=labels, normalize="true")
    acc_t, f1_t = accuracy_score(yt, pt), f1_score(yt, pt, average="macro")
    print(f"typ: presnosť {acc_t:.3f}, macro-F1 {f1_t:.3f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), [TYPE_SK[l] for l in labels], rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), [TYPE_SK[l] for l in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="predikovaný typ", ylabel="skutočný typ", title=f"Typ voice changera (LOSO), presnosť {acc_t * 100:.1f} %")
    ax.grid(False); fig.tight_layout(); fig.savefig(R / "confusion_type.png"); plt.close(fig)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(R / "p4_confusion.csv")
    summary += [f"\n## P4 – určenie typu voice changera (8 tried, LOSO)\nPresnosť {acc_t * 100:.2f} %, macro-F1 {f1_t:.4f}\n"]

    # ------------------------------------------------ P5 odhad poltónov ----- #
    print("P5: odhad posunu ...")
    sm = dfm.ctype.isin(SHIFT_TYPES).values
    ys = dfm.semitones.values
    reg = make_pipeline(StandardScaler(), ExtraTreesRegressor(500, min_samples_leaf=2, n_jobs=-1, random_state=a.seed))
    ps = np.zeros(len(ys))
    for s in speakers:
        tr, te = sm & (spk != s), sm & (spk == s)
        ps[te] = clone(reg).fit(Xf[tr], ys[tr]).predict(Xf[te])
    mae = mean_absolute_error(ys[sm], ps[sm])
    sign = (np.sign(ps[sm]) == np.sign(ys[sm])).mean()
    within2 = (np.abs(ps[sm] - ys[sm]) <= 2).mean()
    per_t = {t: mean_absolute_error(ys[dfm.ctype.values == t], ps[dfm.ctype.values == t]) for t in SHIFT_TYPES}
    print(f"MAE {mae:.2f} st, smer {sign:.3f}, do ±2 st {within2:.3f}")
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for t in SHIFT_TYPES:
        m = dfm.ctype.values == t
        ax.scatter(ys[m], ps[m], s=8, alpha=0.7, label=TYPE_SK[t])
    ax.plot([-10, 10], [-10, 10], "k:", lw=0.8)
    ax.set(xlabel="skutočný posun [poltóny]", ylabel="odhadnutý posun [poltóny]",
           title=f"Odhad posunu výšky (LOSO), MAE {mae:.2f} pt")
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(R / "semitone_regression.png"); plt.close(fig)
    # odhad posunu formantov (log2 faktora) – potrebné pre spätnú obnovu hlasu
    fm_ = (y == 1) & (dfm.ctype.values != "robot")
    yf = 12 * np.log2(dfm.formant_factor.values)
    pf = np.zeros(len(yf))
    for s in speakers:
        tr, te = fm_ & (spk != s), fm_ & (spk == s)
        pf[te] = clone(reg).fit(Xf[tr], yf[tr]).predict(Xf[te])
    mae_f = mean_absolute_error(yf[fm_], pf[fm_])
    print(f"formanty: MAE {mae_f:.2f} st")
    summary += [f"\n## P5 – odhad posunu výšky hlasu (LOSO)\nMAE {mae:.2f} poltónu, správny smer posunu {sign * 100:.1f} %, "
                f"odchýlka do ±2 poltónov {within2 * 100:.1f} %\n\nMAE podľa typu: " +
                ", ".join(f"{TYPE_SK[t]} {v:.2f}" for t, v in per_t.items()) +
                f"\n\nOdhad posunu formantov (v poltónových jednotkách, všetky typy okrem robota): MAE {mae_f:.2f}\n"]

    # -------------------------------------------- dôležitosť príznakov ------ #
    print("Dôležitosť príznakov ...")
    te = spk == speakers[0]
    m = clone(best_clf).fit(Xm[~te], y[~te])
    pi = permutation_importance(m, Xm[te], y[te], n_repeats=8, random_state=a.seed, scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=names).sort_values(ascending=False)
    imp.to_csv(R / "feature_importance.csv", header=["importance"])
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.barh(top.index, top.values, color="#6a1b9a")
    ax.set(xlabel="pokles AUC po permutácii", title=f"20 najdôležitejších príznakov ({best})")
    fig.tight_layout(); fig.savefig(R / "feature_importance.png"); plt.close(fig)

    # --------------------------------------------- finálne modely ----------- #
    print("Tréning finálnych modelov na všetkých dátach ...")
    det = clone(best_clf).fit(Xm, y)
    final = dict(detector=det, detector_name=best, threshold=0.5, eer_threshold_cv=thr_best,
                 type_clf=clone(type_clf).fit(Xf, yt),
                 shift_reg=clone(reg).fit(Xf[sm], ys[sm]),
                 formant_reg=clone(reg).fit(Xf[fm_], yf[fm_]),
                 feature_names=names_full, detector_mask=det_mask, type_sk=TYPE_SK,
                 cv=dict(eer=float(pooled.EER[best]), auc=float(pooled.AUC[best])))
    joblib.dump(final, M / "detector.joblib", compress=3)

    # ------------------------------------------ externý test ---------------- #
    ext = (df.subset == "external_test").values
    if ext.any():
        pe = det.predict_proba(X[ext])[:, 1]   # X je už zúžené maskou detektora
        de = df[ext].assign(skore=pe, verdikt=np.where(pe >= 0.5, "VOICE CHANGER", "pravý"))
        de.to_csv(R / "external_test.csv", index=False)
        ye = df.label.values[ext]
        summary += ["\n## Externý test – nahrávky zo skutočných aplikácií\n",
                    f"Nahrávok {ext.sum()}, presnosť {accuracy_score(ye, pe >= 0.5) * 100:.1f} %" +
                    (f", EER {eer(ye, pe)[0] * 100:.2f} %" if len(set(ye)) == 2 else "") + "\n"]

    (R / "summary.md").write_text("\n".join(summary))
    json.dump(dict(best=best, eer=float(pooled.EER[best]), auc=float(pooled.AUC[best]), type_acc=acc_t, shift_mae=mae),
              open(R / "summary.json", "w"), indent=1)
    print(f"Hotovo. Najlepší: {best}, EER {pooled.EER[best] * 100:.2f} %  -> {R}/summary.md")


if __name__ == "__main__":
    main()
