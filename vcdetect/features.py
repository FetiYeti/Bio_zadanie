"""Extrakcia príznakov na detekciu voice changera (čisté numpy/scipy).

Skupiny príznakov (všetky sa agregujú cez rámce na mean/std a pod.):
  cepstral   MFCC (mel) a LFCC (lineárne pásma — citlivé na artefakty vo vyšších frekvenciách) + delty
  spectral   centroid, šírka pásma, rolloff, plochosť, tok, pomer energie v pásme 4–8 kHz
  ltas       dlhodobé priemerné spektrum (24 log. pásiem, normalizované)
  prosody    F0 (YIN), rozsah, jitter, shimmer, periodicita/HNR, podiel znelých rámcov
  formant    F1–F3 z LPC, rozptyl formantov -> odhad dĺžky hlasového traktu, pomer F0/formanty
  phase      konzistencia fázy medzi rámcami (artefakty fázového vokódera, "phasiness")
  modulation modulačné spektrum energetickej obálky (robot, vibrato, tremolo)
"""
import numpy as np
from scipy.fft import dct
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from .audio import SR

# Príznaky opisujúce IDENTITU hovoriaceho (absolútna výška, poloha formantov, priemerné spektrum).
# Detektor ich nepoužíva (inak sa učí "vysoký hlas = voice changer"); používa ich identifikácia hovoriaceho.
SPEAKER_FEATS = ("f0_mean", "f0_median", "F1_med", "F2_med", "F3_med", "formant_disp", "vtl", "ltas",
                 "lfccraw", "centroid_mean", "rolloff_mean", "bandwidth_mean")

N_FFT = 512
HOP = 160          # 10 ms
WIN = 400          # 25 ms


# ----------------------------------------------------------------- základ --- #
def frame(x, win, hop):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def stft(x):
    fr = frame(x, WIN, HOP) * np.hamming(WIN)
    return np.fft.rfft(fr, n=N_FFT, axis=1)


def mel_fb(n_filt=40, n_fft=N_FFT, sr=SR, fmin=20, fmax=None, linear=False):
    fmax = fmax or sr / 2
    if linear:
        pts = np.linspace(fmin, fmax, n_filt + 2)
    else:
        m = np.linspace(2595 * np.log10(1 + fmin / 700), 2595 * np.log10(1 + fmax / 700), n_filt + 2)
        pts = 700 * (10 ** (m / 2595) - 1)
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_filt, len(bins)))
    for i in range(n_filt):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l + 1e-9), (r - bins) / (r - c + 1e-9)), 0, None)
    return fb


_MEL = mel_fb(40)
_LIN = mel_fb(40, linear=True)


def deltas(f, w=2):
    pad = np.pad(f, ((w, w), (0, 0)), mode="edge")
    num = sum(n * (pad[w + n:len(f) + w + n] - pad[w - n:len(f) + w - n]) for n in range(1, w + 1))
    return num / (2 * sum(n * n for n in range(1, w + 1)))


def cepstra(P, fb, n=20):
    e = np.log(P @ fb.T + 1e-10)
    c = dct(e, type=2, axis=1, norm="ortho")[:, :n]
    return c - c.mean(0)      # CMN – odstráni vplyv kanála


# ------------------------------------------------------------------ F0 ----- #
def yin_f0(x, fmin=60, fmax=500, win=640, hop=HOP, thr=0.15):
    """F0 metódou YIN. Vracia (f0 [Hz, 0 = neznelé], periodicita 0..1)."""
    fr = frame(x, win, hop)
    fr = fr - fr.mean(1, keepdims=True)
    tmax = int(SR / fmin)
    tmin = int(SR / fmax)
    W = win - tmax
    # rozdielová funkcia cez autokoreláciu (FFT)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    F1 = np.fft.rfft(fr[:, :W], nfft, axis=1)      # okno dĺžky W
    F2 = np.fft.rfft(fr, nfft, axis=1)
    acf = np.fft.irfft(np.conj(F1) * F2, nfft, axis=1)[:, :tmax + 1]   # sum x[j] x[j+tau], j<W
    cs = np.cumsum(np.pad(fr ** 2, ((0, 0), (1, 0))), axis=1)
    e0 = cs[:, W] - cs[:, 0]
    lags = np.arange(tmax + 1)
    eL = cs[:, lags + W] - cs[:, lags]
    d = e0[:, None] + eL - 2 * acf
    cmnd = np.ones_like(d)
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = d[:, 1:] * lags[1:] / np.maximum(cum, 1e-12)
    f0 = np.zeros(len(fr))
    per = np.zeros(len(fr))
    energy = np.sqrt(e0 / W)
    e_thr = 0.05 * np.percentile(energy, 95)
    for i in range(len(fr)):
        c = cmnd[i, tmin:tmax]
        below = np.where(c < thr)[0]
        if len(below) and energy[i] > e_thr:
            j = below[0]
            while j + 1 < len(c) and c[j + 1] < c[j]:
                j += 1
            t = j + tmin
            if 1 <= t < tmax:   # parabolická interpolácia
                a, b, cc = cmnd[i, t - 1], cmnd[i, t], cmnd[i, t + 1]
                den = a - 2 * b + cc
                t = t + (0.5 * (a - cc) / den if abs(den) > 1e-12 else 0)
            f0[i] = SR / t
            per[i] = 1 - min(c[j], 1)
    return f0, per


# -------------------------------------------------------------- formanty --- #
def lpc_formants(x, order=14, n_max=3):
    xe = lfilter([1, -0.97], 1, x)
    fr = frame(xe, WIN, HOP) * np.hamming(WIN)
    out = np.full((len(fr), n_max), np.nan)
    en = (fr ** 2).sum(1)
    e_thr = 0.05 * np.percentile(en, 95)
    for i, f in enumerate(fr):
        if en[i] < e_thr:
            continue
        r = np.correlate(f, f, "full")[WIN - 1:WIN + order]
        if r[0] <= 0:
            continue
        r[0] *= 1.0001
        try:
            a = solve_toeplitz(r[:order], -r[1:order + 1])
        except Exception:
            continue
        roots = np.roots(np.r_[1, a])
        roots = roots[np.imag(roots) > 0]
        fr_hz = np.angle(roots) * SR / (2 * np.pi)
        bw = -np.log(np.abs(roots) + 1e-12) * SR / np.pi
        ok = (fr_hz > 150) & (fr_hz < 5000) & (bw < 500)
        fz = np.sort(fr_hz[ok])[:n_max]
        out[i, :len(fz)] = fz
    return out


# ---------------------------------------------------------------- fáza ----- #
def phase_features(S, P):
    """Druhá diferencia fázy v silných binoch: prirodzená reč má koherentnú fázu harmonických,
    fázový vokóder a WSOLA ju narušujú."""
    ph = np.angle(S)
    k = np.arange(S.shape[1])
    expected = 2 * np.pi * HOP * k / N_FFT
    d1 = ph[1:] - ph[:-1] - expected
    d1 = np.angle(np.exp(1j * d1))             # odchýlka okamžitej frekvencie
    d2 = np.angle(np.exp(1j * (d1[1:] - d1[:-1])))
    mag = P[2:]
    thr = np.percentile(mag, 90, axis=1, keepdims=True)
    mask = mag >= thr
    bands = [(0, 1000), (1000, 3000), (3000, 8000)]
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    feats = []
    for lo, hi in bands:
        sel = mask & ((freqs >= lo) & (freqs < hi))[None, :]
        v = np.abs(d2[sel]) if sel.any() else np.array([np.pi / 2])
        feats += [v.mean(), np.median(v), (v < 0.3).mean()]
    # konzistencia odchýlky IF v čase pri harmonických (nízke pásmo)
    return np.array(feats)


# --------------------------------------------------- harmonická štruktúra -- #
def harmonic_features(x, f0, win=1024, hop=320, nfft=4096):
    """HNR po pásmach (harmonické vs. medziharmonické), CPP a spektrálny cutoff.
    Posun výšky / prevzorkovanie rozmazáva harmonické a mení ich pomer k šumu inak
    v nízkych a vysokých pásmach, než je to pri prirodzenej reči."""
    fr = frame(x, win, hop) * np.hanning(win)
    P = np.abs(np.fft.rfft(fr, nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    # F0 je vo výpočte s hop=160 -> prevzorkovať na hop=320 (stred okna)
    idx = np.clip((np.arange(len(fr)) * hop + win // 2 - 320) // HOP, 0, len(f0) - 1)
    f0f = f0[idx]
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    hnr = [[] for _ in bands]
    for i in np.where(f0f > 0)[0]:
        F = f0f[i]
        hw = F / 6
        k = np.arange(1, int(8000 / F))
        for b, (lo, hi) in enumerate(bands):
            kk = k[(k * F >= lo) & (k * F < hi)]
            if len(kk) == 0:
                continue
            h = sum(P[i, (freqs >= c - hw) & (freqs <= c + hw)].sum() for c in kk * F)
            n = sum(P[i, (freqs >= c - hw) & (freqs <= c + hw)].sum() for c in (kk + 0.5) * F)
            hnr[b].append(10 * np.log10((h + 1e-12) / (n + 1e-12)))
    out = []
    for h in hnr:
        out += [np.mean(h), np.std(h)] if len(h) > 3 else [0.0, 0.0]
    # CPP – cepstral peak prominence
    L = 10 * np.log10(P + 1e-12)
    C = np.abs(np.fft.irfft(L, axis=1))[:, :nfft // 2]
    q = np.arange(nfft // 2) / SR
    qr = (q >= 1 / 400) & (q <= 1 / 60)
    v = f0f > 0
    if v.sum() > 3:
        cpp = []
        for c in C[v]:
            seg = c[qr]
            j = np.argmax(seg)
            line = np.polyval(np.polyfit(q[qr], seg, 1), q[qr][j])
            cpp.append(seg[j] - line)
        out += [np.mean(cpp), np.std(cpp)]
    else:
        out += [0.0, 0.0]
    # spektrálny cutoff a sklon najvyšších pásiem
    act = P.sum(1) > 1e-3 * P.sum(1).max()
    lt = 10 * np.log10(P[act].mean(0) + 1e-12)
    lt -= lt.max()
    above = np.where(lt > -50)[0]
    cutoff = freqs[above[-1]] / 1000 if len(above) else 0
    hi_tilt = lt[(freqs > 6500) & (freqs < 7800)].mean() - lt[(freqs > 3000) & (freqs < 4000)].mean()
    mid_tilt = lt[(freqs > 4500) & (freqs < 5500)].mean() - lt[(freqs > 3000) & (freqs < 4000)].mean()
    out += [cutoff, hi_tilt, mid_tilt]
    names = [f"hnr_{lo // 1000}-{hi // 1000}k_{s}" for lo, hi in bands for s in ("mean", "std")]
    names += ["cpp_mean", "cpp_std", "cutoff_khz", "tilt_hi", "tilt_mid"]
    return np.array(out, float), names


# ------------------------------------------------------------ modulácie ---- #
def modulation_features(P):
    env = np.log(P.sum(1) + 1e-10)
    env = env - env.mean()
    if len(env) < 64:
        env = np.pad(env, (0, 64 - len(env)))
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fm = np.fft.rfftfreq(len(env), HOP / SR)
    tot = spec[(fm > 0.5) & (fm < 50)].sum() + 1e-10
    bands = [(0.5, 4), (4, 8), (8, 16), (16, 32), (32, 50)]
    return np.array([spec[(fm >= lo) & (fm < hi)].sum() / tot for lo, hi in bands])


# ----------------------------------------------------------------- spolu --- #
def _stats(m, name):
    m = np.asarray(m, float)
    if m.ndim == 1:
        m = m[:, None]
    return ([*np.nanmean(m, 0), *np.nanstd(m, 0)],
            [f"{name}{i}_mean" for i in range(m.shape[1])] + [f"{name}{i}_std" for i in range(m.shape[1])])


def extract(x, return_names=False):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    x = x / (np.max(np.abs(x)) + 1e-9)
    S = stft(x)
    P = np.abs(S) ** 2
    # aktívne rámce (odstránenie ticha)
    e = 10 * np.log10(P.sum(1) + 1e-12)
    act = e > (e.max() - 40)
    Pa = P[act] if act.sum() > 10 else P

    vals, names = [], []

    def add(v, n):
        vals.extend(v)
        names.extend(n)

    mfcc = cepstra(Pa, _MEL, 20)
    lfcc = cepstra(Pa, _LIN, 20)
    for c, n in [(mfcc, "mfcc"), (lfcc, "lfcc")]:
        v, nn = _stats(c, n)
        add(v[len(v) // 2:], nn[len(nn) // 2:])          # std (mean je po CMN nula)
        v, nn = _stats(deltas(c), "d" + n)
        add(v[len(v) // 2:], nn[len(nn) // 2:])
    # surové kepstrálne priemery z LFCC bez CMN (tvar spektra)
    lraw = dct(np.log(Pa @ _LIN.T + 1e-10), type=2, axis=1, norm="ortho")[:, 1:13]
    add(list(lraw.mean(0)), [f"lfccraw{i}_mean" for i in range(1, 13)])

    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    Pn = Pa / (Pa.sum(1, keepdims=True) + 1e-12)
    centroid = (Pn * freqs).sum(1)
    bw = np.sqrt((Pn * (freqs - centroid[:, None]) ** 2).sum(1))
    roll = freqs[np.argmax(np.cumsum(Pn, 1) >= 0.85, axis=1)]
    flat = np.exp(np.mean(np.log(Pa + 1e-12), 1)) / (np.mean(Pa, 1) + 1e-12)
    flux = np.r_[0, np.sqrt((np.diff(np.sqrt(Pn), axis=0) ** 2).sum(1))]
    hi = Pa[:, freqs >= 4000].sum(1) / (Pa.sum(1) + 1e-12)
    vhi = Pa[:, freqs >= 6000].sum(1) / (Pa.sum(1) + 1e-12)
    for arr, n in [(centroid / 1000, "centroid"), (bw / 1000, "bandwidth"), (roll / 1000, "rolloff"),
                   (np.log(flat + 1e-12), "logflatness"), (flux, "flux"),
                   (np.log(hi + 1e-9), "loghiratio"), (np.log(vhi + 1e-9), "logvhiratio")]:
        v, nn = _stats(arr, n)
        add(v, [s.replace("0_", "_") for s in nn])

    # LTAS
    edges = np.geomspace(250, 8000, 21)
    ltas = np.array([Pa[:, (freqs >= edges[i]) & (freqs < edges[i + 1])].mean() for i in range(20)])
    ltas = 10 * np.log10(ltas + 1e-12)
    ltas -= ltas.max()
    add(list(ltas), [f"ltas{i}" for i in range(20)])

    # prozódia
    f0, per = yin_f0(x)
    v = f0 > 0
    vr = v.mean()
    if v.sum() >= 5:
        lf = 12 * np.log2(f0[v] / 100.0)         # poltóny voči 100 Hz
        jit = np.abs(np.diff(lf))
        jit = jit[jit < 2]
        rms = np.sqrt((frame(x, 640, HOP) ** 2).mean(1))[:len(f0)]
        shim = np.abs(np.diff(20 * np.log10(rms[v] + 1e-9)))
        shim = shim[shim < 6]
        pros = [np.mean(lf), np.std(lf), np.percentile(lf, 95) - np.percentile(lf, 5),
                np.median(lf), jit.mean() if len(jit) else 0, np.median(jit) if len(jit) else 0,
                shim.mean() if len(shim) else 0, per[v].mean(), np.percentile(per[v], 10), vr]
        f0m = np.median(f0[v])
    else:
        pros = [0, 0, 0, 0, 0, 0, 0, 0, 0, vr]
        f0m = np.nan
    add(pros, ["f0_mean_st", "f0_std_st", "f0_range_st", "f0_median_st", "jitter_mean",
               "jitter_median", "shimmer_db", "periodicity_mean", "periodicity_p10", "voiced_ratio"])

    # formanty
    F = lpc_formants(x)
    fm = np.nanmedian(F, 0) if np.isfinite(F).any() else np.full(3, np.nan)
    fs = np.nanstd(F, 0) if np.isfinite(F).any() else np.full(3, np.nan)
    disp = np.nanmedian(np.diff(F, axis=1)) if np.isfinite(F).any() else np.nan
    vtl = 35000 / (2 * disp) if disp and np.isfinite(disp) and disp > 0 else np.nan   # cm
    f0_vs_form = (12 * np.log2(f0m / fm[0]) if np.isfinite(f0m) and np.isfinite(fm[0]) else np.nan)
    f0_vs_disp = (12 * np.log2(f0m / disp) if np.isfinite(f0m) and np.isfinite(disp) and disp > 0 else np.nan)
    add([*(fm / 1000), *(fs / 1000), disp / 1000 if np.isfinite(disp) else np.nan, vtl, f0_vs_form, f0_vs_disp],
        ["F1_med", "F2_med", "F3_med", "F1_std", "F2_std", "F3_std", "formant_disp", "vtl_cm",
         "f0_vs_F1_st", "f0_vs_disp_st"])

    add(list(phase_features(S[act] if act.sum() > 10 else S, Pa if act.sum() > 10 else P)),
        [f"phase_{b}_{s}" for b in ("low", "mid", "high") for s in ("mean", "median", "coh")])
    add(list(modulation_features(P)), [f"mod_{b}" for b in ("0-4", "4-8", "8-16", "16-32", "32-50")])
    hv, hn = harmonic_features(x, f0)
    add(list(hv), hn)

    vals = np.nan_to_num(np.array(vals, float), nan=0.0, posinf=0.0, neginf=0.0)
    return (vals, names) if return_names else vals


def feature_names():
    return extract(np.random.default_rng(0).normal(size=SR) * 0.1, return_names=True)[1]
