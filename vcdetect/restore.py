"""Odhalenie: odhad parametrov voice changera a spätná obnova (približne) pôvodného hlasu.

Voice changer typicky robí dve veci: posunie základnú frekvenciu F0 o `s` poltónov a
(nie vždy) posunie formanty – rezonancie hlasového traktu – o `f` poltónov.
Ak oba posuny odhadneme, môžeme ich invertovať:
   1) posun výšky o -s so ZACHOVANÍM formantov (vráti F0)
   2) posun formantov o -f pri zachovaní F0 (vráti "farbu" / dĺžku hlasového traktu)

Odhad posunov:
  * naslepo      – regresné modely natrénované na príznakoch (bez znalosti hovoriaceho)
  * s referenciou – porovnanie s registračnou nahrávkou daného človeka (scenár verifikácie 1:1):
                    s = rozdiel mediánov F0, f = posun spektrálnej obálky na logaritmickej osi
Robotický efekt obnoviť nemožno – priebeh F0 je nenávratne stratený.
"""
import numpy as np

from .audio import SR, normalize
from .changers import formant_shift, pitch_shift
from .features import yin_f0, frame


def restore(x, semitones, formant_semitones):
    y = x
    if abs(semitones) >= 0.5:
        y = pitch_shift(y, -semitones, preserve_formants=True)
    if abs(formant_semitones) >= 0.5:
        y = formant_shift(y, 2 ** (-formant_semitones / 12))
    return normalize(y)


def f0_median_st(x):
    f0, _ = yin_f0(np.asarray(x, float))
    v = f0[f0 > 0]
    return float(np.median(12 * np.log2(v / 100.0))) if len(v) > 5 else np.nan


# log-frekvenčná os: 1/4 poltónu, 150 Hz – 6 kHz
_LOGF = 150 * 2 ** (np.arange(0, 12 * np.log2(6000 / 150) * 4 + 1) / 48)


def envelope_logfreq(x, n_fft=1024, hop=160, n_cep=30):
    """Priemerná spektrálna obálka (kepstrálne vyhladená) na logaritmickej frekvenčnej osi."""
    x = np.asarray(x, float)
    fr = frame(x, n_fft, hop) * np.hanning(n_fft)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    e = P.sum(1)
    P = P[e > 1e-3 * e.max()]
    L = np.log(P + 1e-10)
    C = np.fft.irfft(L, axis=1)
    C[:, n_cep:-n_cep] = 0                      # liftrovanie -> obálka bez harmonických
    env = np.fft.rfft(C, axis=1).real.mean(0)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    v = np.interp(_LOGF, freqs, env)
    return (v - v.mean()) / (v.std() + 1e-9)


def estimate_with_reference(x, refs, max_st=14):
    """Odhad (posun F0, posun formantov) v poltónoch voči registračným nahrávkam `refs`."""
    f_ref = np.nanmedian([f0_median_st(r) for r in refs])
    s = f0_median_st(x) - f_ref
    er = np.mean([envelope_logfreq(r) for r in refs], axis=0)
    et = envelope_logfreq(x)
    best, best_c = 0.0, -np.inf
    for k in range(-max_st * 4, max_st * 4 + 1):         # krok 1/4 poltónu
        a = et[max(k, 0):len(et) + min(k, 0)]
        b = er[max(-k, 0):len(er) + min(-k, 0)]
        if len(a) < len(et) // 2:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best_c:
            best, best_c = k / 4, c
    return float(s), float(best)
