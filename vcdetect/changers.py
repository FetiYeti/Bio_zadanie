"""Simulácia voice changerov.

Každý typ napodobňuje princíp, ktorý používajú bežné programy (Voicemod, MorphVOX,
Clownfish, filtre v Discorde/OBS, mobilné aplikácie):

  pitch_shift        posun výšky aj formantov (rubberband, "chipmunk"/"deep voice")
  pitch_formant      posun výšky so zachovaním formantov (kvalitnejšie zmeny pohlavia)
  resample           zmena vzorkovacej frekvencie + WSOLA na udržanie dĺžky (lacné aplikácie)
  phase_vocoder      klasický fázový vokóder + prevzorkovanie (vlastná implementácia)
  formant_shift      posun iba formantov (zmena "veľkosti" hlasového traktu) pri zachovanej výške
  robot              monotónny robotický hlas (fáza nahradená konštantou)
  fx_combo           posun výšky + efekt (chorus / flanger / echo / vibrato) — "démon", "mimozemšťan"

Každá funkcia vráti (signál, metadáta). Metadáta obsahujú posun v poltónoch,
ktorý sa používa ako cieľ pri odhade posunu a pri spätnej obnove hlasu.
"""
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

from .audio import SR, ffmpeg_filter, normalize

CHANGER_TYPES = ["pitch_shift", "pitch_formant", "resample", "phase_vocoder",
                 "formant_shift", "robot", "fx_combo"]


def _rand_semitones(rng, lo, hi):
    s = rng.uniform(lo, hi)
    return float(s if rng.random() < 0.5 else -s)


def _fit_len(y, n):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


# --------------------------------------------------------------------------- #
def pitch_shift(x, semitones, preserve_formants=False):
    r = 2.0 ** (semitones / 12.0)
    af = f"rubberband=pitch={r:.6f}"
    if preserve_formants:
        af += ":formant=preserved"
    return _fit_len(ffmpeg_filter(x, af), len(x))


def resample_shift(x, semitones):
    r = 2.0 ** (semitones / 12.0)
    af = f"asetrate={SR * r:.3f},aresample={SR},atempo={1.0 / r:.6f}"
    return _fit_len(ffmpeg_filter(x, af), len(x))


def formant_shift(x, factor):
    """Posun formantov o `factor` pri zachovaní F0."""
    af = (f"asetrate={SR * factor:.3f},aresample={SR},atempo={1.0 / factor:.6f},"
          f"rubberband=pitch={1.0 / factor:.6f}:formant=preserved")
    return _fit_len(ffmpeg_filter(x, af), len(x))


def _stft(x, n_fft, hop):
    win = np.hanning(n_fft)
    x = np.pad(x, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1)


def _istft(S, n_fft, hop, length):
    win = np.hanning(n_fft)
    frames = np.fft.irfft(S, n=n_fft, axis=1) * win
    out = np.zeros(hop * (len(frames) - 1) + n_fft)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * hop:i * hop + n_fft] += f
        norm[i * hop:i * hop + n_fft] += win ** 2
    out = out / np.maximum(norm, 1e-6)
    out = out[n_fft // 2:]
    return _fit_len(out, length)


def phase_vocoder_shift(x, semitones, n_fft=1024, hop=256):
    """Klasický fázový vokóder: časové natiahnutie o r a prevzorkovanie o 1/r."""
    r = 2.0 ** (semitones / 12.0)
    S = _stft(x, n_fft, hop)
    n_bins = S.shape[1]
    omega = 2 * np.pi * hop * np.arange(n_bins) / n_fft
    t_out = np.arange(0, S.shape[0] - 1, 1.0 / r)
    phase = np.angle(S[0])
    out = np.zeros((len(t_out), n_bins), dtype=complex)
    for i, t in enumerate(t_out):
        k = int(t)
        a = t - k
        mag = (1 - a) * np.abs(S[k]) + a * np.abs(S[k + 1])
        out[i] = mag * np.exp(1j * phase)
        dphi = np.angle(S[k + 1]) - np.angle(S[k]) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase += omega + dphi
    stretched = _istft(out, n_fft, hop, int(len(x) * r))
    frac = Fraction(r).limit_denominator(200)
    y = resample_poly(stretched, frac.denominator, frac.numerator)
    return _fit_len(y, len(x))


def robot(x, robot_f0=100.0, n_fft=1024):
    """Robotický hlas: zachová spektrálnu obálku, fázu nahradí nulou -> bzučanie na robot_f0."""
    hop = int(round(SR / robot_f0))
    S = _stft(x, n_fft, hop)
    return _istft(np.abs(S).astype(complex), n_fft, hop, len(x))


def apply_changer(x, ctype, rng):
    """Aplikuje náhodne parametrizovaný voice changer daného typu."""
    meta = {"type": ctype, "semitones": 0.0, "formant_factor": 1.0}
    if ctype == "pitch_shift":
        s = _rand_semitones(rng, 2.5, 9)
        y = pitch_shift(x, s)
        meta.update(semitones=s, formant_factor=2 ** (s / 12))
    elif ctype == "pitch_formant":
        s = _rand_semitones(rng, 2.5, 9)
        y = pitch_shift(x, s, preserve_formants=True)
        meta.update(semitones=s)
    elif ctype == "resample":
        s = _rand_semitones(rng, 2.5, 8)
        y = resample_shift(x, s)
        meta.update(semitones=s, formant_factor=2 ** (s / 12))
    elif ctype == "phase_vocoder":
        s = _rand_semitones(rng, 2.5, 9)
        y = phase_vocoder_shift(x, s)
        meta.update(semitones=s, formant_factor=2 ** (s / 12))
    elif ctype == "formant_shift":
        f = rng.uniform(1.12, 1.3) if rng.random() < 0.5 else rng.uniform(0.75, 0.89)
        y = formant_shift(x, f)
        meta.update(formant_factor=f)
    elif ctype == "robot":
        f0 = rng.uniform(70, 160)
        y = robot(x, f0)
        meta.update(robot_f0=f0)
    elif ctype == "fx_combo":
        s = _rand_semitones(rng, 3, 9)
        fx = rng.choice([
            "chorus=0.6:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3",
            "flanger=delay=2:depth=4:speed=0.6",
            "aecho=0.8:0.7:40|70:0.4|0.3",
            "vibrato=f=6:d=0.35",
        ])
        y = ffmpeg_filter(pitch_shift(x, s), fx)
        y = _fit_len(y, len(x))
        meta.update(semitones=s, formant_factor=2 ** (s / 12), fx=str(fx).split("=")[0])
    else:
        raise ValueError(ctype)
    return normalize(y.astype(np.float32)), meta


# --------------------------------------------------------------------------- #
# Prenosový kanál – aplikuje sa rovnako na pravé aj zmenené nahrávky, aby sa
# model nenaučil rozlišovať kodek namiesto voice changera.
CHANNELS = ["clean", "mp3", "opus", "telephone", "noise"]


def apply_channel(x, ch, rng):
    from .audio import codec_roundtrip
    if ch == "clean":
        y = x
    elif ch == "mp3":
        y = codec_roundtrip(x, "libmp3lame", rng.choice(["32k", "48k", "64k"]))
    elif ch == "opus":
        y = codec_roundtrip(x, "libopus", rng.choice(["12k", "16k", "24k"]))
    elif ch == "telephone":
        y = codec_roundtrip(ffmpeg_filter(x, "highpass=f=300,lowpass=f=3400"), "pcm_mulaw")
    elif ch == "noise":
        snr = rng.uniform(15, 30)
        p = np.mean(x ** 2) + 1e-12
        y = x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(x))
    else:
        raise ValueError(ch)
    g = 10 ** (rng.uniform(-12, 0) / 20)
    return (normalize(y) * g).astype(np.float32)
