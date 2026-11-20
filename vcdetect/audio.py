"""Načítanie / uloženie audia a spúšťanie ffmpeg filtrov (bez ďalších závislostí)."""
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile

SR = 16000


def load(path, sr=SR):
    """Načíta ľubovoľný formát (wav, mp3, ogg, m4a, ...) cez ffmpeg -> mono float32 [-1, 1]."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def save(path, x, sr=SR):
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(path), sr, (x * 32767).astype(np.int16))


def ffmpeg_filter(x, af, sr=SR, out_args=None):
    """Prežene signál x cez ffmpeg audio filter `af` a vráti výsledok (16 kHz mono)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
           "-af", af, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, input=np.asarray(x, np.float32).tobytes(),
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def codec_roundtrip(x, codec, bitrate=None, sr=SR, codec_sr=None):
    """Zakóduje a znova dekóduje signál (simulácia prenosového kanála: mp3, opus, gsm...)."""
    ext = {"libmp3lame": ".mp3", "libopus": ".ogg", "libgsm": ".gsm", "aac": ".m4a",
           "pcm_mulaw": ".wav"}[codec]
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.wav")
        dst = os.path.join(td, "out" + ext)
        save(src, x, sr)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-c:a", codec]
        if bitrate:
            cmd += ["-b:a", bitrate]
        if codec in ("libgsm", "pcm_mulaw"):
            codec_sr = 8000
        if codec_sr:
            cmd += ["-ar", str(codec_sr)]
        if codec == "libgsm":
            cmd += ["-f", "gsm"]
        cmd += [dst]
        subprocess.run(cmd, capture_output=True, check=True)
        y = load(dst, sr)
    n = len(x)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def normalize(x, peak=0.9):
    m = np.max(np.abs(x)) + 1e-9
    return x * (peak / m)
