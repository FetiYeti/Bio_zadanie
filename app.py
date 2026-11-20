#!/usr/bin/env python3
"""Grafická aplikácia – detekcia a odhalenie voice changera.

Spustí lokálny server a otvorí rozhranie v prehliadači (nič sa neodosiela na internet).
  python app.py              # http://127.0.0.1:8765
  python app.py --port 9000 --no-browser
"""
import argparse
import base64
import io
import json
import mimetypes
import re
import socket
import threading
import time
import traceback
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from detect import analyze
from vcdetect.audio import SR, load, normalize, save
from vcdetect.changers import (formant_shift, phase_vocoder_shift, pitch_shift, resample_shift, robot,
                               _fit_len)
from vcdetect.audio import ffmpeg_filter
from vcdetect.features import HOP, yin_f0
from vcdetect.restore import f0_median_st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
USER = ROOT / "user_data"
DIRS = {"examples": ROOT / "examples", "nahrate": USER / "nahrate", "simulovane": USER / "simulovane",
        "obnovene": USER / "obnovene"}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".webm"}
MODEL = None
LOCK = threading.Lock()
_cache = {}


# ------------------------------------------------------------------ súbory --- #
def file_id(p: Path):
    for k, d in DIRS.items():
        try:
            return f"{k}/{p.resolve().relative_to(d.resolve()).as_posix()}"
        except ValueError:
            pass
    raise ValueError("súbor mimo povolených priečinkov")


def resolve(fid: str) -> Path:
    k, _, rest = fid.partition("/")
    if k not in DIRS or not rest or ".." in Path(rest).parts:
        raise ValueError("neplatný súbor")
    p = (DIRS[k] / rest).resolve()
    if DIRS[k].resolve() not in p.parents or not p.exists():
        raise ValueError("súbor neexistuje")
    return p


def safe_name(name, ext=".wav"):
    stem = re.sub(r"[^\w\-+.]+", "_", Path(name).stem, flags=re.UNICODE).strip("._") or "nahravka"
    return stem[:60] + ext


def unique(d: Path, name):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    i = 2
    while p.exists():
        p = d / f"{Path(name).stem}_{i}{Path(name).suffix}"
        i += 1
    return p


def duration(p: Path):
    key = (str(p), p.stat().st_mtime)
    if key not in _cache:
        try:
            _cache[key] = len(load(p)) / SR
        except Exception:
            _cache[key] = 0.0
    return _cache[key]


def list_files():
    groups = {"examples": "Ukážky", "nahrate": "Moje nahrávky", "simulovane": "Simulovaný voice changer",
              "obnovene": "Obnovené (odhalené)"}
    out = []
    for k, d in DIRS.items():
        d.mkdir(parents=True, exist_ok=True)
        key = (lambda q: (q.name.startswith("ref_"), q.name)) if k == "examples" else (lambda q: q.stat().st_mtime)
        for p in sorted(d.iterdir(), key=key):
            if p.suffix.lower() in AUDIO_EXT and p.is_file():
                out.append(dict(id=file_id(p), name=p.stem, group=k, group_sk=groups[k],
                                duration=round(duration(p), 2), ref=p.stem.startswith("ref_")))
    return out


# --------------------------------------------------------------- grafika ---- #
def spectrogram_png(paths_titles, height_per=2.1):
    fig = Figure(figsize=(9.6, height_per * len(paths_titles)), dpi=110, facecolor="#0f1420")
    FigureCanvasAgg(fig)
    axs = fig.subplots(len(paths_titles), 1, sharex=False, squeeze=False)[:, 0]
    for ax, (p, title) in zip(axs, paths_titles):
        x = load(p)
        n, hop = 512, 128
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
        S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6)
        S -= S.max()
        ax.imshow(S.T, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                  extent=[0, len(x) / SR, 0, SR / 2000])
        f0, _ = yin_f0(x.astype(float))
        ax.plot(np.arange(len(f0)) * HOP / SR, np.where(f0 > 0, f0 / 100, np.nan), color="#22d3ee", lw=1.6)
        ax.set_facecolor("#0f1420")
        ax.set_title(title, color="#e5e7eb", fontsize=10, loc="left")
        ax.set_ylabel("kHz", color="#9ca3af", fontsize=8)
        ax.tick_params(colors="#9ca3af", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#374151")
    axs[-1].set_xlabel("čas [s]      azúrová čiara = základná frekvencia F0 (×10)", color="#9ca3af", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def f0_hz(p):
    st = f0_median_st(load(p))
    return None if not np.isfinite(st) else round(100 * 2 ** (st / 12), 1)


# -------------------------------------------------------------- akcie ------- #
def do_analyze(fid, ref_ids):
    p = resolve(fid)
    refs = [load(resolve(r)) for r in ref_ids] if ref_ids else None
    t = time.time()
    r = analyze(p, MODEL, refs)
    if r["verdikt"] == "VOICE CHANGER" and "typ" not in r:
        pass
    r["cas_s"] = round(time.time() - t, 2)
    r["f0_hz"] = f0_hz(p)
    r["dlzka_s"] = round(duration(p), 2)
    r["subor"] = p.stem
    with LOCK:
        r["spektrogram"] = spectrogram_png([(p, p.stem)])
    return r


def do_restore(fid, ref_ids):
    p = resolve(fid)
    refs = [load(resolve(r)) for r in ref_ids] if ref_ids else None
    out = unique(DIRS["obnovene"], safe_name(p.stem + "_obnovene"))
    r = analyze(p, MODEL, refs, restore_to=out)
    if r.get("typ") == "robot":
        out.unlink(missing_ok=True)
        return dict(ok=False, chyba="Robotický efekt nie je možné obnoviť – priebeh výšky hlasu je stratený.")
    res = dict(ok=True, novy=file_id(out), nazov=out.stem, rezim=r.get("odhad_rezim"),
               posun_vysky=r.get("posun_vysky_poltony"), posun_formantov=r.get("posun_formantov_poltony"),
               f0_pred=f0_hz(p), f0_po=f0_hz(out))
    titles = [(p, f"Vstup: {p.stem}"), (out, f"Obnovený hlas ({res['rezim']})")]
    if ref_ids:
        titles.append((resolve(ref_ids[0]), f"Referencia: {resolve(ref_ids[0]).stem}"))
    with LOCK:
        res["spektrogram"] = spectrogram_png(titles)
    return res


SIM = {
    "pitch_shift": "Posun výšky (chipmunk / deep voice)",
    "pitch_formant": "Posun výšky so zachovaním formantov",
    "resample": "Prevzorkovanie (lacné aplikácie)",
    "phase_vocoder": "Fázový vokóder",
    "formant_shift": "Posun formantov (veľkosť hlavy)",
    "robot": "Robot",
    "fx_combo": "Démon / mimozemšťan (výška + efekt)",
}


def do_simulate(fid, ctype, st, fx="chorus"):
    p = resolve(fid)
    x = load(p)
    st = float(st)
    if ctype == "pitch_shift":
        y = pitch_shift(x, st)
    elif ctype == "pitch_formant":
        y = pitch_shift(x, st, preserve_formants=True)
    elif ctype == "resample":
        y = resample_shift(x, st)
    elif ctype == "phase_vocoder":
        y = phase_vocoder_shift(x, st)
    elif ctype == "formant_shift":
        y = formant_shift(x, float(np.clip(2 ** (st / 12), 0.6, 1.6)))
    elif ctype == "robot":
        y = robot(x, float(np.clip(110 * 2 ** (st / 12), 50, 250)))
    elif ctype == "fx_combo":
        fxs = {"chorus": "chorus=0.6:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3", "echo": "aecho=0.8:0.7:40|70:0.4|0.3",
               "flanger": "flanger=delay=2:depth=4:speed=0.6", "vibrato": "vibrato=f=6:d=0.35"}
        y = _fit_len(ffmpeg_filter(pitch_shift(x, st), fxs.get(fx, fxs["chorus"])), len(x))
    else:
        raise ValueError("neznámy typ")
    tag = {"robot": "robot", "formant_shift": f"formanty{st:+.0f}"}.get(ctype, f"{ctype}{st:+.0f}pt")
    out = unique(DIRS["simulovane"], safe_name(f"{p.stem}__{tag}"))
    save(out, normalize(y))
    return dict(ok=True, novy=file_id(out), nazov=out.stem)


def do_upload(name, data):
    ext = Path(name).suffix.lower() or ".webm"
    if ext not in AUDIO_EXT:
        raise ValueError(f"nepodporovaný formát {ext}")
    tmp = unique(DIRS["nahrate"], safe_name(name, ext + ".tmp"))
    tmp.write_bytes(data)
    try:
        x = load(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    if len(x) < SR * 0.5:
        raise ValueError("nahrávka je príliš krátka")
    out = unique(DIRS["nahrate"], safe_name(name))
    save(out, normalize(x))
    return dict(ok=True, novy=file_id(out), nazov=out.stem)


def do_delete(fid):
    p = resolve(fid)
    if fid.startswith("examples/"):
        raise ValueError("ukážky nie je možné zmazať")
    p.unlink()
    return dict(ok=True)


# ---------------------------------------------------------------- server ---- #
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/files":
                return self.send(200, list_files())
            if u.path == "/api/info":
                return self.send(200, dict(model=MODEL["detector_name"], eer=MODEL["cv"]["eer"],
                                           auc=MODEL["cv"]["auc"], typy=SIM))
            if u.path.startswith("/audio/"):
                p = resolve(unquote(u.path[len("/audio/"):]))
                ct = mimetypes.guess_type(p.name)[0] or "audio/wav"
                return self.send(200, p.read_bytes(), ct)
            self.send(404, {"chyba": "nenájdené"})
        except Exception as e:
            self.send(400, {"chyba": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        try:
            if u.path == "/api/upload":
                return self.send(200, do_upload(unquote(self.headers.get("X-Filename", "nahravka.webm")), raw))
            q = json.loads(raw or b"{}")
            if u.path == "/api/analyze":
                return self.send(200, do_analyze(q["id"], q.get("refs") or []))
            if u.path == "/api/restore":
                return self.send(200, do_restore(q["id"], q.get("refs") or []))
            if u.path == "/api/simulate":
                return self.send(200, do_simulate(q["id"], q["typ"], q.get("poltony", 0), q.get("efekt", "chorus")))
            if u.path == "/api/delete":
                return self.send(200, do_delete(q["id"]))
            self.send(404, {"chyba": "nenájdené"})
        except Exception as e:
            traceback.print_exc()
            self.send(400, {"chyba": str(e)})


def free_port(start):
    for p in range(start, start + 30):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print("Načítavam model ...")
    MODEL = joblib.load(ROOT / "models" / "detector.joblib")
    port = free_port(a.port)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Detektor voice changera beží na {url}\n  (okno nechajte otvorené, ukončenie Ctrl+C)\n")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Koniec.")


if __name__ == "__main__":
    main()
