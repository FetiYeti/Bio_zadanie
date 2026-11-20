# Detekcia a odhalenie voice changera v reči

Semestrálne zadanie – Biometrické systémy bezpečnosti (TUKE FEI, 2026/27)

Nástroj zistí, či nahrávka reči prešla **voice changerom** (softvérovou zmenou hlasu). Ak áno,
**odhalí** ho: určí typ zmeny, odhadne posun výšky hlasu a formantov v poltónoch a
spätne **obnoví približne pôvodný hlas**. Obnovený hlas je opäť rozpoznateľný systémom
na identifikáciu hovoriaceho.

```
nahrávka ──► príznaky (173) ──► detektor ──► pravý hlas
                                   │
                                   └──► VOICE CHANGER ──► typ zmeny (7 typov)
                                                     ──► posun F0 a formantov [poltóny]
                                                     ──► inverzná transformácia ──► obnovený hlas
```

## Inštalácia

Potrebný je Python ≥ 3.10 a [ffmpeg](https://ffmpeg.org/) (so syntézou flite a filtrom rubberband;
bežné buildy ich obsahujú).

```bash
#(stačí aj, keď v súbore spustíte setup_windows.bat ((ak nemáte  ffmpeg tkak je to potrebné 2 krát po sebe :) ))
pip install -r requirements.txt
ffmpeg -filters | grep -E "flite|rubberband"      # overenie
```

Nepoužívajú sa ťažké knižnice (librosa, torch). Všetky príznaky sú naprogramované v numpy/scipy.

## Grafická aplikácia

```bash
python app.py          # Windows: dvojklik na spusti_aplikaciu.bat 
```

Otvorí sa v prehliadači (lokálne na `http://127.0.0.1:8765`, nič sa neodosiela na internet). V aplikácii sa dá:
- vybrať nahrávku zo zoznamu, pridať vlastný súbor alebo nahrať hlas z mikrofónu,
- prehrať nahrávku, analyzovať ju (verdikt, pravdepodobnosť, typ úpravy, posun výšky, spektrogram),
- označiť referenčné nahrávky (REF) a **odhaliť** voice changer – obnoviť pôvodný hlas a porovnať ho so zachyteným,
- vytvoriť zmenený hlas zo zvolenej nahrávky (7 typov voice changera) a hneď ho otestovať.

## Rýchle použitie z príkazového riadku (natrénovaný model je v `models/`)

```bash
# jedna nahrávka (wav, mp3, m4a, ogg, ...)
python detect.py nahravka.wav

# odhalenie + obnova pôvodného hlasu
python detect.py nahravka.wav --restore obnovena.wav

# presnejší odhad, ak máme registračnú vzorku hlasu tej osoby (verifikácia 1:1)
python detect.py nahravka.wav --reference moj_hlas1.wav moj_hlas2.wav --restore obnovena.wav

# celý priečinok, výsledky do CSV / JSON
python detect.py priecinok/ --csv vysledky.csv
python detect.py nahravka.wav --json
```

Príklad výstupu:

```
=== zmeneny_hlas.wav
  Verdikt:        VOICE CHANGER   (P = 0.947, prah 0.50)
  Typ zmeny:      posun výšky (pitch_shift), istota 1.00
  Posun výšky:    +7.2 poltónov (hlas je vyššie ako pôvodný) [s referenciou]
  Posun formantov:+7.2 poltónov
  Obnovený hlas:  obnovena.wav
```

## Reprodukcia experimentu

```bash
bash run_all.sh            # alebo jednotlivo:
python generate_dataset.py # 1400 nahrávok (~2 min)
python extract_features.py # príznaky (~2 min)
python train_eval.py       # protokoly P0–P5, grafy, models/detector.joblib (~3 min)
python eval_restore.py     # vyhodnotenie obnovy hlasu (~4 min)
python make_figures.py     # spektrogramy do správy
```

Všetky náhodné procesy majú pevný seed, takže výsledky sú reprodukovateľné.

### Vlastné nahrávky (odporúčané)

1. **Vlastné pravé hlasy:** `data/own/<meno>/*.wav`. Jeden podpriečinok zodpovedá jednému
   hovoriacemu. Stačí 20–40 viet na osobu (napr. nahlas čítať `data/texts.txt`). Skript na ne
   aplikuje všetky voice changery a pridá ich do tréningu.
2. **Skutočné aplikácie (externý test):** nahrávky prehnané napr. cez Voicemod, Clownfish,
   MorphVOX alebo mobilnú aplikáciu uložte do `data/real_app/<aplikácia_efekt>/*.wav` a pravé
   nahrávky do `data/real_app/genuine/`. Tieto nahrávky sa **nepoužijú na tréning**, iba na
   nezávislý test (`results/external_test.csv`).

Potom stačí znova spustiť `bash run_all.sh`.

## Štruktúra repozitára

| súbor | obsah |
|---|---|
| `vcdetect/audio.py` | načítanie ľubovoľného formátu cez ffmpeg, kodeky, filtre |
| `vcdetect/changers.py` | 7 simulovaných voice changerov + 5 prenosových kanálov |
| `vcdetect/features.py` | príznaky: MFCC, LFCC, spektrálne, LTAS, F0 (YIN), jitter, shimmer, LPC formanty, fáza, modulácie, HNR po pásmach, CPP, spektrálny cutoff |
| `vcdetect/restore.py` | odhad posunov (naslepo / s referenciou) a inverzná transformácia |
| `generate_dataset.py` | tvorba datasetu |
| `extract_features.py` | výpočet príznakov |
| `train_eval.py` | tréning, protokoly P0–P5, grafy, finálny model |
| `eval_restore.py` | vyhodnotenie obnovy hlasu |
| `detect.py` | nástroj pre používateľa |
| `results/` | tabuľky (`summary.md`, `restore_summary.md`, `*.csv`) a grafy (`*.png`) |

## Simulované voice changery

| typ | princíp | napodobňuje |
|---|---|---|
| `pitch_shift` | posun výšky aj formantov (rubberband) | „chipmunk“, „deep voice“ |
| `pitch_formant` | posun výšky so zachovaním formantov | kvalitné efekty muž ↔ žena |
| `resample` | zmena vzorkovacej frekvencie + WSOLA | lacné aplikácie, staré efekty |
| `phase_vocoder` | fázový vokóder + prevzorkovanie | klasické DSP pitch shiftery |
| `formant_shift` | posun iba formantov | „väčšia/menšia hlava“, zmena pohlavia |
| `robot` | fáza nahradená konštantou | robotický hlas |
| `fx_combo` | posun výšky + chorus/flanger/echo/vibrato | „démon“, „mimozemšťan“ |

Posun je náhodný, 2,5–9 poltónov nahor alebo nadol. Na pravé aj zmenené nahrávky sa rovnako
aplikuje náhodný kanál (čistý, MP3, Opus, telefón G.711 + pásmo 300–3400 Hz, šum 15–30 dB SNR),
aby sa model nenaučil rozpoznávať kodek namiesto voice changera.

## Hlavné výsledky

Dataset: 1 400 nahrávok (700 pravých, 700 zmenených), 4 hovoriaci (syntetické hlasy flite).

| scenár | metrika | výsledok |
|---|---|---|
| P0 známy hovoriaci (verifikácia 1:1) | EER | **0,9 %** (SVM), 2,6 % (RF) |
| P1 neznámy hovoriaci (LOSO) | EER / AUC | **11,8 %** / 0,944 (RandomForest) |
| P2 neznámy typ útoku | detegované | 76–100 % podľa typu |
| P3 vyradenie príznakov identity | EER | 14,7 % → 11,8 % |
| P4 určenie typu (8 tried, LOSO) | presnosť | 63,6 % |
| Obnova – identifikácia hovoriaceho | bez obnovy → naslepo → s referenciou | **61 % → 72 % → 92 %** |
| Obnova – chyba F0 (medián) | bez obnovy → s referenciou | 4,6 → **0,36** poltónu |

Podrobnosti sú v `results/summary.md` a `results/restore_summary.md`.

## Obmedzenia (dôležité pre diskusiu)

- **Pravá reč je syntetická (TTS flite)**, keďže v čase vývoja nebol dostupný rečový korpus.
  Model natrénovaný iba na TTS nemusí dobre fungovať na prirodzenej reči. Preto je pripravený
  priečinok `data/own` na vlastné nahrávky.
- **Málo hovoriacich (4, z toho 1 ženský hlas).** Pri teste na ženskom hlase (`slt`) je EER
  24 %, lebo model sa z mužských hlasov čiastočne naučil, že „vysoký hlas = zmenený“. Vyradenie
  príznakov identity (P3) to zmierňuje, ale riešením sú až ďalší hovoriaci.
- **Voice changery sú simulované.** Skutočné aplikácie môžu používať iné algoritmy, napríklad
  neurónovú konverziu hlasu (RVC, so-vits-svc). Na tie je potrebný externý test (`data/real_app`).
- Robotický efekt sa deteguje spoľahlivo, ale spätne obnoviť ho nemožno, pretože priebeh F0 je
  nenávratne stratený.
- Odhad posunu **naslepo** je hrubý (MAE ~3 poltóny). Spoľahlivú obnovu dáva iba režim
  s referenčnou nahrávkou.
