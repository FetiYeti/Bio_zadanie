#!/usr/bin/env bash
# Kompletná reprodukcia experimentu
set -e
python generate_dataset.py
python extract_features.py
python train_eval.py
python eval_restore.py
python make_figures.py
echo "Hotovo – výsledky v results/, model v models/detector.joblib"
