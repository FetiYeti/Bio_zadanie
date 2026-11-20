#!/usr/bin/env python3
"""Vypočíta príznaky pre všetky nahrávky z metadata.csv -> features.npz

Použitie:
  python extract_features.py --meta data/generated/metadata.csv --out data/generated/features.npz
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vcdetect.audio import load
from vcdetect.features import extract, feature_names


def one(path):
    return extract(load(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/generated/metadata.csv")
    ap.add_argument("--out", default="data/generated/features.npz")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    df = pd.read_csv(a.meta)
    t = time.time()
    X = np.vstack(Parallel(n_jobs=a.jobs)(delayed(one)(f) for f in df.file))
    print(f"{X.shape[0]} nahrávok x {X.shape[1]} príznakov za {time.time() - t:.1f} s")
    np.savez_compressed(a.out, X=X, names=np.array(feature_names()))


if __name__ == "__main__":
    main()
