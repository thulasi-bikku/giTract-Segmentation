"""Patient-disjoint split and baseline-independent difficulty strata.

* patients are assigned to train / val / test (no patient in more than one partition);
* S2 min/max, N_max and the group thresholds are estimated on TRAINING patients only;
* `stratum_eval` is defined from S1-S3 only (no trained model) and is used to report results;
* with --oof-dice, `stratum_train` uses the full S1-S4 score with out-of-fold S4 for the training schedule;
  otherwise `stratum_train` equals `stratum_eval`.
"""
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from gitract import ORGANS
from gitract.data import load_slice
from gitract.difficulty import (assign_groups, fit_normalizer, fit_thresholds, slice_features, slice_score)
from gitract.splits import assert_disjoint, patient_split

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/index.csv")
ap.add_argument("--out", default="data/index_split.csv")
ap.add_argument("--stats-out", default="data/difficulty_stats.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.1, 0.2))
ap.add_argument("--oof-dice", default=None, help="CSV (id, organ, dice) of out-of-fold baseline predictions")
a = ap.parse_args()

df = pd.read_csv(a.index)
df["split"] = df["patient"].map(patient_split(df["patient"], tuple(a.fractions), a.seed))
assert_disjoint(df)

feats = []
for _, r in tqdm(df.iterrows(), total=len(df), desc="features"):
    img, msk = load_slice(r)
    feats.append(slice_features(img, msk))
tr = df["split"].to_numpy() == "train"
norm = fit_normalizer([f for f, t in zip(feats, tr) if t])

ind = np.array([slice_score(f, norm) for f in feats])
thr = fit_thresholds(ind[tr])
df["score_indep"] = ind
df["stratum_eval"] = assign_groups(ind, thr)

if a.oof_dice:
    oof = pd.read_csv(a.oof_dice)
    lookup = {}
    for _, r in oof.iterrows():
        lookup.setdefault(r["id"], {})[r["organ"]] = 1.0 - r["dice"]
    full = np.array([slice_score(f, norm, s4=lookup.get(i)) for f, i in zip(feats, df["id"])])
    thr_full = fit_thresholds(full[tr])
    df["score_full"] = full
    df["stratum_train"] = assign_groups(full, thr_full)
else:
    thr_full = thr
    df["stratum_train"] = df["stratum_eval"]

df.to_csv(a.out, index=False)
with open(a.stats_out, "w") as f:
    json.dump({"normalizer": norm, "thresholds_indep": thr, "thresholds_full": thr_full,
               "n_patients": {s: int(df.loc[df.split == s, "patient"].nunique()) for s in ("train", "val", "test")},
               "n_slices": {s: int((df.split == s).sum()) for s in ("train", "val", "test")},
               "test_stratum_counts": df.loc[df.split == "test", "stratum_eval"].value_counts().to_dict()}, f, indent=1)
print(open(a.stats_out).read())
