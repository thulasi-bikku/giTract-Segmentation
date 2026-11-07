"""Build the slice index (one row per annotated slice) from the Kaggle UW-Madison GI Tract data."""
import argparse

from gitract.data import build_index

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True, help="folder containing train.csv and train/")
ap.add_argument("--out", default="data/index.csv")
a = ap.parse_args()
df = build_index(a.data_root)
df.to_csv(a.out, index=False)
print(f"{len(df)} annotated slices from {df.patient.nunique()} patients -> {a.out}")
