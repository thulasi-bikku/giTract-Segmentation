"""Train one model under one schedule and seed. The test partition is never read here."""
import argparse
import json
import os

import pandas as pd
import torch

from gitract.engine import train
from gitract.schedules import SCHEDULES, build_schedule, expected_exposure

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/index_split.csv")
ap.add_argument("--schedule", choices=SCHEDULES, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs-per-stage", type=int, default=25)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--strata-col", default="stratum_train")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(a.index)
tr = df[df.split == "train"].reset_index(drop=True)
va = df[df.split == "val"].reset_index(drop=True)
stages = build_schedule(a.schedule, tr[a.strata_col].to_numpy(), a.epochs_per_stage, a.seed)
per_group, total = expected_exposure(stages, tr[a.strata_col].to_numpy())
with open(f"{a.out}/exposure.json", "w") as f:
    json.dump({"presentations_per_slice": per_group, "dataset_equivalent_epochs": total, "args": vars(a)}, f, indent=1)
print("expected exposure:", per_group, "dataset-equivalent epochs:", round(total, 2))
train(tr, stages, dev, out_dir=a.out, val_df=va, batch=a.batch, lr=a.lr, workers=a.workers, seed=a.seed)
