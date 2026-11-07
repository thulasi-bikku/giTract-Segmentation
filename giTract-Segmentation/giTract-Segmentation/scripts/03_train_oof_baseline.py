"""Out-of-fold baseline Dice for the training patients (used as S4 in the training schedule).

For each of K folds of TRAINING patients, a baseline AGTSNet is trained on the other folds and predicts the held-out
fold. Validation and test patients are never used. Writes a CSV (id, organ, dice)."""
import argparse

import pandas as pd
import torch

from gitract import ORGANS
from gitract.data import load_slice
from gitract.engine import SliceDataset, seed_everything, train
from gitract.metrics import dice
from gitract.schedules import build_schedule
from gitract.splits import kfold_patients

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/index_split.csv")
ap.add_argument("--out", default="data/oof_dice.csv")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs-per-stage", type=int, default=25)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(a.index)
tr = df[df.split == "train"].reset_index(drop=True)
fold = tr["patient"].map(kfold_patients(tr["patient"], a.k, a.seed))
rows = []
for k in range(a.k):
    fit, held = tr[fold != k].reset_index(drop=True), tr[fold == k].reset_index(drop=True)
    stages = build_schedule("baseline", ["Easy"] * len(fit), a.epochs_per_stage, a.seed)
    model = train(fit, stages, dev, batch=a.batch, workers=a.workers, seed=a.seed + k)
    model.eval()
    with torch.no_grad():
        for i in range(len(held)):
            img, msk = load_slice(held.iloc[i])
            p = (torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev)))[0] > 0.5).cpu().numpy()
            for j, o in enumerate(ORGANS):
                if msk[j].any():
                    rows.append({"id": held.loc[i, "id"], "organ": o, "dice": dice(p[j], msk[j].astype(bool))})
    print(f"fold {k}: done")
pd.DataFrame(rows).to_csv(a.out, index=False)
