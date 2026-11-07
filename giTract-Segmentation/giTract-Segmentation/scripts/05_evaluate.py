"""Evaluate a trained model on the held-out test patients (all test slices, natural distribution)."""
import argparse

import pandas as pd
import torch

from gitract.engine import evaluate
from gitract.models import AGTSNet

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/index_split.csv")
ap.add_argument("--run", required=True, help="run directory containing final.pt")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(a.index)
te = df[df.split == "test"].reset_index(drop=True).rename(columns={})
te["stratum"] = te["stratum_eval"]
model = AGTSNet().to(dev)
model.load_state_dict(torch.load(f"{a.run}/final.pt", map_location=dev))
res = evaluate(model, te, dev, workers=a.workers)
res["stratum"] = te["stratum"].to_numpy()
res.to_csv(f"{a.run}/test_per_slice.csv", index=False)
print(res.groupby("stratum")[["dice", "iou", "hd95"]].mean())
