"""Summarize runs: mean over seeds, patient-level cluster-bootstrap CIs, paired differences between schedules.

Example: python scripts/06_analyze.py --runs runs/baseline_s* --ref runs/curriculum_s*
"""
import argparse
import glob

import pandas as pd

from gitract.stats import group_summary, paired_difference

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="run dirs of model A (glob patterns allowed)")
ap.add_argument("--ref", nargs="+", default=None, help="run dirs of model B for the paired difference A - B")
ap.add_argument("--metric", default="dice")
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()


def load(patterns):
    dirs = sorted(d for p in patterns for d in glob.glob(p))
    return [pd.read_csv(f"{d}/test_per_slice.csv") for d in dirs], dirs


def seed_mean(frames):
    base = frames[0][["id", "patient", "stratum"]].copy()
    base[a.metric] = sum(f.set_index("id").loc[base["id"], a.metric].to_numpy() for f in frames) / len(frames)
    return base


A, dirs_a = load(a.runs)
print(f"A: {len(A)} runs")
mA = seed_mean(A)
for g, (m, lo, hi, n) in group_summary(mA, a.metric, n_boot=a.n_boot).items():
    print(f"{g:7s} {a.metric} {m:.4f} [{lo:.4f}, {hi:.4f}]  patients={n}")
if a.ref:
    B, _ = load(a.ref)
    mB = seed_mean(B)
    print(f"\nPaired difference (A - B), {len(B)} reference runs")
    for g, (m, lo, hi, n) in paired_difference(mA, mB, a.metric, n_boot=a.n_boot).items():
        print(f"{g:7s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  patients={n}")
