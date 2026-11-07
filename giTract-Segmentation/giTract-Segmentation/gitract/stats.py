"""Patient-level statistics."""
import numpy as np
import pandas as pd

from . import GROUPS, PREVALENCE


def prevalence_weighted(group_means, prevalence=PREVALENCE):
    """Eq. (14) for group means given in the order Easy, Medium, Hard."""
    return float(np.dot(group_means, prevalence))


def cluster_bootstrap(per_patient, col, n_boot=2000, seed=0, alpha=0.05):
    """Mean over patients of `col` (already averaged within patient) with a percentile cluster-bootstrap CI."""
    x = per_patient[col].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def group_summary(df, metric="dice", group_col="stratum", n_boot=2000, seed=0):
    """Patient-level mean and CI of `metric` for every stratum and for all test slices."""
    out = {}
    for g in list(GROUPS) + ["All"]:
        sub = df if g == "All" else df[df[group_col] == g]
        if sub.empty:
            continue
        pp = sub.groupby("patient")[metric].mean().reset_index()
        out[g] = cluster_bootstrap(pp, metric, n_boot, seed) + (int(pp.shape[0]),)
    return out


def paired_difference(df_a, df_b, metric="dice", group_col="stratum", n_boot=2000, seed=0):
    """Paired patient-level difference A - B (Eq. 15) with cluster-bootstrap CI per stratum.
    df_a and df_b are per-slice tables over the same 'id' set."""
    m = df_a[["id", "patient", group_col, metric]].merge(df_b[["id", metric]], on="id", suffixes=("_a", "_b"))
    m["diff"] = m[f"{metric}_a"] - m[f"{metric}_b"]
    out = {}
    for g in list(GROUPS) + ["All"]:
        sub = m if g == "All" else m[m[group_col] == g]
        if sub.empty:
            continue
        pp = sub.groupby("patient")["diff"].mean().reset_index()
        out[g] = cluster_bootstrap(pp, "diff", n_boot, seed) + (int(pp.shape[0]),)
    return out  # {group: (mean, lo, hi, n_patients)}
