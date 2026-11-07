"""Composite slice difficulty (Eqs. 4-9 of the paper) with train-only normalization.

Two variants are supported:
  * independent score: S1, S2, S3 only (no trained model involved); used to define test strata;
  * full score: S1-S4, where S4 = 1 - Dice of an *out-of-fold* baseline prediction.
"""
import numpy as np
from scipy import ndimage

from . import GROUPS, ORGANS, PREVALENCE


def s1_size(mask):
    return 1.0 - float(mask.sum()) / mask.size


def s2_entropy(img):
    """Shannon entropy (bits) of the 256-bin intensity histogram of an image scaled to [0, 1]."""
    h, _ = np.histogram(np.clip(img, 0, 1), bins=256, range=(0, 1))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def n_components(mask):
    return int(ndimage.label(mask)[1])


def slice_features(img, masks):
    """Per-slice raw features. `masks` has shape [3, H, W]; only non-empty organs are annotated."""
    feats = {"s2": s2_entropy(img)}
    for k, o in enumerate(ORGANS):
        m = masks[k]
        if m.any():
            feats[f"s1_{o}"] = s1_size(m)
            feats[f"ncomp_{o}"] = n_components(m)
    return feats


def fit_normalizer(rows):
    """Estimate S2 min/max and N_max from training slices only."""
    s2 = np.array([r["s2"] for r in rows])
    nmax = max(v for r in rows for k, v in r.items() if k.startswith("ncomp_"))
    return {"s2_min": float(s2.min()), "s2_max": float(s2.max()), "n_max": int(max(nmax, 1))}


def slice_score(feats, norm, s4=None, weights=None):
    """Highest organ-wise composite score ('most complex organ' rule).
    s4: optional dict organ -> (1 - out-of-fold Dice). Without it the independent 3-component score is used."""
    rng = norm["s2_max"] - norm["s2_min"]
    s2n = float(np.clip((feats["s2"] - norm["s2_min"]) / rng, 0, 1)) if rng > 0 else 0.0
    best = -1.0
    for o in ORGANS:
        if f"s1_{o}" not in feats:
            continue
        parts = [feats[f"s1_{o}"], s2n, min(feats[f"ncomp_{o}"] / norm["n_max"], 1.0)]
        w = list(weights[:3]) if weights is not None else [1.0] * 3
        if s4 is not None and o in s4:
            parts.append(float(s4[o]))
            w = list(weights) if weights is not None else [1.0] * 4
        best = max(best, float(np.dot(parts, w) / sum(w)))
    return best


def fit_thresholds(train_scores, prevalence=PREVALENCE):
    """Cut points at the cumulative prevalence of Easy and Easy+Medium, estimated on training slices."""
    c = np.cumsum(prevalence)[:2]
    return tuple(float(np.quantile(train_scores, q)) for q in c)


def assign_groups(scores, thresholds):
    t1, t2 = thresholds
    s = np.asarray(scores)
    return np.where(s < t1, GROUPS[0], np.where(s < t2, GROUPS[1], GROUPS[2]))
