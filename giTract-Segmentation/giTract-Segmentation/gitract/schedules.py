"""Training schedules: the curriculum and the control schedules specified in Section V-F."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import GROUPS


@dataclass
class Stage:
    epochs: int
    indices: np.ndarray                    # positions in the training dataframe used in this stage
    weights: Optional[np.ndarray] = None   # per-slice sampling weights (full length) if weighted sampling
    samples_per_epoch: int = 0             # used with weights


def _idx(groups, wanted):
    return np.where(np.isin(groups, wanted))[0]


def presentation_counts(name, epochs_per_stage=25):
    """Eq. (10): presentations of a slice of each group for the nested curricula."""
    E, M, H = GROUPS
    s = epochs_per_stage
    if name == "curriculum":
        return {E: 3 * s, M: 2 * s, H: s}
    if name == "reversed":
        return {E: s, M: 2 * s, H: 3 * s}
    raise ValueError(name)


def build_schedule(name, groups, epochs_per_stage=25, seed=0, hard_factor=3.0):
    """groups: array of 'Easy'/'Medium'/'Hard' for the training slices. Returns list[Stage]."""
    g = np.asarray(groups)
    N = len(g)
    E, M, H = GROUPS
    T = 3 * epochs_per_stage
    if name == "baseline":
        return [Stage(T, np.arange(N))]
    if name == "curriculum":
        return [Stage(epochs_per_stage, _idx(g, [E])),
                Stage(epochs_per_stage, _idx(g, [E, M])),
                Stage(epochs_per_stage, np.arange(N))]
    if name == "reversed":
        return [Stage(epochs_per_stage, _idx(g, [H])),
                Stage(epochs_per_stage, _idx(g, [H, M])),
                Stage(epochs_per_stage, np.arange(N))]
    if name == "random_stage":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        n1 = int((g == E).sum())
        n2 = n1 + int((g == M).sum())
        return [Stage(epochs_per_stage, np.sort(perm[:n1])),
                Stage(epochs_per_stage, np.sort(perm[:n2])),
                Stage(epochs_per_stage, np.arange(N))]
    if name == "matched_exposure":
        n_c = presentation_counts("curriculum", epochs_per_stage)
        w = np.array([n_c[x] for x in g], dtype=float)
        spe = int(round(w.sum() / T))
        return [Stage(T, np.arange(N), weights=w, samples_per_epoch=spe)]
    if name == "oversample_hard":
        w = np.where(g == H, hard_factor, 1.0)
        return [Stage(T, np.arange(N), weights=w, samples_per_epoch=N)]
    raise ValueError(f"unknown schedule: {name}")


SCHEDULES = ("baseline", "curriculum", "reversed", "random_stage", "matched_exposure", "oversample_hard")


def expected_exposure(stages, groups):
    """Expected presentations per slice for each group, and dataset-equivalent epochs (Eq. 11)."""
    g = np.asarray(groups)
    N = len(g)
    per_group = {x: 0.0 for x in GROUPS}
    total = 0.0
    for st in stages:
        if st.weights is not None:
            p = st.weights / st.weights.sum()
            draws = st.samples_per_epoch * st.epochs
            for x in GROUPS:
                m = g == x
                if m.any():
                    per_group[x] += draws * p[m].sum() / m.sum()
            total += draws / N
        else:
            members = np.zeros(N, bool)
            members[st.indices] = True
            for x in GROUPS:
                m = g == x
                if m.any():
                    per_group[x] += st.epochs * members[m].mean()
            total += st.epochs * len(st.indices) / N
    return per_group, total
