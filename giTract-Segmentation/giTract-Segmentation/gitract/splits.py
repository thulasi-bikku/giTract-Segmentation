"""Patient-disjoint partitions."""
import numpy as np


def patient_split(patients, fractions=(0.7, 0.1, 0.2), seed=0):
    """Assign each unique patient to train/val/test. Returns {patient: split}."""
    uniq = np.array(sorted(set(patients)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    out = {}
    for i, p in enumerate(uniq):
        out[p] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


def kfold_patients(patients, k=5, seed=0):
    """Return {patient: fold} with folds balanced in patient count."""
    uniq = np.array(sorted(set(patients)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return {p: i % k for i, p in enumerate(uniq)}


def assert_disjoint(df, col="split", patient="patient"):
    sets = [set(df.loc[df[col] == s, patient]) for s in df[col].unique()]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "patients shared between partitions"
