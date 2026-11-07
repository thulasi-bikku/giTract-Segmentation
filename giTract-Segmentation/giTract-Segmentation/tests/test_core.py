import numpy as np
import pandas as pd

from gitract import GROUPS, PREVALENCE
from gitract.data import rle_decode, rle_encode
from gitract.difficulty import (assign_groups, fit_normalizer, fit_thresholds, slice_features, slice_score)
from gitract.metrics import dice, hd95, iou
from gitract.schedules import SCHEDULES, build_schedule, expected_exposure
from gitract.splits import assert_disjoint, kfold_patients, patient_split
from gitract.stats import cluster_bootstrap, paired_difference, prevalence_weighted


def groups(n=10000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(GROUPS, size=n, p=PREVALENCE)


def test_rle_roundtrip():
    m = np.zeros((20, 30), np.uint8)
    m[3:7, 5:9] = 1
    m[15, :] = 1
    assert (rle_decode(rle_encode(m), m.shape) == m).all()


def test_split_disjoint():
    pats = [f"case{i}" for i in range(85) for _ in range(5)]
    sp = patient_split(pats, seed=1)
    df = pd.DataFrame({"patient": pats})
    df["split"] = df["patient"].map(sp)
    assert_disjoint(df)
    assert set(df.split) == {"train", "val", "test"}
    f = kfold_patients(pats, 5)
    assert len(set(f.values())) == 5


def test_difficulty_train_only_and_prevalence():
    rng = np.random.default_rng(0)
    feats = []
    for _ in range(400):
        img = rng.random((64, 64))
        m = np.zeros((3, 64, 64), np.uint8)
        a = rng.integers(2, 30)
        m[0, :a, :a] = 1
        m[1, 40:40 + a // 2, 40:50] = 1
        feats.append(slice_features(img, m))
    norm = fit_normalizer(feats[:300])
    sc = np.array([slice_score(f, norm) for f in feats])
    assert ((sc >= 0) & (sc <= 1)).all()
    thr = fit_thresholds(sc[:300])
    g = assign_groups(sc[:300], thr)
    frac = [(g == x).mean() for x in GROUPS]
    assert np.allclose(frac, PREVALENCE, atol=0.01)
    s4 = {"large_bowel": 0.9}
    assert slice_score(feats[0], norm, s4=s4) >= 0


def test_exposure_matches_paper():
    g = groups()
    pg, total = expected_exposure(build_schedule("curriculum", g), g)
    assert np.allclose([pg[x] for x in GROUPS], [75, 50, 25])
    assert abs(total - 68.4) < 0.5
    pb, tb = expected_exposure(build_schedule("baseline", g), g)
    assert np.allclose([pb[x] for x in GROUPS], 75) and abs(tb - 75) < 1e-9


def test_controls_are_exposure_matched():
    g = groups()
    pg, total = expected_exposure(build_schedule("matched_exposure", g), g)
    assert np.allclose([pg[x] for x in GROUPS], [75, 50, 25], atol=0.5)
    pr, tr = expected_exposure(build_schedule("random_stage", g, seed=3), g)
    assert abs(tr - total) < 0.5               # same total amount of training as the curriculum
    ps, _ = expected_exposure(build_schedule("random_stage", g, seed=3), g)
    spread = max(ps.values()) - min(ps.values())
    assert spread < 6                          # exposure no longer depends on difficulty
    rv, _ = expected_exposure(build_schedule("reversed", g), g)
    assert np.allclose([rv[x] for x in GROUPS], [25, 50, 75])
    for s in SCHEDULES:
        assert build_schedule(s, g)


def test_metrics():
    a = np.zeros((40, 40), bool); a[10:20, 10:20] = True
    b = np.zeros((40, 40), bool); b[12:22, 10:20] = True
    assert abs(dice(a, a) - 1) < 1e-9 and iou(a, a) == 1
    assert 0 < dice(a, b) < 1 and hd95(a, a) == 0 and hd95(a, b) > 0


def test_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(30):
        for s in range(10):
            base = rng.normal(0.7, 0.1)
            rows.append(dict(id=f"p{p}_{s}", patient=f"p{p}", stratum="Hard", dice=base))
    a = pd.DataFrame(rows)
    b = a.copy()
    b["dice"] = a["dice"] + rng.normal(0, 0.01, len(a))
    m, lo, hi, n = paired_difference(a, b, n_boot=500)["Hard"]
    assert lo < 0 < hi and n == 30
    c = a.copy(); c["dice"] = a["dice"] + 0.05
    m, lo, hi, n = paired_difference(c, a, n_boot=500)["Hard"]
    assert lo > 0
    assert abs(prevalence_weighted([0.9050, 0.7961, 0.5280]) - 0.8678) < 1e-3
