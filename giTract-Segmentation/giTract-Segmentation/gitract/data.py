"""Data access for the UW-Madison GI Tract MRI dataset (Kaggle layout)."""
import glob
import os
import re

import numpy as np
import pandas as pd

from . import ORGANS

ID_RE = re.compile(r"^(case\d+)_(day\d+)_slice_(\d+)$")
FILE_RE = re.compile(r"slice_(\d+)_(\d+)_(\d+)_")


def parse_id(slice_id):
    m = ID_RE.match(slice_id)
    if m is None:
        raise ValueError(f"unexpected slice id: {slice_id}")
    return m.group(1), m.group(2), int(m.group(3))


def rle_decode(rle, shape):
    """Decode Kaggle run-length encoding (1-based start, row-major) to a uint8 mask of `shape` (H, W)."""
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = np.asarray(rle.split(), dtype=np.int64)
        for a, n in zip(s[0::2] - 1, s[1::2]):
            flat[a:a + n] = 1
    return flat.reshape(shape)


def rle_encode(mask):
    flat = np.concatenate([[0], mask.flatten(), [0]])
    runs = np.where(flat[1:] != flat[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))


def build_index(root):
    """One row per annotated slice: id, patient (case), day, slice number, image path, H, W and one RLE column per organ."""
    df = pd.read_csv(os.path.join(root, "train.csv"))
    wide = df.pivot(index="id", columns="class", values="segmentation").reset_index()
    for o in ORGANS:
        if o not in wide:
            wide[o] = np.nan
    wide = wide[wide[list(ORGANS)].notna().any(axis=1)].copy()
    parsed = wide["id"].map(parse_id)
    wide["patient"] = [p[0] for p in parsed]
    wide["day"] = [p[1] for p in parsed]
    wide["slice_no"] = [p[2] for p in parsed]
    paths, heights, widths = [], [], []
    for _, r in wide.iterrows():
        pat = os.path.join(root, "train", r["patient"], f"{r['patient']}_{r['day']}", "scans",
                           f"slice_{r['slice_no']:04d}_*.png")
        hit = glob.glob(pat)
        if len(hit) != 1:
            raise FileNotFoundError(pat)
        m = FILE_RE.search(os.path.basename(hit[0]))
        paths.append(hit[0])
        widths.append(int(m.group(2)))
        heights.append(int(m.group(3)))
    wide["path"], wide["height"], wide["width"] = paths, heights, widths
    return wide.reset_index(drop=True)


def load_slice(row, size=256):
    """Return (image float32 [size,size] in [0,1], masks uint8 [3,size,size]) for one index row."""
    import cv2
    img = cv2.imread(row["path"], cv2.IMREAD_UNCHANGED).astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    shape = (int(row["height"]), int(row["width"]))
    masks = np.stack([
        cv2.resize(rle_decode(row[o], shape), (size, size), interpolation=cv2.INTER_NEAREST) for o in ORGANS
    ])
    return img, masks
