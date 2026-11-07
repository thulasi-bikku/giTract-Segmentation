"""Overlap and distance metrics."""
import numpy as np
from scipy import ndimage


def dice(p, g):
    inter = float((p & g).sum())
    den = float(p.sum() + g.sum())
    return 2 * inter / den if den > 0 else 1.0


def iou(p, g):
    union = float((p | g).sum())
    return float((p & g).sum()) / union if union > 0 else 1.0


def hd95(p, g):
    """95th percentile symmetric surface distance in pixels; NaN if either mask is empty."""
    if not p.any() or not g.any():
        return float("nan")
    dp = ndimage.distance_transform_edt(~p)
    dg = ndimage.distance_transform_edt(~g)
    surf_p = p & ~ndimage.binary_erosion(p)
    surf_g = g & ~ndimage.binary_erosion(g)
    d = np.concatenate([dg[surf_p], dp[surf_g]])
    return float(np.percentile(d, 95))


def slice_metrics(pred, gt):
    """pred, gt: bool arrays [3,H,W]. Averages over the annotated organs of the slice."""
    d, i, h = [], [], []
    for k in range(gt.shape[0]):
        if gt[k].any():
            d.append(dice(pred[k], gt[k]))
            i.append(iou(pred[k], gt[k]))
            h.append(hd95(pred[k], gt[k]))
    return {"dice": float(np.mean(d)), "iou": float(np.mean(i)),
            "hd95": float(np.nanmean(h)) if not np.all(np.isnan(h)) else float("nan")}
