"""Training and evaluation loops."""
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler

from .data import load_slice
from .losses import FocalDiceLoss
from .metrics import slice_metrics
from .models import AGTSNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SliceDataset(Dataset):
    def __init__(self, df, size=256):
        self.df = df.reset_index(drop=True)
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img, msk = load_slice(self.df.iloc[i], self.size)
        return torch.from_numpy(img)[None], torch.from_numpy(msk).float(), i


def make_loader(ds, stage, batch, workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    if stage.weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(stage.weights, dtype=torch.double),
                                        num_samples=stage.samples_per_epoch, replacement=True, generator=g)
    else:
        sampler = SubsetRandomSampler(stage.indices.tolist(), generator=g)
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=True)


@torch.no_grad()
def evaluate(model, df, device, batch=16, workers=4, size=256, thr=0.5):
    """Per-slice metrics (Dice, IoU, HD95 averaged over annotated organs) for every row of df."""
    model.eval()
    ds = SliceDataset(df, size)
    rows = []
    for x, y, idx in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers):
        pred = (torch.sigmoid(model(x.to(device))) > thr).cpu().numpy().astype(bool)
        gt = y.numpy().astype(bool)
        for k in range(len(idx)):
            rows.append(slice_metrics(pred[k], gt[k]))
    out = df[[c for c in ("id", "patient", "stratum_eval", "stratum_train") if c in df.columns]].reset_index(drop=True).copy()
    for key in ("dice", "iou", "hd95"):
        out[key] = [r[key] for r in rows]
    return out


def train(train_df, stages, device, out_dir=None, val_df=None, batch=8, lr=1e-5, workers=4, seed=0, size=256, amp=True):
    """Run the stages in order; each stage starts from the weights of the previous one. Returns the model."""
    seed_everything(seed)
    model = AGTSNet().to(device)
    loss_fn = FocalDiceLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    ds = SliceDataset(train_df, size)
    log, ep_global = [], 0
    for s_i, st in enumerate(stages):
        loader = make_loader(ds, st, batch, workers, seed + 1000 * s_i)
        for ep in range(st.epochs):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * x.size(0)
                n += x.size(0)
            rec = {"epoch": ep_global, "stage": s_i, "train_loss": tot / max(n, 1), "sec": time.time() - t0}
            if val_df is not None:
                v = evaluate(model, val_df, device, workers=workers, size=size)
                rec["val_dice"] = float(v["dice"].mean())
            log.append(rec)
            ep_global += 1
            if out_dir:
                with open(f"{out_dir}/train_log.json", "w") as f:
                    json.dump(log, f, indent=1)
    if out_dir:
        torch.save(model.state_dict(), f"{out_dir}/final.pt")
    return model
