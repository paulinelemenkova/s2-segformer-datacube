from __future__ import annotations
import argparse
import json
import os
import time
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from common import device

N_BANDS, N_CLASSES, IGNORE = (10, 6, 255)


class GCELoss(nn.Module):

    def __init__(self, q: float = 0.7, ignore_index: int = IGNORE):
        super().__init__()
        self.q, self.ignore_index = (q, ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        prob = F.softmax(logits, dim=1)
        p_y = prob.gather(1, target.clamp(max=logits.shape[1] - 1).unsqueeze(1)).squeeze(1)
        loss = (1.0 - p_y.clamp_min(1e-07) ** self.q) / self.q
        return loss[valid].mean()


@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, t: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    probs = torch.stack([F.softmax(upsampled_logits(model, x), dim=1) for _ in range(t)]).mean(0)
    entropy = -(probs * probs.clamp_min(1e-08).log()).sum(1)
    return (probs, entropy)


def upsampled_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "segformer"):
        return model(x)
    logits = model(pixel_values=x).logits
    return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build(arch: str = "segformer", pretrained: bool = True) -> nn.Module:
    if arch == "unet":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            "resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=N_BANDS, classes=N_CLASSES
        )
    return build_model(pretrained)


def run_name(arch: str, q: float, fold: int) -> str:
    return f"{arch}{('' if abs(q - 0.7) < 1e-09 else f'_q{q:g}')}_fold{fold}"


def build_model(pretrained: bool = True) -> SegformerForSemanticSegmentation:
    cfg = SegformerConfig.from_pretrained(
        "nvidia/mit-b2",
        num_labels=N_CLASSES,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
    )
    if not pretrained:
        model = SegformerForSemanticSegmentation(cfg)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", config=cfg, ignore_mismatched_sizes=True
        )
    name, old = next(
        ((n, m) for n, m in model.segformer.named_modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)
    )
    new = nn.Conv2d(N_BANDS, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, N_BANDS, 1, 1) * 3 / N_BANDS)
        new.bias.copy_(old.bias)
    parent, attr = name.rsplit(".", 1)
    setattr(model.segformer.get_submodule(parent), attr, new)
    model.config.num_channels = N_BANDS
    return model


class Chips(Dataset):

    def __init__(self, files: list[str], mean: np.ndarray, sd: np.ndarray, augment: bool):
        self.files, self.mean, self.sd = (files, mean[:, None, None], sd[:, None, None])
        self.aug = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()]) if augment else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as z:
            x, y = (z["x"].astype("float32"), z["y"])
        if self.augment:
            x = x * np.random.uniform(0.95, 1.05)
            out = self.aug(image=x.transpose(1, 2, 0), mask=y)
            x, y = (out["image"].transpose(2, 0, 1), out["mask"])
        x = (x - self.mean) / self.sd
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long())


def band_stats(files: list[str], n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    xs = []
    for f in rng.choice(files, size=min(n, len(files)), replace=False):
        with np.load(f) as z:
            xs.append(z["x"].astype("float32").reshape(N_BANDS, -1)[:, ::16])
    x = np.concatenate(xs, axis=1)
    return (x.mean(1), x.std(1) + 1e-06)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="chips.csv")
    ap.add_argument("--chip_dir", default="chips")
    ap.add_argument("--arch", choices=("segformer", "unet"), default="segformer")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--q", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default 6e-5 (SegFormer), 1e-4 (U-Net)")
    ap.add_argument("--max_chips", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true", help="random initialisation (offline tests only)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    lr = a.lr or (0.0001 if a.arch == "unet" else 6e-05)
    name = run_name(a.arch, a.q, a.fold)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(a.chips)
    df = df[df.year != 2025]
    tr = df if a.fold < 0 else df[df.fold != a.fold]
    va = df[df.fold == a.fold] if a.fold >= 0 else df.iloc[:0]
    if len(tr) > a.max_chips:
        tr = tr.sample(a.max_chips, random_state=42)
    if len(va) > a.max_chips // 4:
        va = va.sample(a.max_chips // 4, random_state=42)
    f_tr = [f"{a.chip_dir}/{c}.npz" for c in tr.chip]
    f_va = [f"{a.chip_dir}/{c}.npz" for c in va.chip]
    mean, sd = band_stats(f_tr)
    dev = device()
    amp = dev.type == "cuda"
    model = build(a.arch, not a.no_pretrained).to(dev)
    dl_tr = DataLoader(
        Chips(f_tr, mean, sd, True),
        batch_size=a.batch,
        shuffle=True,
        num_workers=a.workers,
        drop_last=True,
        persistent_workers=a.workers > 0,
    )
    dl_va = DataLoader(Chips(f_va, mean, sd, False), batch_size=a.batch, num_workers=a.workers) if f_va else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=a.epochs * len(dl_tr), power=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = GCELoss(a.q)
    log, best = ([], np.inf)
    print(f"{name}: {len(f_tr)} training / {len(f_va)} validation chips on {dev}", flush=True)
    for ep in range(a.epochs):
        t0, tot = (time.time(), 0.0)
        model.train()
        for x, y in dl_tr:
            x, y = (x.to(dev), y.to(dev))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                lg = upsampled_logits(model, x)
            loss = crit(lg.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x)
        rec = {"run": name, "epoch": ep + 1, "train_loss": tot / (len(dl_tr) * a.batch), "sec": round(time.time() - t0)}
        if dl_va is not None:
            model.eval()
            vl, agree, n = (0.0, 0, 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                for x, y in dl_va:
                    x, y = (x.to(dev), y.to(dev))
                    lg = upsampled_logits(model, x).float()
                    vl += crit(lg, y).item() * len(x)
                    m = y != IGNORE
                    agree += (lg.argmax(1)[m] == y[m]).sum().item()
                    n += m.sum().item()
            rec.update(val_loss=vl / len(dl_va.dataset), val_rule_agreement=agree / max(n, 1))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec.get("val_loss", -ep)
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean,
                    "sd": sd,
                    "q": a.q,
                    "fold": a.fold,
                    "arch": a.arch,
                    "epoch": ep + 1,
                    "n_train": len(f_tr),
                },
                f"{a.out}/{name}.pt",
            )
    with open(f"{a.out}/log_{name}.json", "w") as fh:
        json.dump(log, fh, indent=1)


if __name__ == "__main__":
    main()
