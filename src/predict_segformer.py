from __future__ import annotations
import argparse
import os
import numpy as np
import rasterio
import torch
from common import BOXES, CRS, device, load_scene, masked
from train_segformer import build, mc_predict, upsampled_logits

SIZE, STRIDE = (256, 192)


@torch.no_grad()
def predict_box(model, x: np.ndarray, mean, sd, dev, t: int) -> tuple[np.ndarray, np.ndarray]:
    c, h, w = x.shape
    xn = (x - mean[:, None, None]) / sd[:, None, None]
    ph, pw = (max(h, SIZE), max(w, SIZE))
    xp = np.zeros((c, ph, pw), "float32")
    xp[:, :h, :w] = xn
    prob = np.zeros((6, ph, pw), "float32")
    count = np.zeros((ph, pw), "float32")
    rows = sorted({*range(0, ph - SIZE + 1, STRIDE), ph - SIZE})
    cols = sorted({*range(0, pw - SIZE + 1, STRIDE), pw - SIZE})
    amp = dev.type == "cuda"
    model.eval()
    for r in rows:
        for cc in cols:
            tile = torch.from_numpy(xp[None, :, r : r + SIZE, cc : cc + SIZE]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if t > 1:
                    p, _ = mc_predict(model, tile, t)
                else:
                    p = torch.softmax(upsampled_logits(model, tile).float(), dim=1)
            prob[:, r : r + SIZE, cc : cc + SIZE] += p[0].float().cpu().numpy()
            count[r : r + SIZE, cc : cc + SIZE] += 1
    prob = prob[:, :h, :w] / count[:h, :w]
    entropy = -(prob * np.log(np.clip(prob, 1e-08, 1))).sum(0)
    return (prob.argmax(0).astype("uint8") + 1, entropy.astype("float32"))


def save(fn, arr, tf, nodata):
    with rasterio.open(
        fn,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype=arr.dtype,
        crs=CRS,
        transform=tf,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--lakes", nargs="+", default=list(BOXES))
    ap.add_argument("--t", type=int, default=20)
    ap.add_argument("--out", default="pred")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = device()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "segformer")
    model = build(arch, pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev)
    t = a.t if arch == "segformer" else 1
    for sid in a.scenes:
        day = sid.split("_")[2]
        for lake in a.lakes:
            fc = f"{a.out}/{lake}_{day}_class.tif"
            if os.path.exists(fc) and (t == 1 or os.path.exists(f"{a.out}/{lake}_{day}_entropy.tif")):
                continue
            x, scl, tf = load_scene(sid, lake, a.cache)
            cls, ent = predict_box(model, x, ck["mean"], ck["sd"], dev, t)
            bad = masked(x, scl)
            save(fc, np.where(bad, 0, cls).astype("uint8"), tf, 0)
            if t > 1:
                save(f"{a.out}/{lake}_{day}_entropy.tif", np.where(bad, np.nan, ent).astype("float32"), tf, np.nan)
            print(os.path.basename(a.ckpt), sid, lake, flush=True)


if __name__ == "__main__":
    main()
