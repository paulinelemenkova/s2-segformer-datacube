from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from common import CRS, features, load_scene, masked
from predict_segformer import save

IGNORE = 255


def training_pixels(files: list[str], per_chip: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = ([], [])
    for f in files:
        with np.load(f) as z:
            x, y = (z["x"].astype("float32"), z["y"])
        idx = np.flatnonzero(y.ravel() != IGNORE)
        if idx.size == 0:
            continue
        idx = rng.choice(idx, min(per_chip, idx.size), replace=False)
        xs.append(features(x).reshape(14, -1)[:, idx].T)
        ys.append(y.ravel()[idx] + 1)
    return (np.concatenate(xs), np.concatenate(ys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="chips.csv")
    ap.add_argument("--chip_dir", default="chips")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--max_chips", type=int, default=3000)
    ap.add_argument("--per_chip", type=int, default=150)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_mask", action="store_true", help="keep the class of cloud/bright pixels")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.chips)
    df = df[(df.year != 2025) & (df.fold != a.fold)]
    if len(df) > a.max_chips:
        df = df.sample(a.max_chips, random_state=42)
    x, y = training_pixels([f"{a.chip_dir}/{c}.npz" for c in df.chip], a.per_chip)
    print(f"rf fold {a.fold}: {len(y)} training pixels from {len(df)} chips", flush=True)
    rf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=42).fit(x, y)
    pts = pd.read_csv(a.points)
    pts["day"] = pts["date"].str.replace("-", "")
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    for sid in a.scenes:
        day = sid.split("_")[2]
        for lake, g in pts[pts.day == day].groupby("lake"):
            xs, scl, tf = load_scene(sid, lake, a.cache)
            out = np.zeros(scl.shape, "uint8")
            ex, ny = tr.transform(g.lon.to_numpy(), g.lat.to_numpy())
            rr, cc = rasterio.transform.rowcol(tf, ex, ny)
            rr, cc = (np.asarray(rr), np.asarray(cc))
            ok = (rr >= 0) & (rr < out.shape[0]) & (cc >= 0) & (cc < out.shape[1])
            rr, cc = (rr[ok], cc[ok])
            f = features(xs[:, rr, cc]).T
            pred = rf.predict(f).astype("uint8")
            bad = masked(xs[:, rr, cc][:, :, None], scl[rr, cc][:, None])[:, 0]
            out[rr, cc] = pred if a.no_mask else np.where(bad, 0, pred)
            save(f"{a.out}/{lake}_{day}_class.tif", out, tf, 0)
            print(f"rf fold {a.fold}", sid, lake, int(ok.sum()), "points", flush=True)


if __name__ == "__main__":
    main()
