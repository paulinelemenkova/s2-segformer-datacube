from __future__ import annotations
import csv
import glob
import json
import os
import re
import sys
import torch
from pyproj import Transformer
from common import BOXES, CRS, device, grid, load_scene, masked
from siteconfig import REF_SCENES
from make_chips import BLOCK, RES, SIZE, fold_table
from train_segformer import build, upsampled_logits

SCENES = {s.split("_")[2]: s for s in REF_SCENES}


def main(work: str, meta: str) -> None:
    dev = device()
    with open(os.path.join(work, "block_folds.json")) as fh:
        bf = json.load(fh)
    with open(meta) as fh:
        pts = list(csv.DictReader(fh))
    fallback = fold_table({lake: grid(box, RES) for lake, box in BOXES.items()})
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    ckpts = sorted(glob.glob(os.path.join(work, "runs", "*_fold[0-9].pt")))
    if not ckpts:
        sys.exit("no fold checkpoints in WORK/runs")
    by_scene: dict[tuple[str, str], list[dict]] = {}
    for p in pts:
        by_scene.setdefault((p["date"].replace("-", ""), p["lake"]), []).append(p)
    rows = []
    for ck in ckpts:
        name = os.path.basename(ck)[:-3]
        run, k = re.match("(.+)_fold(\\d)$", name).groups()
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model = build(state.get("arch", "segformer"), pretrained=False)
        model.load_state_dict(state["model"])
        model.to(dev).eval()
        mean, sd = (state["mean"][:, None, None], state["sd"][:, None, None])
        n = 0
        for (day, lake), group in by_scene.items():
            x, scl, tf = (None, None, None)
            for p in group:
                ex, ny = tr.transform(float(p["lon"]), float(p["lat"]))
                if tf is None:
                    x, scl, tf = load_scene(SCENES[day], lake, os.path.join(work, "cache"))
                c, r = ((ex - tf.c) / RES, (tf.f - ny) / RES)
                r0, c0 = (int(r // SIZE) * SIZE, int(c // SIZE) * SIZE)
                block = f"{lake}_{int((c0 + SIZE / 2) * RES // BLOCK)}_{int((r0 + SIZE / 2) * RES // BLOCK)}"
                kb = bf.get(
                    block,
                    fallback.get((lake, int((c0 + SIZE / 2) * RES // BLOCK), int((r0 + SIZE / 2) * RES // BLOCK))),
                )
                if str(kb) != k:
                    continue
                ri, ci = (int(r), int(c))
                h, w = scl.shape
                rs = min(max(ri - SIZE // 2, 0), h - SIZE)
                cs = min(max(ci - SIZE // 2, 0), w - SIZE)
                win = (x[:, rs : rs + SIZE, cs : cs + SIZE] - mean) / sd
                with torch.no_grad():
                    lg = upsampled_logits(model, torch.from_numpy(win[None].astype("float32")).to(dev))
                pred = int(lg[0, :, ri - rs, ci - cs].argmax()) + 1
                msk = bool(masked(x[:, ri : ri + 1, ci : ci + 1], scl[ri : ri + 1, ci : ci + 1])[0, 0])
                rows.append(
                    {
                        "point_id": p["point_id"],
                        "lake": lake,
                        "date": p["date"],
                        "year": p["date"][:4],
                        "fold": k,
                        "run": run,
                        "pred": pred,
                        "masked": int(msk),
                    }
                )
                n += 1
        print(f"{name}: {n} points", flush=True)
    out = os.path.join(work, "ref_predictions_nomask.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
