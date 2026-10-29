from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch
from common import BOXES, device, grid, load_scene, occurrence
from predict_segformer import predict_box
from train_segformer import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes_file", required=True)
    ap.add_argument("--out", default="series_counts.json")
    ap.add_argument("--cache", default=None, help="optional scene cache (about 60 MB per scene and lake)")
    a = ap.parse_args()
    with open(a.scenes_file) as fh:
        scenes = json.load(fh)
    res = {"pixel_km2": 0.0001}
    if os.path.exists(a.out):
        with open(a.out) as fh:
            res = json.load(fh)
    dev = device()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build(ck.get("arch", "segformer"), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev)
    beds = {lake: occurrence(*grid(box, 10.0)) > 0 for lake, box in BOXES.items()}
    for i, sid in enumerate(scenes):
        if sid in res:
            continue
        res[sid] = {}
        for lake in BOXES:
            x, scl, _ = load_scene(sid, lake, a.cache)
            cls, _ = predict_box(model, x, ck["mean"], ck["sd"], dev, 1)
            bed = beds[lake]
            cloud = np.isin(scl, (0, 1, 3, 8, 9, 10)) | np.any(x <= 0, axis=0)
            bright = ((scl == 11) | (x[1] > 0.4)) & ~cloud
            ok = bed & ~cloud
            n = [int((ok & (cls == k)).sum()) for k in range(1, 7)]
            res[sid][lake] = n + [0, int((bed & cloud).sum()), int((bed & bright).sum()), int(bed.sum())]
        with open(a.out + ".tmp", "w") as fh:
            json.dump(res, fh)
        os.replace(a.out + ".tmp", a.out)
        print(f"[{i + 1}/{len(scenes)}] {sid}", flush=True)


if __name__ == "__main__":
    main()
