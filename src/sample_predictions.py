from __future__ import annotations
import csv
import json
import os
import sys
import rasterio
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segformer"))
from common import BOXES, CRS, grid
from make_chips import BLOCK, RES, SIZE, fold_table


def main(meta: str, root: str, out: str = "ref_predictions.csv", block_folds: str = "") -> None:
    grids = {lake: grid(box, RES) for lake, box in BOXES.items()}
    folds = fold_table(grids)
    balanced = {}
    if block_folds and os.path.exists(block_folds):
        with open(block_folds) as fh:
            balanced = json.load(fh)
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    with open(meta) as fh:
        pts = list(csv.DictReader(fh))
    runs = sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))))
    rows, missing = ([], 0)
    for p in pts:
        lake = p["lake"]
        tf, _, _ = grids[lake]
        x, y = tr.transform(float(p["lon"]), float(p["lat"]))
        c, r = ((x - tf.c) / RES, (tf.f - y) / RES)
        r0, c0 = (int(r // SIZE) * SIZE, int(c // SIZE) * SIZE)
        block = (lake, int((c0 + SIZE / 2) * RES // BLOCK), int((r0 + SIZE / 2) * RES // BLOCK))
        k = balanced.get(f"{block[0]}_{block[1]}_{block[2]}", folds.get(block))
        day = p["date"].replace("-", "")
        for run in runs:
            fn = os.path.join(root, run, f"fold{k}", f"{lake}_{day}_class.tif")
            v = 0
            if k is not None and os.path.exists(fn):
                with rasterio.open(fn) as src:
                    row, col = src.index(x, y)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        v = int(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
            missing += v == 0
            rows.append(
                {
                    "point_id": p["point_id"],
                    "lake": lake,
                    "date": p["date"],
                    "year": p["date"][:4],
                    "block": f"{block[0]}_{block[1]}_{block[2]}",
                    "fold": "" if k is None else k,
                    "run": run,
                    "pred": v,
                }
            )
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(pts)} points x {len(runs)} runs -> {out}; {missing} masked or missing values")


if __name__ == "__main__":
    main(*sys.argv[1:])
