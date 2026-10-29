from __future__ import annotations
import csv
import json
import os
import sys
import zlib
import numpy as np
from common import BOXES, IGNORE, grid, occurrence, read_scene, rule_labels

SIZE, RES, BLOCK, FOLDS = (256, 10.0, 5000.0, 5)


def fold_table(tf_by_lake: dict) -> dict:
    rng = np.random.default_rng(42)
    table = {}
    for lake, (tf, w, h) in tf_by_lake.items():
        bx = np.arange(int(np.ceil(w * RES / BLOCK)))
        by = np.arange(int(np.ceil(h * RES / BLOCK)))
        blocks = [(lake, int(i), int(j)) for j in by for i in bx]
        for k, b in enumerate(rng.permutation(len(blocks))):
            table[blocks[b]] = k % FOLDS
    return table


def main(scenes_file: str = "fui_scenes_top1.json", out: str = ".", max_per_box: str = "12") -> None:
    cap = int(max_per_box)
    with open(scenes_file) as fh:
        scenes = json.load(fh)
    os.makedirs(f"{out}/chips", exist_ok=True)
    os.makedirs(f"{out}/rows", exist_ok=True)
    grids = {lake: grid(box, RES) for lake, box in BOXES.items()}
    occs = {lake: occurrence(*g) for lake, g in grids.items()}
    folds = fold_table(grids)
    fields = ["chip", "scene", "lake", "year", "row", "col", "block", "fold", "labelled"]
    for i, sid in enumerate(scenes):
        done = f"{out}/rows/{sid}.csv"
        if os.path.exists(done):
            continue
        year = int(sid.split("_")[2][:4])
        rows = []
        for lake, (tf, w, h) in grids.items():
            x, scl = read_scene(sid, tf, w, h, RES)
            lab = rule_labels(x, scl, occs[lake])
            y = np.where(lab > 0, lab - 1, IGNORE).astype("uint8")
            cand = []
            for r in range(0, h - SIZE + 1, SIZE):
                for c in range(0, w - SIZE + 1, SIZE):
                    yy = y[r : r + SIZE, c : c + SIZE]
                    if (yy != IGNORE).mean() >= 0.5 and np.isin(yy, (0, 1)).any():
                        cand.append((r, c))
            rng = np.random.default_rng(zlib.crc32(f"{sid}_{lake}".encode()))
            if len(cand) > cap:
                cand = [cand[j] for j in sorted(rng.choice(len(cand), cap, replace=False))]
            for r, c in cand:
                yy = y[r : r + SIZE, c : c + SIZE]
                name = f"{sid}_{lake}_{r}_{c}"
                np.savez_compressed(
                    f"{out}/chips/{name}.npz", x=x[:, r : r + SIZE, c : c + SIZE].astype("float16"), y=yy
                )
                block = (lake, int((c + SIZE / 2) * RES // BLOCK), int((r + SIZE / 2) * RES // BLOCK))
                rows.append(
                    {
                        "chip": name,
                        "scene": sid,
                        "lake": lake,
                        "year": year,
                        "row": r,
                        "col": c,
                        "block": f"{block[0]}_{block[1]}_{block[2]}",
                        "fold": folds[block],
                        "labelled": round(float((yy != IGNORE).mean()), 3),
                    }
                )
        with open(done + ".tmp", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        os.replace(done + ".tmp", done)
        print(f"[{i + 1}/{len(scenes)}] {sid}: {len(rows)} chips", flush=True)
    allrows = []
    for sid in scenes:
        with open(f"{out}/rows/{sid}.csv") as fh:
            allrows += list(csv.DictReader(fh))
    with open(f"{out}/chips.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(allrows)
    print(f"{len(allrows)} chips -> {out}/chips.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
