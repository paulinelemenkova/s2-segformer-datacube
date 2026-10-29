from __future__ import annotations
import json
import os
import shutil
import sys
import pandas as pd

FOLDS = 5


def main(work: str) -> None:
    src = os.path.join(work, "chips_original.csv")
    if not os.path.exists(src):
        shutil.copy(os.path.join(work, "chips.csv"), src)
    df = pd.read_csv(src)
    per_fold = {k: 0 for k in range(FOLDS)}
    table: dict[str, int] = {}
    for lake, g in df.groupby("lake"):
        counts = g["block"].value_counts().sort_index().sort_values(ascending=False, kind="stable")
        lake_fold = {k: 0 for k in range(FOLDS)}
        for block, n in counts.items():
            k = min(range(FOLDS), key=lambda f: (lake_fold[f], per_fold[f], f))
            table[block] = k
            lake_fold[k] += int(n)
            per_fold[k] += int(n)
    df["fold"] = df["block"].map(table)
    df.to_csv(os.path.join(work, "chips.csv"), index=False)
    with open(os.path.join(work, "block_folds.json"), "w") as fh:
        json.dump(table, fh, indent=1)
    print(df.groupby(["lake", "fold"]).size().unstack(fill_value=0))
    print("blocks per lake:", df.groupby("lake")["block"].nunique().to_dict())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work")
