from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

K = 6
CLASSES = (
    "Deep open water",
    "Shallow/turbid water",
    "Wet shoreline mud",
    "Evaporite crust",
    "Dry lake-bed sediment",
    "Emergent vegetation",
)
MAIN = ("segformer", "unet", "rf")


def confusion(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    c = np.zeros((K, K + 1), int)
    for a, b in zip(t, p, strict=True):
        c[a - 1, b - 1 if 1 <= b <= K else K] += 1
    return c


def scores(t: np.ndarray, p: np.ndarray) -> dict:
    c = confusion(t, p)
    tp = np.diag(c[:, :K]).astype(float)
    fn = c.sum(1) - tp
    fp = c[:, :K].sum(0) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    present = c.sum(1) > 0
    return {
        "n": len(t),
        "oa": float(tp.sum() / max(len(t), 1)),
        "miou": float(np.nanmean(iou[present])),
        "macro_f1": float(np.nanmean(f1[present])),
        "iou": [None if np.isnan(v) else float(v) for v in iou],
        "f1": [None if np.isnan(v) else float(v) for v in f1],
        "confusion": c.tolist(),
        "coverage": float(np.mean((p >= 1) & (p <= K))),
    }


def mcnemar(ok_a: np.ndarray, ok_b: np.ndarray) -> dict:
    b, c = (int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b)))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only": b, "b_only": c, "p": float(p)}


def evaluate(ref: pd.DataFrame, pred: pd.DataFrame, demo: bool = False) -> dict:
    ref = ref[ref["label_code"].between(1, K)].copy()
    wide = pred.pivot_table(index="point_id", columns="run", values="pred", aggfunc="first")
    wide["rules"] = ref.set_index("point_id")["rule_label"]
    fold = pred.drop_duplicates("point_id").set_index("point_id")["fold"]
    df = ref.set_index("point_id").join(wide, how="inner").join(fold)
    t = df["label_code"].to_numpy(int)
    runs = [r for r in wide.columns if r in df]
    res = {
        "demo": demo,
        "classes": CLASSES,
        "n_points": len(df),
        "n_per_class": np.bincount(t - 1, minlength=K).tolist(),
        "runs": {},
    }
    year = df["year"].astype(int).to_numpy()
    for r in runs:
        p = df[r].fillna(0).to_numpy(int)
        s = scores(t, p)
        s["y2025"] = {k: v for k, v in scores(t[year == 2025], p[year == 2025]).items() if k != "confusion"}
        s["y2019_2023"] = {k: v for k, v in scores(t[year < 2025], p[year < 2025]).items() if k != "confusion"}
        folds = {}
        for k in sorted(df["fold"].dropna().unique()):
            m = (df["fold"] == k).to_numpy()
            fs = scores(t[m], p[m])
            folds[str(int(k))] = {"f1": fs["f1"], "miou": fs["miou"], "n": fs["n"]}
        s["folds"] = folds
        if r != "segformer" and "segformer" in df:
            s["mcnemar_vs_segformer"] = mcnemar(df["segformer"].fillna(0).to_numpy(int) == t, p == t)
        res["runs"][r] = s
    qs = {}
    for r in runs:
        if r == "segformer":
            qs["0.7"] = r
        elif r.startswith("segformer_q"):
            qs[r.split("_q")[1]] = r
    qfolds = sorted(
        {int(k) for r in qs.values() if r != "segformer" for k in df.loc[df[r].fillna(0) > 0, "fold"].dropna().unique()}
    )
    mq = df["fold"].isin(qfolds).to_numpy() if qfolds else np.ones(len(df), bool)
    res["q_sensitivity"] = {}
    for q, r in sorted(qs.items(), key=lambda kv: float(kv[0])):
        p = df[r].fillna(0).to_numpy(int)
        fm = [
            scores(t[mq & (df["fold"] == k).to_numpy()], p[mq & (df["fold"] == k).to_numpy()])["miou"]
            for k in qfolds or sorted(df["fold"].dropna().unique())
        ]
        res["q_sensitivity"][q] = {
            "run": r,
            "miou": scores(t[mq], p[mq])["miou"],
            "fold_miou": fm,
            "folds": qfolds,
            "n": int(mq.sum()),
        }
    return res


def demo_inputs(meta: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    m = pd.read_csv(meta)
    near = {1: [2], 2: [1, 3], 3: [2, 4, 5], 4: [3, 5], 5: [4, 3], 6: [3, 5]}
    truth = np.where(m["rule_label"] > 0, m["rule_label"], rng.integers(1, K + 1, len(m)))
    flip = rng.random(len(m)) < 0.15
    truth[flip] = [rng.choice(near[v]) for v in truth[flip]]
    code = truth.copy()
    out = rng.random(len(m)) < 0.08
    code[out] = rng.choice([7, 8], out.sum())
    ref = m.assign(label_code=code, year=m["date"].str[:4].astype(int))
    blocks = {c: i % 5 for i, c in enumerate(rng.permutation(m["chip"].unique()))}
    acc = {
        "segformer": 0.86,
        "unet": 0.8,
        "rf": 0.71,
        "segformer_q0.3": 0.8,
        "segformer_q0.5": 0.83,
        "segformer_q0.9": 0.82,
    }
    rows = []
    for run, a in acc.items():
        for i, p in m.iterrows():
            t = int(truth[i])
            a2 = a - (0.06 if p["date"].startswith("2025") else 0) + rng.normal(0, 0.03)
            v = t if rng.random() < a2 else int(rng.choice(near[t]))
            rows.append({"point_id": p["point_id"], "fold": blocks[p["chip"]], "run": run, "pred": v})
    return (ref, pd.DataFrame(rows))


def main() -> None:
    if sys.argv[1] == "--demo":
        ref, pred = demo_inputs(sys.argv[2])
        out, demo = ("accuracy_results_DEMO.json", True)
    else:
        lab = pd.read_csv(sys.argv[1])
        meta = pd.read_csv(sys.argv[2])
        ref = meta.merge(lab[["point_id", "label_code"]], on="point_id", how="left")
        ref["label_code"] = ref["label_code"].fillna(0).astype(int)
        ref["year"] = ref["date"].str[:4].astype(int)
        pred = pd.read_csv(sys.argv[3])
        out, demo = (sys.argv[4] if len(sys.argv) > 4 else "accuracy_results.json", False)
    res = evaluate(ref, pred, demo)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for r, s in res["runs"].items():
        print(
            f"{r:16s} OA {s['oa']:.3f}  mIoU {s['miou']:.3f}  macro F1 {s['macro_f1']:.3f}  2025 mIoU {s['y2025']['miou']:.3f}  McNemar p {s.get('mcnemar_vs_segformer', {}).get('p', '--')}"
        )


if __name__ == "__main__":
    main()
