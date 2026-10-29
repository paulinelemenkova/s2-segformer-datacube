from __future__ import annotations
import json
import sys
from datetime import date
import numpy as np
import pymannkendall as mk
import ruptures as rpt
from siteconfig import BOXES, PERENNIAL

LAKES = tuple(BOXES)
PIX = 0.0064
MAX_MASKED = 0.1
SNOW_MONTHS = (12, 1, 2, 3)


def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / (date(d.year + 1, 1, 1) - start).days


def sen_slope(t: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    i, j = np.triu_indices(len(t), k=1)
    slope = float(np.median((a[j] - a[i]) / (t[j] - t[i])))
    intercept = float(np.median(a - slope * t))
    return (slope, intercept)


def series_stats(good: list[dict], classes: tuple[int, ...]) -> dict:
    t = np.array([r["t"] for r in good])
    a = np.array([sum((r["areas"][k] for k in classes)) for r in good])
    months = np.array([int(r["date"][5:7]) for r in good])
    test = mk.hamed_rao_modification_test(a)
    slope, icpt = sen_slope(t, a)
    clim = {m: np.median(a[months == m]) for m in set(months)}
    anom = a - np.array([clim[m] for m in months])
    z = (anom - anom.mean()) / anom.std()
    bkps = rpt.Pelt(model="l2", min_size=6).fit(z).predict(pen=2 * np.log(len(z)))
    fin = lambda x: float(x) if np.isfinite(x) else None
    return {
        "mk": {"trend": test.trend, "p": fin(test.p), "z": fin(test.z), "tau": float(test.Tau)},
        "sen_km2_per_yr": round(slope, 3),
        "sen_intercept": icpt,
        "change_points": [good[b]["date"] for b in bkps[:-1]],
        "segment_mean_km2": [round(float(a[s:e].mean()), 2) for s, e in zip([0, *bkps[:-1]], bkps, strict=True)],
        "median_km2": round(float(np.median(a)), 2),
    }


def main(src: str, dst: str) -> None:
    global PIX
    with open(src) as fh:
        counts = json.load(fh)
    PIX = counts.pop("pixel_km2", PIX)
    out = {}
    for lake in LAKES:
        rows = []
        for sid, v in sorted(counts.items(), key=lambda kv: kv[0].split("_")[2]):
            c = v[lake]
            d = sid.split("_")[2]
            day = date(int(d[:4]), int(d[4:6]), int(d[6:]))
            if c[7] / c[9] >= MAX_MASKED or (day.month in SNOW_MONTHS and c[8] / c[9] >= MAX_MASKED):
                continue
            rows.append(
                {"date": day.isoformat(), "t": round(decimal_year(day), 4), "areas": [round(x * PIX, 3) for x in c[:7]]}
            )
        water_all = np.array([r["areas"][0] + r["areas"][1] for r in rows])
        med = float(np.median(water_all))
        for r, w in zip(rows, water_all, strict=True):
            deficit = med - w
            r["flag"] = bool(lake in PERENNIAL and w < 0.8 * med and (r["areas"][2] + r["areas"][6] >= 0.5 * deficit))
        good = [r for r in rows if not r["flag"]]
        out[lake] = {
            "rows": rows,
            "n": len(rows),
            "n_flagged": len(rows) - len(good),
            "bed_km2": round(counts[next(iter(counts))][lake][9] * PIX, 2),
            "water": series_stats(good, (0, 1)),
            "crust": series_stats(good, (3,)),
        }
        w, c = (out[lake]["water"], out[lake]["crust"])
        print(
            lake,
            len(rows),
            len(rows) - len(good),
            "water",
            w["mk"]["trend"],
            f"p={w['mk']['p']}",
            f"sen={w['sen_km2_per_yr']}",
            w["change_points"],
            w["segment_mean_km2"],
            "| crust",
            c["mk"]["trend"],
            f"p={c['mk']['p']}",
            f"sen={c['sen_km2_per_yr']}",
            c["change_points"],
            c["segment_mean_km2"],
        )
    with open(dst, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
