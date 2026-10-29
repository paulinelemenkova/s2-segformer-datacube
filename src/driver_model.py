from __future__ import annotations
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xarray as xr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from siteconfig import BOXES

LAKES = {k: str(i + 1) for i, k in enumerate(BOXES)}
FEATURES = ["P_1", "P_3", "P_6", "P_12", "E_3", "E_12", "T_3", "T_12", "PmE_12", "CumPmE", "NDVI_anom", "NDVI_summer"]
PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=7,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


def era5_by_basin() -> dict[str, pd.DataFrame]:
    ds = xr.open_dataset(os.environ.get("ERA5", "era5land_monthly.nc"))
    t = pd.DatetimeIndex(ds["valid_time"].values)
    days = xr.DataArray(t.days_in_month.values, dims="valid_time")
    p = (ds["tp"] * 1000.0 * days).values
    e = (-ds["e"] * 1000.0 * days).values
    tt = (ds["t2m"] - 273.15).values
    w = pd.read_csv(os.environ.get("WEIGHTS", "basin_weights.csv"))
    lat, lon = (np.round(ds["latitude"].values, 1), np.round(ds["longitude"].values, 1))
    out = {}
    for b, g in w.groupby("basin"):
        iy = [int(np.argmin(np.abs(lat - v))) for v in g["lat"]]
        ix = [int(np.argmin(np.abs(lon - v))) for v in g["lon"]]
        wt = g["npix"].to_numpy(float) / g["npix"].sum()
        out[str(b)] = pd.DataFrame(
            {k: (a[:, iy, ix] * wt).sum(1) for k, a in (("P", p), ("E", e), ("T", tt))}, index=t.to_period("M")
        )
    return out


def climate_features(c: pd.DataFrame, month: pd.Period) -> dict:
    pme = c["P"] - c["E"]
    cum = (pme - pme.groupby(pme.index.month).transform("mean")).cumsum()
    hist = c.loc[: month - 1]
    if len(hist) < 12:
        return {}
    f = {f"P_{k}": hist["P"].iloc[-k:].sum() for k in (1, 3, 6, 12)}
    f.update({f"E_{k}": hist["E"].iloc[-k:].sum() for k in (3, 12)})
    f.update({f"T_{k}": hist["T"].iloc[-k:].mean() for k in (3, 12)})
    f["PmE_12"] = f["P_12"] - f["E_12"]
    f["CumPmE"] = cum.loc[: month - 1].iloc[-1]
    return f


def ndvi_table() -> pd.DataFrame:
    with open(os.environ.get("NDVI", "ndvi_monthly.json")) as fh:
        d = json.load(fh)
    rows = []
    for sid, v in d.items():
        day = pd.Timestamp(sid.split("_")[2])
        for b, (ndvi, frac) in v.items():
            rows.append({"basin": b, "date": day, "ndvi": ndvi if ndvi is not None and frac >= 0.5 else np.nan})
    df = pd.DataFrame(rows)
    df["month"] = df["date"].dt.month
    df["anom"] = df["ndvi"] - df.groupby(["basin", "month"])["ndvi"].transform("median")
    return df


def build() -> pd.DataFrame:
    with open(os.environ.get("AREA", "area_stats.json")) as fh:
        st = json.load(fh)
    era = era5_by_basin()
    nd = ndvi_table()
    rows = []
    for lake, b in LAKES.items():
        r = pd.DataFrame([x for x in st[lake]["rows"] if not x["flag"]])
        r["date"] = pd.to_datetime(r["date"])
        r["water"] = [a[0] + a[1] for a in r["areas"]]
        r["month"] = r["date"].dt.month
        anom = r["water"] - r.groupby("month")["water"].transform("median")
        r["y"] = anom / anom.std()
        r["y_km2"] = anom
        ndb = nd[nd["basin"] == b].set_index("date")
        for _, x in r.iterrows():
            f = climate_features(era[b], x["date"].to_period("M"))
            if not f:
                continue
            f["NDVI_anom"] = ndb["anom"].get(x["date"], np.nan)
            ys = x["date"].year if x["date"].month >= 9 else x["date"].year - 1
            sm = ndb[(ndb.index.year == ys) & ndb.index.month.isin([6, 7, 8])]["ndvi"]
            f["NDVI_summer"] = sm.mean() if sm.notna().any() else np.nan
            rows.append(
                {
                    "lake": lake,
                    "date": x["date"].strftime("%Y-%m-%d"),
                    "year": x["date"].year,
                    "y": x["y"],
                    "y_km2": x["y_km2"],
                    **f,
                }
            )
    return pd.DataFrame(rows)


def main() -> dict:
    df = build()
    X, y = (df[FEATURES], df["y"].to_numpy())
    Xf = X.fillna(X.median())
    oof, oof_lin = (np.full(len(df), np.nan), np.full(len(df), np.nan))
    for yr in sorted(df["year"].unique()):
        te = df["year"] == yr
        m = lgb.LGBMRegressor(**PARAMS).fit(X[~te], y[~te])
        oof[te.to_numpy()] = m.predict(X[te])
        lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))).fit(Xf[~te], y[~te])
        oof_lin[te.to_numpy()] = lin.predict(Xf[te])
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = np.abs(sv).mean(0)
    order = [FEATURES[i] for i in np.argsort(-imp)]
    per_lake = {
        lk: {
            "n": int((df["lake"] == lk).sum()),
            "r2": round(float(r2_score(y[df["lake"] == lk], oof[df["lake"] == lk])), 3),
            "r2_ridge": round(float(r2_score(y[df["lake"] == lk], oof_lin[df["lake"] == lk])), 3),
        }
        for lk in LAKES
    }
    corr = {f: round(float(df[[f, "y"]].corr().iloc[0, 1]), 3) for f in FEATURES}
    res = {
        "n": len(df),
        "years": [int(v) for v in sorted(df["year"].unique())],
        "r2_loyo": round(float(r2_score(y, oof)), 3),
        "rmse_loyo": round(float(np.sqrt(mean_squared_error(y, oof))), 3),
        "r2_loyo_ridge": round(float(r2_score(y, oof_lin)), 3),
        "corr_with_y": corr,
        "r2_fit": round(float(r2_score(y, model.predict(X))), 3),
        "per_lake": per_lake,
        "mean_abs_shap": {f: round(float(v), 4) for f, v in zip(FEATURES, imp, strict=True)},
        "order": order,
        "expected_value": float(np.ravel(shap.TreeExplainer(model).expected_value)[0]),
        "features": FEATURES,
        "lake": df["lake"].tolist(),
        "date": df["date"].tolist(),
        "X": np.round(X.to_numpy(float), 4).tolist(),
        "shap": np.round(sv, 4).tolist(),
        "y": np.round(y, 4).tolist(),
        "oof": np.round(oof, 4).tolist(),
        "oof_ridge": np.round(oof_lin, 4).tolist(),
        "versions": {"lightgbm": lgb.__version__, "shap": shap.__version__},
    }
    with open(os.environ.get("OUT", "shap_results.json"), "w") as fh:
        json.dump(res, fh)
    return res


if __name__ == "__main__":
    r = main()
    print({k: r[k] for k in ("n", "years", "r2_loyo", "r2_loyo_ridge", "rmse_loyo", "r2_fit", "per_lake", "order")})
    print(r["corr_with_y"])
    print(r["mean_abs_shap"])
