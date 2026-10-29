from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rasterio
import rasterio.errors
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from siteconfig import BOXES, CRS, JRC_TILES, OFFSET_BOX

HREF = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/{z}/{b}/{sq}/{y}/{m}/{id}/{f}.tif"
JRC = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{}v1_4_2021.tif"
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
NATIVE = {
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B08": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B8A": 20,
    "B11": 20,
    "B12": 20,
    "SCL": 20,
}
TAU = {"t1": 0.2, "t2": 0.0, "t3": 0.05, "t4": -0.15, "t5": 0.3, "t6": 0.4}
CLASSES = (
    "Deep open water",
    "Shallow/turbid water",
    "Wet shoreline mud",
    "Evaporite crust",
    "Dry lake-bed sediment",
    "Emergent vegetation",
)
COLOURS = ("#08306b", "#3fa9f5", "#8c510a", "#d01c8b", "#e6b450", "#1a9641")
IGNORE = 255
os.environ.update(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="1",
)


def grid(box, res):
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = tr.transform([box[0], box[2], box[2], box[0]], [box[1], box[1], box[3], box[3]])
    x0, y1 = (np.floor(min(xs) / res) * res, np.ceil(max(ys) / res) * res)
    w, h = (int(np.ceil((max(xs) - x0) / res)), int(np.ceil((y1 - min(ys)) / res)))
    return (rasterio.transform.from_origin(x0, y1, res, res), w, h)


def warp(href, tf, w, h, resampling, nodata=0, **opts):
    for attempt in range(9):
        try:
            with rasterio.open(href, **opts) as src, WarpedVRT(
                src, crs=CRS, transform=tf, width=w, height=h, resampling=resampling, nodata=nodata
            ) as vrt:
                return vrt.read(1)
        except (rasterio.errors.RasterioIOError, rasterio.errors.WarpOperationError):
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(href)


def href(sid: str, band: str) -> str:
    tile, day = sid.split("_")[1:3]
    return HREF.format(z=tile[:2], b=tile[2], sq=tile[3:], y=day[:4], m=int(day[4:6]), id=sid, f=band)


def read_scene(sid: str, tf, w: int, h: int, res: float, bands=BANDS) -> tuple[np.ndarray, np.ndarray]:

    def one(b):
        level = int(np.log2(res / NATIVE[b])) - 1
        opts = {"overview_level": level} if level >= 0 else {}
        rs = Resampling.nearest if b == "SCL" else Resampling.average if res > NATIVE[b] else Resampling.bilinear
        return warp(href(sid, b), tf, w, h, rs, **opts)

    with ThreadPoolExecutor(len(bands) + 1) as ex:
        arrs = list(ex.map(one, (*bands, "SCL")))
    scl = arrs.pop()
    stf, sw, sh = grid(OFFSET_BOX, 80.0)
    b12 = warp(href(sid, "B12"), stf, sw, sh, Resampling.average, overview_level=1)
    v = b12[b12 > 0]
    off = 1000 if v.size and np.percentile(v, 1) > 800 else 0
    x = np.stack([np.where(a > off, (a.astype("float32") - off) / 10000.0, 0) for a in arrs])
    return (x, scl)


def occurrence(tf, w, h) -> np.ndarray:
    occ = np.zeros((h, w), "uint8")
    for t in JRC_TILES:
        a = warp(JRC.format(t), tf, w, h, Resampling.nearest, nodata=255)
        occ = np.maximum(occ, np.where(a <= 100, a, 0).astype("uint8"))
    return occ


def rule_labels(x: np.ndarray, scl: np.ndarray, occ: np.ndarray, bands=BANDS) -> np.ndarray:
    g = dict(zip(bands, x, strict=True))
    blue, green, red, nir, sw1, sw2 = (g["B02"], g["B03"], g["B04"], g["B08"], g["B11"], g["B12"])
    t = TAU
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - sw1) / (green + sw1)
        ndvi = (nir - red) / (nir + red)
    awei = blue + 2.5 * green - 1.5 * (nir + sw1) - 0.25 * sw2
    beta = np.sqrt((green**2 + red**2 + nir**2 + sw1**2) / 4)
    nveg, bed = (ndvi <= t["t6"], occ > 0)
    rules = np.array(
        [
            (mndwi > t["t1"]) & (awei > t["t2"]) & (occ >= 90),
            (mndwi > t["t1"]) & ((red > t["t3"]) | (occ < 90)),
            (mndwi > t["t4"]) & (mndwi <= t["t1"]) & (beta < t["t5"]) & nveg & bed,
            (beta >= t["t5"]) & (mndwi <= t["t1"]) & nveg & bed,
            (mndwi <= t["t4"]) & (beta < t["t5"]) & nveg & bed,
            (ndvi > t["t6"]) & bed,
        ]
    )
    lab = np.where(rules.sum(axis=0) == 1, rules.argmax(axis=0) + 1, 0)
    bad = np.isin(scl, (0, 1, 3, 8, 9, 10, 11)) | (green > 0.4) | np.any(x <= 0, axis=0)
    return np.where(bad, 0, lab).astype("uint8")


def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_scene(sid: str, lake: str, cache: str | None = None) -> tuple[np.ndarray, np.ndarray, object]:
    tf, w, h = grid(BOXES[lake], 10.0)
    fn = os.path.join(cache, f"{sid}_{lake}.npz") if cache else None
    if fn and os.path.exists(fn):
        with np.load(fn) as z:
            return (z["x"].astype("float32"), z["scl"], tf)
    x, scl = read_scene(sid, tf, w, h, 10.0)
    if fn:
        os.makedirs(cache, exist_ok=True)
        np.savez_compressed(fn, x=x.astype("float16"), scl=scl)
    return (x, scl, tf)


def masked(x: np.ndarray, scl: np.ndarray) -> np.ndarray:
    return np.isin(scl, (0, 1, 3, 8, 9, 10, 11)) | (x[1] > 0.4) | np.any(x <= 0, axis=0)


def features(x: np.ndarray) -> np.ndarray:
    g = dict(zip(BANDS, x, strict=True))
    blue, green, red, nir, sw1, sw2 = (g["B02"], g["B03"], g["B04"], g["B08"], g["B11"], g["B12"])
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = np.nan_to_num((green - sw1) / (green + sw1))
        ndvi = np.nan_to_num((nir - red) / (nir + red))
    awei = blue + 2.5 * green - 1.5 * (nir + sw1) - 0.25 * sw2
    beta = np.sqrt((green**2 + red**2 + nir**2 + sw1**2) / 4)
    return np.concatenate([x, np.stack([mndwi, ndvi, awei, beta])]).astype("float32")
