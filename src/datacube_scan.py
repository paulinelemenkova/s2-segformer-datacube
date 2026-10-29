from __future__ import annotations
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rasterio
import rasterio.errors
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject
from siteconfig import CRS, TILES, WINDOW

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
LONLAT = WINDOW
RES, OVERVIEW = (320.0, 3)
VALID = np.array([2, 4, 5, 6, 7, 11])
CLOUD = np.array([8, 9, 10])
os.environ.update(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_TIMEOUT="30",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="1",
)


def grid() -> tuple[tuple[float, float, float, float], int, int]:
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = tr.transform([LONLAT[0], LONLAT[2], LONLAT[2], LONLAT[0]], [LONLAT[1], LONLAT[1], LONLAT[3], LONLAT[3]])
    b = (
        np.floor(min(xs) / RES) * RES,
        np.floor(min(ys) / RES) * RES,
        np.ceil(max(xs) / RES) * RES,
        np.ceil(max(ys) / RES) * RES,
    )
    return (b, round((b[2] - b[0]) / RES), round((b[3] - b[1]) / RES))


def list_prefix(prefix: str) -> list[str]:
    for attempt in range(8):
        try:
            url = f"{BUCKET}/?list-type=2&delimiter=/&prefix={prefix}"
            xml = urllib.request.urlopen(url, timeout=30).read().decode()
            return re.findall("<Prefix>([^<]+)</Prefix>", xml)[1:]
        except OSError:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"listing failed: {prefix}")


def scenes(years: list[int]) -> list[str]:
    prefixes = [
        f"sentinel-s2-l2a-cogs/{z}/{lat}/{sq}/{y}/{m}/" for z, lat, sq in TILES for y in years for m in range(1, 13)
    ]
    with ThreadPoolExecutor(6) as ex:
        found = [p for chunk in ex.map(list_prefix, prefixes) for p in chunk]
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for p in found:
        _sat, tile, day, ver, _ = p.rstrip("/").split("/")[-1].split("_")
        key = (tile, day)
        if key not in best or int(ver) > best[key][0]:
            best[key] = (int(ver), p)
    return sorted((v[1] for v in best.values()))


def read_scene(prefix: str, b, w: int, h: int) -> tuple[np.ndarray, float] | None:
    for attempt in range(5):
        try:
            with rasterio.open(f"{BUCKET}/{prefix}SCL.tif", overview_level=OVERVIEW) as src:
                scl = src.read(1)
                obs = scl > 0
                cloud = float(np.isin(scl, CLOUD)[obs].mean() * 100) if obs.any() else np.nan
                dst = np.zeros((h, w), "uint8")
                reproject(
                    scl,
                    dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=rasterio.transform.from_origin(b[0], b[3], RES, RES),
                    dst_crs=CRS,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                return (dst, cloud)
        except rasterio.errors.RasterioIOError as err:
            if "404" in str(err):
                return None
            time.sleep(2 + 2 * attempt)
    return None


def main(years: list[int]) -> str:
    t0 = time.time()
    b, w, h = grid()
    pre = scenes(years)
    print("listed", len(pre), round(time.time() - t0), flush=True)
    with ThreadPoolExecutor(int(os.environ.get("THREADS", "16"))) as ex:
        res = list(ex.map(lambda p: read_scene(p, b, w, h), pre))
    ok = [(p, r) for p, r in zip(pre, res, strict=True) if r is not None]
    print(f"{years}: {len(pre)} scenes, {len(pre) - len(ok)} without SCL, {time.time() - t0:.0f} s")
    days = sorted({p.rstrip("/").split("_")[-3] for p, _ in ok})
    count = np.zeros((h, w), "uint16")
    water = np.zeros((h, w), "uint16")
    d_valid, d_tiles = ([], [])
    for day in days:
        sub = [r[0] for p, r in ok if p.rstrip("/").split("_")[-3] == day]
        obs = np.zeros((h, w), bool)
        val = np.zeros((h, w), bool)
        wat = np.zeros((h, w), bool)
        for s in sub:
            obs |= s > 0
            val |= np.isin(s, VALID)
            wat |= s == 6
        count += val
        water += wat & val
        d_valid.append(val.sum() / max(obs.sum(), 1))
        d_tiles.append(len(sub))
    out = f"scan_{years[0]}_{years[-1]}.npz"
    np.savez_compressed(
        out,
        count=count,
        water=water,
        bounds=np.array(b),
        days=np.array(days),
        day_valid=np.array(d_valid, "float32"),
        day_tiles=np.array(d_tiles, "uint8"),
        scene=np.array([p.rstrip("/").split("/")[-1] for p, _ in ok]),
        scene_cloud=np.array([r[1] for _, r in ok], "float32"),
        listed=np.array([p.rstrip("/").split("/")[-1] for p in pre]),
    )
    return out


if __name__ == "__main__":
    print(main([int(y) for y in sys.argv[1:]]))
