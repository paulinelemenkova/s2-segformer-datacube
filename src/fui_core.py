from __future__ import annotations
import numpy as np

WX = (8.356, 12.04, 53.696, 32.028, 0.529, 0.016)
WY = (0.993, 23.122, 65.702, 16.808, 0.192, 0.006)
WZ = (43.487, 61.055, 1.778, 0.015, 0.0, 0.0)
POLY = (-161.23, 1117.08, -2950.14, 3612.17, -1943.57, 364.28)
FU_HUE = np.array(
    [
        229.45,
        224.79,
        217.12,
        203.09,
        178.91,
        147.64,
        118.289,
        99.75,
        88.37,
        78.25,
        71.08,
        65.06,
        59.56,
        53.64,
        47.89,
        42.18,
        37.23,
        32.63,
        28.38,
        24.3,
        20.98,
    ]
)
FU_RGB = (
    (33, 88, 188),
    (49, 109, 197),
    (50, 124, 187),
    (75, 128, 160),
    (86, 143, 150),
    (109, 146, 152),
    (105, 140, 134),
    (117, 158, 114),
    (123, 166, 84),
    (125, 174, 56),
    (149, 182, 69),
    (148, 182, 96),
    (165, 188, 118),
    (170, 184, 109),
    (173, 181, 95),
    (168, 169, 101),
    (174, 159, 92),
    (179, 160, 83),
    (175, 138, 68),
    (164, 105, 5),
    (161, 77, 4),
)


def hue_angle(b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, b5: np.ndarray) -> np.ndarray:
    bands = (b2, b2, b3, b4, b5, b5)
    x_, y_, z_ = (sum((w * b for w, b in zip(ws, bands, strict=True))) for ws in (WX, WY, WZ))
    tot = x_ + y_ + z_
    with np.errstate(invalid="ignore", divide="ignore"):
        x, y = (x_ / tot, y_ / tot)
        alpha = np.degrees(np.arctan2(y - 1 / 3, x - 1 / 3)) % 360
    a = alpha / 100
    corr = np.polyval(POLY, a)
    return np.where(tot > 0, alpha + corr, np.nan)


def fu_class(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype="float64")
    idx = np.abs(a[..., None] - FU_HUE).argmin(axis=-1) + 1
    return np.where(np.isfinite(a), idx, 0).astype("uint8")
