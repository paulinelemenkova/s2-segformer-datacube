from __future__ import annotations
import json
import os

PATH = os.environ.get("SITES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "sites.json"))
with open(PATH) as fh:
    C = json.load(fh)
CRS = C["crs"]
BOXES = {k: tuple(v) for k, v in C["sites"].items()}
OFFSET_BOX = tuple(C["offset_box"])
WINDOW = tuple(C["window"])
JRC_TILES = tuple(C["jrc_tiles"])
TILES = tuple((tuple(t.split("/")) for t in C.get("tiles", [])))
REF_SCENES = tuple(C.get("reference_scenes", []))
MAP_SCENES = tuple(C.get("map_scenes", []))
PERENNIAL = tuple(C.get("perennial", []))
