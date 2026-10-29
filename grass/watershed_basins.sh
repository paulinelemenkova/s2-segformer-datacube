#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 DEM.tif DEPRESSIONS.tif [N_TARGETS] [EPSG]"
  exit 1
fi
DEM=$(realpath "$1")
DEP=$(realpath "$2")
N=${3:-4}
EPSG=${4:-4326}
OUT=${OUT:-$PWD}
WRK=$(mktemp -d)
cd "$WRK"
cp "$DEM" dem.tif
cp "$DEP" lakes.tif

cat > grass_steps.sh <<EOS
set -e
r.in.gdal input=dem.tif output=dem --quiet
r.in.gdal input=lakes.tif output=lakes --quiet
g.region raster=dem
r.mapcalc "lakes_i = round(lakes)" --quiet
r.mapcalc "dep = if(lakes_i > 0, 1, null())" --quiet
r.watershed elevation=dem depression=dep drainage=dir memory=1500 --quiet
N=$N python3 label_basins.py
r.mapcalc "basin = if(basin0 > 0, basin0, null())" --quiet
r.to.vect -s input=basin output=basin_v type=area --quiet
v.out.ogr input=basin_v output=basins.gpkg format=GPKG --quiet --overwrite
r.out.gdal input=basin output=basins.tif type=Int16 createopt=COMPRESS=DEFLATE --quiet --overwrite
echo "basin_id  basin_km2  depression_km2" > basin_stats.txt
for i in \$(seq 1 $N); do
  b=\$(r.stats -an input=basin | awk -v i=\$i '\$1==i{printf "%.1f",\$2/1e6}')
  l=\$(r.stats -an input=lakes_i | awk -v i=\$i '\$1==i{printf "%.1f",\$2/1e6}')
  echo "\$i  \$b  \$l" >> basin_stats.txt
done
r.mapcalc "edge = if(row()==1 || row()==nrows() || col()==1 || col()==ncols(), basin, null())" --quiet
echo "basins touching edge: \$(r.stats -n input=edge | tr '\n' ' ')" >> basin_stats.txt
EOS

cat > label_basins.py <<'EOP'
import os

import numpy as np
from grass.script import array as garray

n_targets = int(os.environ["N"])
d = garray.array(mapname="dir")
lk = garray.array(mapname="lakes_i")
nr, nc = d.shape
idx = np.arange(nr * nc).reshape(nr, nc)
code = np.asarray(d).astype(np.int64)
dr = {1: -1, 2: -1, 3: -1, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0}
dc = {1: 1, 2: 0, 3: -1, 4: -1, 5: -1, 6: 0, 7: 1, 8: 1}
rr, cc = np.divmod(idx, nc)
nxt = idx.copy()
for k in range(1, 9):
    m = code == k
    r2 = np.clip(rr[m] + dr[k], 0, nr - 1)
    c2 = np.clip(cc[m] + dc[k], 0, nc - 1)
    nxt[m] = r2 * nc + c2
nxt = nxt.ravel()
for _ in range(40):
    n2 = nxt[nxt]
    if np.array_equal(n2, nxt):
        break
    nxt = n2
lake = np.asarray(lk).ravel().astype(np.int64)
lab = lake[nxt].reshape(nr, nc)
lab[lab > n_targets] = 0
lab[code.reshape(nr, nc) < 0] = 0
out = garray.array(dtype=np.int32)
out[...] = lab
out.write(mapname="basin0", overwrite=True)
EOP

grass --tmp-location "EPSG:$EPSG" --exec bash grass_steps.sh
cp basins.gpkg basins.tif basin_stats.txt "$OUT"/
cat basin_stats.txt
