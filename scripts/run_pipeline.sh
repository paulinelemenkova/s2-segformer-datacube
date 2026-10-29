#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/src
WORK=${WORK:-work}
CHIPS=${CHIPS:-$WORK/chips}
EPOCHS=${EPOCHS:-25}
MAX_CHIPS=${MAX_CHIPS:-3000}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-4}
POINTS=${POINTS:-$ROOT/config/points.csv}
SCENES=${SCENES:-$ROOT/config/scenes.json}
export SITES=${SITES:-$ROOT/config/sites.json}
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
REF=$(python3 -c "from siteconfig import REF_SCENES; print(' '.join(REF_SCENES))")
MAPS=$(python3 -c "from siteconfig import MAP_SCENES; print(' '.join(MAP_SCENES))")
mkdir -p "$WORK/runs" "$WORK/logs"

usage() {
  echo "usage: $0 {check|chips|folds|points|final|series|qsens|pack|all}"
  echo "env: WORK CHIPS EPOCHS MAX_CHIPS BATCH WORKERS POINTS SCENES SITES QFOLDS"
}

train() {
  local arch=$1 q=$2 k=$3 name
  name=$(python3 -c "from train_segformer import run_name; print(run_name('$arch', $q, $k))")
  if [ -f "$WORK/runs/log_$name.json" ]; then echo "skip $name (done)"; return; fi
  python3 "$SRC/train_segformer.py" --chips "$WORK/chips.csv" --chip_dir "$CHIPS" --arch "$arch" --q "$q" \
      --fold "$k" --epochs "$EPOCHS" --batch "$BATCH" --max_chips "$MAX_CHIPS" --workers "$WORKERS" \
      --out "$WORK/runs" 2>&1 | tee "$WORK/logs/$name.txt"
}

predict_ref() {
  local run=$1 ck=$2 k
  k=$(python3 -c "import torch; print(torch.load('$ck', map_location='cpu', weights_only=False)['fold'])")
  python3 "$SRC/predict_segformer.py" --ckpt "$ck" --scenes $REF --t 1 --out "$WORK/pred_ref/$run/fold$k" \
      --cache "$WORK/cache"
}

stage_check() {
  python3 - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0])
for m in ("numpy", "pandas", "rasterio", "pyproj", "torch", "transformers", "segmentation_models_pytorch",
          "albumentations", "sklearn"):
    try:
        print(f"  {m:28s} {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"  {m:28s} MISSING ({e})")
from common import JRC, JRC_TILES, device, href
from siteconfig import MAP_SCENES, REF_SCENES
dev = device()
print("device:", dev)
if dev.type == "cpu":
    print("  WARNING: no CUDA device found")
import rasterio
sid = (MAP_SCENES or REF_SCENES)[0]
for name, url in (("Sentinel-2 COG", href(sid, "B03")), ("JRC occurrence", JRC.format(JRC_TILES[0]))):
    try:
        with rasterio.open(url) as src:
            print(f"network test {name}: OK ({src.width} x {src.height})")
    except Exception as e:
        print(f"network test {name}: FAILED ({e})")
try:
    from transformers import SegformerConfig
    SegformerConfig.from_pretrained("nvidia/mit-b2")
    print("network test Hugging Face (nvidia/mit-b2): OK")
except Exception as e:
    print(f"network test Hugging Face: FAILED ({type(e).__name__})")
EOF
}

stage_chips() {
  python3 "$SRC/make_chips.py" "$SCENES" "$WORK" 12
  python3 -c "import pandas as pd; d=pd.read_csv('$WORK/chips.csv'); print(d.groupby(['lake','fold']).size().unstack())"
}

stage_folds() {
  [ -f "$WORK/block_folds.json" ] || python3 "$SRC/refold.py" "$WORK"
  for k in 0 1 2 3 4; do
    train segformer 0.7 $k
    predict_ref segformer "$WORK/runs/segformer_fold$k.pt"
    train unet 0.7 $k
    predict_ref unet "$WORK/runs/unet_fold$k.pt"
    if [ ! -d "$WORK/pred_ref/rf/fold$k" ]; then
      python3 "$SRC/train_rf.py" --chips "$WORK/chips.csv" --chip_dir "$CHIPS" --fold $k --points "$POINTS" \
          --scenes $REF --max_chips "$MAX_CHIPS" --out "$WORK/pred_ref/rf/fold$k.tmp" --cache "$WORK/cache"
      mv "$WORK/pred_ref/rf/fold$k.tmp" "$WORK/pred_ref/rf/fold$k"
    fi
  done
}

stage_final() {
  train segformer 0.7 -1
  python3 "$SRC/predict_segformer.py" --ckpt "$WORK/runs/segformer_fold-1.pt" --scenes $MAPS --t 20 \
      --out "$WORK/pred" --cache "$WORK/cache"
}

stage_series() {
  python3 "$SRC/series_counts.py" --ckpt "$WORK/runs/segformer_fold-1.pt" --scenes_file "$SCENES" \
      --out "$WORK/series_counts.json"
}

stage_qsens() {
  for q in 0.3 0.5 0.9; do
    for k in ${QFOLDS:-0}; do
      train segformer $q $k
      predict_ref "segformer_q$q" "$WORK/runs/segformer_q${q}_fold$k.pt"
    done
  done
}

stage_points() {
  python3 "$SRC/predict_points.py" "$WORK" "$POINTS"
  for k in 0 1 2 3 4; do
    if [ ! -d "$WORK/pred_ref_nomask/rf/fold$k" ]; then
      python3 "$SRC/train_rf.py" --chips "$WORK/chips.csv" --chip_dir "$CHIPS" --fold $k --points "$POINTS" \
          --scenes $REF --max_chips "$MAX_CHIPS" --no_mask --out "$WORK/pred_ref_nomask/rf/fold$k.tmp" \
          --cache "$WORK/cache"
      mv "$WORK/pred_ref_nomask/rf/fold$k.tmp" "$WORK/pred_ref_nomask/rf/fold$k"
    fi
  done
  python3 "$SRC/sample_predictions.py" "$POINTS" "$WORK/pred_ref_nomask" "$WORK/ref_predictions_rf_nomask.csv" \
      "$WORK/block_folds.json"
}

stage_pack() {
  python3 "$SRC/sample_predictions.py" "$POINTS" "$WORK/pred_ref" "$WORK/ref_predictions.csv" "$WORK/block_folds.json"
  (cd "$WORK" && zip -q -r ../results.zip ref_predictions*.csv chips.csv block_folds.json runs/log_*.json logs \
      pred $( [ -f series_counts.json ] && echo series_counts.json ))
  ls -lh results.zip
}

case "${1:-}" in
  check|chips|folds|points|final|series|qsens|pack) "stage_$1" ;;
  all) for s in check chips folds points final series pack; do "stage_$s"; done ;;
  *) usage; exit 1 ;;
esac
