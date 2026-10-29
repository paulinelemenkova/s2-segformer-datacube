# s2-segformer-datacube

**Cloud-native multispectral datacube engine for weakly supervised semantic segmentation, Bayesian uncertainty and spectral colorimetry of Sentinel-2 MSI imagery: SegFormer / U-Net / Random Forest on CUDA GPU (Google Colab), GRASS GIS hydrology and explainable gradient boosting.**

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c) ![transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![grass](https://img.shields.io/badge/GRASS%20GIS-8.x-green) ![colab](https://img.shields.io/badge/Google-Colab%20T4-orange) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

## Description

An end-to-end, scriptable remote sensing pipeline that turns a cloud-optimised GeoTIFF (COG) archive into analysis-ready tensors, trains hierarchical vision transformers and convolutional baselines under label noise, and propagates pixel-level predictive uncertainty into long time series. No proprietary platform is required: the datacube is assembled on the fly through HTTP range requests against a public STAC-indexed object store, warped in memory with GDAL virtual rasters, and streamed into PyTorch.

The core is a **ten-band MiT-B2 SegFormer** whose patch-embedding stem is re-initialised from ImageNet RGB weights for multispectral input, trained with the noise-robust **Generalised Cross-Entropy (GCE)** loss on weak labels produced by a deterministic spectral-index rule cascade. Inference uses **Monte Carlo dropout** to estimate pixel-wise predictive entropy. Evaluation uses **spatially blocked k-fold cross-validation** with balanced fold assignment, out-of-fold scoring against independent reference points, and paired **McNemar** tests between models.

## Technical highlights

- **Cloud-native datacube**: STAC/COG access over HTTPS (`/vsicurl`), COG overview pyramids selected per target resolution, `WarpedVRT` on-the-fly reprojection, threaded band reads, exponential back-off on transient I/O errors, and scene-level caching of float16 tensors.
- **Radiometric harmonisation**: data-driven detection and removal of the Level-2A BOA additive offset (processing baseline ≥ 04.00) from the dark-target percentile of SWIR-2.
- **Weak supervision**: rule cascade on MNDWI, NDVI, AWEI_sh and visible–NIR–SWIR brightness, conditioned on a multi-decadal water-occurrence prior; Scene Classification Layer (SCL) masking of cloud, shadow, cirrus, snow and saturated pixels; ambiguous pixels sent to the ignore index.
- **Deep learning**:
  - SegFormer MiT-B2 (Hugging Face Transformers), with a ten-channel stem re-initialised from mean RGB weights.
  - U-Net with a ResNet-34 encoder (segmentation-models-pytorch).
  - GCE loss (q = 0.7), AdamW, polynomial learning-rate decay.
  - CUDA automatic mixed precision (fp16 autocast and GradScaler).
  - Albumentations geometric and radiometric augmentation.
  - Resumable checkpointing.
- **Inference**: overlapping sliding-window prediction (256 px tiles, 192 px stride) with softmax averaging; T-sample MC-dropout forward passes, posterior mean softmax and predictive entropy maps written as GeoTIFF.
- **Classical machine learning baseline**: pixel-wise Random Forest (500 trees, Gini impurity) on 14 spectral features, trained on the same folds.
- **Validation design**: spatial blocks, greedy balanced fold assignment, a withheld temporal hold-out year, per-class IoU/F1, mIoU, macro F1, confusion matrices and exact binomial McNemar tests.
- **Water colorimetry**: Forel–Ule index from MSI reflectance through CIE 1931 tristimulus integration, chromaticity hue angle and a fifth-order sensor-specific polynomial correction.
- **Time-series analytics**: Hamed–Rao modified Mann–Kendall test, Theil–Sen slope, and PELT change-point detection (`ruptures`) on deseasonalised, standardised anomalies.
- **Explainable AI**: LightGBM gradient boosting on lagged reanalysis and vegetation-index predictors, TreeSHAP attributions, leave-one-year-out validation, and a ridge-regression benchmark.
- **GRASS GIS hydrology**: `r.watershed` D8 flow routing with explicit depressions, vectorised pointer-jumping (O(log L)) labelling of flow-path termini in NumPy through `grass.script.array`, and vector and raster export (`r.to.vect`, `v.out.ogr`, `r.out.gdal`).
- **Archive scanning**: S3 prefix listing of the MGRS tile and month hierarchy, de-duplication of reprocessed items, and per-pixel valid-observation and water-frequency counts from SCL overviews.

## Repository layout

```
config/            site and scene configuration (examples; real files are git-ignored)
src/
  siteconfig.py         configuration loader
  common.py             COG/STAC I/O, warping, BOA harmonisation, rule cascade, masks, features
  make_chips.py         256 x 256 training chip generation with spatial blocks
  refold.py             balanced spatial fold assignment
  train_segformer.py    SegFormer / U-Net training (GCE, AMP, MC dropout)
  predict_segformer.py  sliding-window inference, class and entropy GeoTIFFs
  predict_points.py     out-of-fold inference at reference points
  train_rf.py           Random Forest baseline
  series_counts.py      model inference over a full image time series
  sample_predictions.py fold-aware sampling of prediction rasters
  accuracy_eval.py      accuracy metrics and McNemar tests
  area_stats.py         trend and change-point analysis
  fui_core.py           Forel-Ule colorimetry
  driver_model.py       LightGBM + SHAP attribution
  datacube_scan.py      archive scanning and observation counts
scripts/run_pipeline.sh resumable stage driver
grass/watershed_basins.sh GRASS GIS flow routing and basin labelling
notebooks/colab_pipeline.ipynb Google Colab (CUDA) runner
```

## Requirements

- Python ≥ 3.10.
- PyTorch with CUDA. An NVIDIA T4 (16 GB) is sufficient. CPU and Apple MPS fall back automatically.
- GDAL/rasterio with network access to public COG endpoints.
- GRASS GIS 8.x (for `grass/`).

```bash
pip install -r requirements.txt
```

## Configuration

Copy the examples and fill in your own areas of interest, projected CRS, tiles and scene identifiers:

```bash
cp config/sites.example.json  config/sites.json
cp config/scenes.example.json config/scenes.json
cp config/points.example.csv  config/points.csv
```

`sites.json` contains the following keys:

- `crs`: the projected CRS.
- `sites`: per-site lon/lat boxes.
- `offset_box`: the dark-target window used for BOA harmonisation.
- `window`: the union window.
- `jrc_tiles`: the occurrence-prior tiles.
- `tiles`: the MGRS tiles.
- `reference_scenes`, `map_scenes` and `perennial`: the scene lists and the perennial sites.

## Usage

```bash
bash scripts/run_pipeline.sh check    # environment, CUDA device, endpoints
bash scripts/run_pipeline.sh chips    # chip extraction (CPU, network)
bash scripts/run_pipeline.sh folds    # k-fold SegFormer, U-Net, Random Forest (GPU)
bash scripts/run_pipeline.sh points   # out-of-fold inference at reference points
bash scripts/run_pipeline.sh final    # final model + MC-dropout maps
bash scripts/run_pipeline.sh series   # inference over the full time series
bash scripts/run_pipeline.sh pack     # collect results
```

Environment variables: `WORK`, `CHIPS`, `EPOCHS`, `BATCH`, `MAX_CHIPS`, `WORKERS`, `SITES`, `SCENES`, `POINTS`.

Each stage skips work that is already done, so a stage interrupted by the session limits of a free Colab GPU continues from the last completed model.

```bash
python3 src/accuracy_eval.py labels.csv config/points.csv work/ref_predictions.csv
python3 src/area_stats.py work/series_counts.json area_stats.json
bash grass/watershed_basins.sh dem.tif depressions.tif 4 EPSG_CODE
```

## Google Colab

`notebooks/colab_pipeline.ipynb` mounts Google Drive and copies the chips to the VM's local SSD for I/O throughput. It installs the dependencies and runs the stages on a CUDA T4. Chips, checkpoints and predictions are persisted in Drive.

## Topics

`remote-sensing` `image-processing` `semantic-segmentation` `pattern-recognition` `computer-vision` `deep-learning` `vision-transformer` `segformer` `unet` `pytorch` `cuda` `gpu` `google-colab` `huggingface-transformers` `weak-supervision` `noisy-labels` `uncertainty-quantification` `monte-carlo-dropout` `random-forest` `lightgbm` `shap` `explainable-ai` `sentinel-2` `multispectral` `cloud-native` `datacube` `stac` `cog` `gdal` `rasterio` `grass-gis` `time-series-analysis` `change-point-detection` `colorimetry`

## License

MIT © Polina Lemenkova
