# Monocular Depth Estimation for Urban Tree Monitoring

Code for the MSc AI thesis (Julian van Pol, VU Amsterdam): estimating urban tree canopy
height in Amsterdam from aerial imagery, validated against AHN LiDAR. A seasonal **Siamese**
adaptation of **Depth Anything V2** passes a summer and winter image through a shared backbone;
a small CNN fuses the depth maps (+ optional CIR/NDVI) into canopy height in metres. Full
details are in the thesis report.

## Key results

Per-pixel metrics on tree pixels. Metre-level accuracy within a year; errors and a
year-specific bias grow across years and to a second city.

| Evaluation set      | MAE (m) | RMSE (m) | R²   | Bias (m) |
|---------------------|:-------:|:--------:|:----:|:--------:|
| Amsterdam test 2022 | 1.95    | 2.45     | 0.32 | −0.31    |
| Amsterdam test 2024 | 2.22    | 2.72     | 0.17 | +0.90    |
| Amsterdam test 2025 | 2.26    | 2.78     | 0.14 | +0.87    |
| Utrecht 2022        | 2.29    | 2.84     | 0.26 | −1.45    |

## Map structure

```
MSc-AI-Thesis-Urban-Tree-Canopy-Height/
├── depth_anything_v2_siamese_ThesisFinal.ipynb   ← MAIN NOTEBOOK
├── src/
│   ├── dataset.py                    Data loading, tree mask, PyTorch Dataset
│   ├── depth_anything_v2_siamese.py  The Siamese model + DepthFusionHead
│   ├── simple_dav2_training.py       Config, DataLoaders, training, evaluation, plots
│   ├── inference.py                  Sliding-window inference over a full raster (for vondelpark app)
│   └── depth_anything_v2_utils.py    Loads the official DA-V2 weights, device helpers
├── data/processed/                   ← aligned GeoTIFF rasters (+ utrecht/)
├── outputs/                          ← checkpoints/, figures/, height_maps/, patch_index/
├── third_party/Depth-Anything-V2/    ← official DA-V2 repo + pretrained weights
├── amsterdam_ahn_merged.tif          ← raw merged AHN DSM (source)
└── sampled_veg_ams.tif               ← Amsterdam vegetation mask (optional)
```

## Getting started

Requires Python 3.9+ with `torch torchvision rasterio numpy matplotlib scipy`.

This repository holds the **code only** — the large files (data, DA-V2 weights, trained model)
are not included. To run it, add three things:

1. **Depth Anything V2** — clone the official repo into `third_party/Depth-Anything-V2/`
   (`git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2`)
   and download the ViT-B weights (~380 MB) to
   `third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth`:
   https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

2. **Input rasters** — place the GeoTIFFs in `data/processed/` (see `data/processed/README.md`).

3. **Trained model** — not included (too large for GitHub). Run the training cell to regenerate it,
   or place a trained `..._best.pt` in `outputs/checkpoints/`.

Then open the notebook and run top to bottom; to evaluate without retraining, skip the training
cell (needs the checkpoint from step 3).

## Data & outputs

`data/processed/` holds the aligned 0.5 m rasters (summer/winter RGB, AHN, CIR for 2022/2024/2025
+ Utrecht); the winter image and AHN reference are fixed across years. `outputs/figures/` contains
the report figures; the trained checkpoint and height maps are regenerated when you run the notebook.
