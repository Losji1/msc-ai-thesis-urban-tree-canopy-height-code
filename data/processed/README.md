# data/processed/ — required rasters (not included)

The imagery/LiDAR is not part of this handoff (files were too big). Download the imagery from **PDOK** and clip to the study extent (EPSG:28992, 0.5 m), and save the
files here with exactly these names:

- `clipped_2022_end.tif`, `cir_clipped_2022.tif`, `ahn4_2022_ams.tif` — summer RGB / CIR / AHN DSM (2022)
- `winter_clipped_lowres.tif` — winter RGB (fixed reference, used for all years)
- `rgb_0.5_ams.tif`, `clipped_cir_ams.tif` — summer RGB / CIR (2024)
- `clipped_2025_end.tif`, `cir_clipped_2025.tif` — summer RGB / CIR (2025)
- `utrecht/` — `summer_clipped_utrecht_2022.tif`, `winter_clipped_utrecht_2022.tif`, `ahn4_utrecht.tif`, `cir_utrecht_2022.tif`

Paths match the defaults in `SimpleDAV2Config` (`src/simple_dav2_training.py`).
