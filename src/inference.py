"""
What this file does:
  - Slides a 256×256 pixel window across the entire Amsterdam raster
  - Predicts tree height per patch with the trained Siamese DA-V2 model
  - Averages overlapping patches so no hard boundaries are visible
  - Saves the result as a GeoTIFF (same size as the input)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from src.dataset import ndvi_from_cir, scale_raster_channels
from src.depth_anything_v2_utils import checkpoint_path, get_best_device
from src.simple_dav2_training import (
    SimpleDAV2Config,
    denormalize_prediction,
    make_model,
)


def _read_patch(src: rasterio.io.DatasetReader, x: int, y: int, patch_size: int) -> np.ndarray:
    """Read one patch from a raster as a float32 array in [0, 1]."""
    window = Window(x, y, patch_size, patch_size)
    patch = src.read(window=window).astype(np.float32)
    return scale_raster_channels(patch)


def run_inference(
    checkpoint_path: Path,
    summer_path: Path,
    winter_path: Path,
    output_path: Path,
    cir_path: Path | None = None,
    stride: int = 128,
    patch_size: int = 256,
    device: torch.device | None = None,
    max_patches: int | None = None,
    clip_bounds: tuple[float, float, float, float] | None = None,
) -> None:

    device = device or get_best_device()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model and config from checkpoint
    print(f"Loading checkpoint: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = SimpleDAV2Config(**{
        k: v for k, v in ckpt["config"].items()
        if k in SimpleDAV2Config.__dataclass_fields__
    })

    model = make_model(config, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded  |  use_cir={config.use_cir}  use_ndvi={config.use_ndvi}")

    if (config.use_cir or config.use_ndvi) and cir_path is None:
        raise ValueError(
            "This checkpoint was trained with use_cir=True or use_ndvi=True, "
            "but cir_path was not passed to run_inference()."
        )

    # Open rasters
    summer_src = rasterio.open(summer_path)
    winter_src = rasterio.open(winter_path)
    cir_src    = rasterio.open(cir_path) if (config.use_cir or config.use_ndvi) else None

    width     = summer_src.width
    height    = summer_src.height
    profile   = summer_src.profile.copy()
    transform = summer_src.transform
    
    if clip_bounds is not None:
        left, bottom, right, top = clip_bounds
        x_off = int((left          - transform.c) / transform.a)
        y_off = int((transform.f   - top)         / (-transform.e))
        x_end = int((right         - transform.c) / transform.a)
        y_end = int((transform.f   - bottom)      / (-transform.e))
        x_off = max(0, x_off);  y_off = max(0, y_off)
        x_end = min(width, x_end);  y_end = min(height, y_end)
        print(f"Subregion: pixels ({x_off},{y_off}) → ({x_end},{y_end})")
    else:
        x_off, y_off = 0, 0
        x_end, y_end = width, height

    area_width  = x_end - x_off
    area_height = y_end - y_off
    print(f"Raster size: {width}×{height} pixels  |  Processing area: {area_width}×{area_height} pixels")
    print(f"Stride: {stride}px  →  patch overlap: {patch_size - stride}px")

    prediction_sum = np.zeros((area_height, area_width), dtype=np.float32)
    count          = np.zeros((area_height, area_width), dtype=np.float32)

    xs    = list(range(x_off, x_end - patch_size + 1, stride))
    ys    = list(range(y_off, y_end - patch_size + 1, stride))
    total = len(xs) * len(ys)
    print(f"Total patches to process: {total}")
    if max_patches is not None:
        print(f"Test mode: at most {max_patches} patches will be processed.")

    # Sliding window inference
    processed = 0
    with torch.no_grad():
        for y in ys:
            for x in xs:
                if max_patches is not None and processed >= max_patches:
                    break

                summer_patch = _read_patch(summer_src, x, y, patch_size)
                winter_patch = _read_patch(winter_src, x, y, patch_size)

                inputs = {
                    "summer": torch.from_numpy(summer_patch).unsqueeze(0).to(device),
                    "winter": torch.from_numpy(winter_patch).unsqueeze(0).to(device),
                }

                # Read CIR and/or NDVI only if the model expects them
                if config.use_cir or config.use_ndvi:
                    cir_patch = _read_patch(cir_src, x, y, patch_size)
                    if config.use_cir:
                        inputs["cir"] = torch.from_numpy(cir_patch).unsqueeze(0).to(device)
                    if config.use_ndvi:
                        ndvi_patch = ndvi_from_cir(
                            cir_patch,
                            nir_band=config.cir_nir_band - 1,
                            red_band=config.cir_red_band - 1,
                        )
                        inputs["ndvi"] = torch.from_numpy(ndvi_patch).unsqueeze(0).unsqueeze(0).to(device)

                pred = model(**inputs)

                if config.normalize_target:
                    pred = denormalize_prediction(pred, config)

                pred_np = pred.squeeze().cpu().numpy()

                ry = y - y_off
                rx = x - x_off
                prediction_sum[ry:ry + patch_size, rx:rx + patch_size] += pred_np
                count[ry:ry + patch_size, rx:rx + patch_size]          += 1.0

                processed += 1
                if processed % 500 == 0:
                    print(f"  {processed}/{total} patches processed...")

            if max_patches is not None and processed >= max_patches:
                break

    # Compute the average
    valid      = count > 0
    height_map = np.where(valid, prediction_sum / np.maximum(count, 1), 0.0)
    print(f"Height range in output: {height_map[valid].min():.2f}m – {height_map[valid].max():.2f}m")

    # Save as GeoTIFF
    out_transform = rasterio.transform.from_origin(
        west  = transform.c + x_off * transform.a,
        north = transform.f + y_off * transform.e,
        xsize = transform.a,
        ysize = -transform.e,
    )
    profile.update(count=1, dtype="float32", compress="lzw",
                   width=area_width, height=area_height, transform=out_transform)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(height_map.astype(np.float32), 1)
    print(f"Height map saved: {output_path}")

    summer_src.close()
    winter_src.close()
    if cir_src is not None:
        cir_src.close()
