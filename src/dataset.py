"""
Provides the SiameseTreeDataset: given patch coordinates, it reads the aligned
GeoTIFF rasters (summer RGB, winter RGB, AHN height, optional CIR), builds a
tree-pixel mask from the AHN height, and returns one (summer, winter, target,
mask) sample per patch. Optional extras: CIR/NDVI channels and random-flip
augmentation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RasterPaths:
    """Paths to the four input rasters."""
    summer: Path      # Summer RGB  (3 channels, uint8 or float)
    winter: Path      # Winter RGB  (3 channels, uint8 or float)
    ahn: Path         # AHN height  (1 channel, float32, metres above ground)
    cir: Path | None = None         # Colour-Infrared (optional)
    vegetation: Path | None = None  # Amsterdam vegetation mask (optional)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_raster_alignment(paths: RasterPaths) -> dict[str, object]:
    """Check that all rasters share the same resolution, CRS and extent.

    Raises an AssertionError if something is off — run this once before
    training.
    """
    metadata = {}

    raster_items = [
        ("summer", paths.summer),
        ("winter", paths.winter),
        ("ahn", paths.ahn),
    ]
    if paths.cir is not None:
        raster_items.append(("cir", paths.cir))
    if paths.vegetation is not None:
        raster_items.append(("vegetation", paths.vegetation))

    for name, path in raster_items:
        with rasterio.open(path) as src:
            metadata[name] = {
                "count": src.count,
                "width": src.width,
                "height": src.height,
                "crs": src.crs,
                "res": src.res,
                "bounds": src.bounds,
                "transform": src.transform,
                "dtype": src.dtypes,
            }

    summer = metadata["summer"]

    for name, info in metadata.items():
        assert summer["crs"] == info["crs"], f"CRS mismatch: {name}"
        assert summer["res"] == info["res"], f"Resolution mismatch: {name}"

        # The vegetation raster may be larger (it is cropped via the window)
        if name == "vegetation":
            vb = info["bounds"]
            sb = summer["bounds"]
            assert vb.left <= sb.left and vb.right >= sb.right, "vegetation too small (left/right)"
            assert vb.bottom <= sb.bottom and vb.top >= sb.top, "vegetation too small (top/bottom)"
            continue

        assert summer["width"] == info["width"], f"Width mismatch: {name}"
        assert summer["height"] == info["height"], f"Height mismatch: {name}"
        assert summer["bounds"] == info["bounds"], f"Bounds mismatch: {name}"
        assert summer["transform"] == info["transform"], f"Transform mismatch: {name}"

    return metadata


def read_single_band_matching_window(
    reference_src: rasterio.io.DatasetReader,
    target_src: rasterio.io.DatasetReader,
    reference_window: Window,
    out_shape: tuple[int, int],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Read one band from a raster that may have a different extent/resolution.
    Uses the geographic coordinates of the reference raster to find the
    corresponding window in the target raster.
    """
    left, bottom, right, top = window_bounds(reference_window, reference_src.transform)
    target_window = from_bounds(left, bottom, right, top, transform=target_src.transform)
    target_window = target_window.round_offsets().round_lengths()
    return target_src.read(
        1,
        window=target_window,
        out_shape=out_shape,
        boundless=True,
        fill_value=fill_value,
        resampling=Resampling.nearest,
    )

def build_tree_mask(
    ahn_array: np.ndarray,
    min_height: float = 2.0,
    max_height: float = 25.0,
    vegetation_array: np.ndarray | None = None,
    vegetation_threshold: float = 0.5,
    close_radius: int = 0,
) -> np.ndarray:
    """Build a binary mask of pixels that are likely a tree.

    A pixel counts as a tree if:
      - the AHN height is finite and between min_height and max_height
      - (optionally) the vegetation score is above vegetation_threshold

    close_radius > 0: morphological closing fills small gaps within tree crowns.
    """
    height_mask = np.isfinite(ahn_array) & (ahn_array >= min_height) & (ahn_array <= max_height)
    if vegetation_array is None:
        mask = height_mask
    else:
        vegetation_mask = np.isfinite(vegetation_array) & (vegetation_array > vegetation_threshold)
        mask = height_mask & vegetation_mask

    if close_radius > 0:
        from scipy.ndimage import binary_closing
        struct = np.ones((close_radius * 2 + 1, close_radius * 2 + 1), dtype=bool)
        mask = binary_closing(mask, structure=struct)

    return mask


# Split patches geographically into three sets: train, val and test.
def spatial_train_val_test_split(coords: Iterable[tuple[int, int]], raster_width: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    coords = list(coords)
    split_train = int(raster_width * train_ratio)
    split_val   = int(raster_width * (train_ratio + val_ratio))
    train_coords = [(x, y) for (x, y) in coords if x < split_train]
    val_coords   = [(x, y) for (x, y) in coords if split_train <= x < split_val]
    test_coords  = [(x, y) for (x, y) in coords if x >= split_val]
    return train_coords, val_coords, test_coords


def scale_raster_channels(array: np.ndarray) -> np.ndarray:
    dtype = array.dtype
    array = array.astype(np.float32)

    if np.issubdtype(dtype, np.integer):
        return array / float(np.iinfo(dtype).max)

    finite = np.isfinite(array)
    if finite.any() and float(np.nanmax(array[finite])) > 1.5:
        return array / 255.0
    return array


# Compute NDVI from the CIR channels.
def ndvi_from_cir(cir: np.ndarray, nir_band: int = 0, red_band: int = 1) -> np.ndarray:
    nir = cir[nir_band]
    red = cir[red_band]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


# Flip a 2D or 3D array horizontally and/or vertically
def _flip_array_2d(arr: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    if horizontal:
        arr = arr[..., ::-1]
    if vertical:
        arr = arr[..., ::-1, :]
    return np.ascontiguousarray(arr)


def smooth_target_within_mask(target: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    target_filled = target * (mask > 0.5)

    # Smooth the whole field
    smoothed = gaussian_filter(target_filled.astype(np.float64), sigma=sigma)
    mask_smoothed = gaussian_filter((mask > 0.5).astype(np.float64), sigma=sigma)
    mask_smoothed = np.where(mask_smoothed > 0.01, mask_smoothed, 1.0)
    smoothed = (smoothed / mask_smoothed).astype(np.float32)

    # Use the smoothed value only where the original mask is valid
    return np.where(mask > 0.5, smoothed, target)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SiameseTreeDataset(Dataset):

    def __init__(
        self,
        paths: RasterPaths,
        coords: list[tuple[int, int]],
        patch_size: int = 256,
        min_height: float = 2.0,
        max_height: float = 25.0,
        augment: bool = False,
        smooth_target_sigma: float = 0.0,
        use_cir: bool = False,
        use_ndvi: bool = False,
        use_vegetation_mask: bool = False,
        vegetation_threshold: float = 0.5,
        use_ndvi_mask: bool = False,
        ndvi_mask_threshold: float = 0.2,
        cir_nir_band: int = 1,
        cir_red_band: int = 2,
        mask_close_radius: int = 0,

    ) -> None:
        self.paths = paths
        self.coords = coords
        self.patch_size = patch_size
        self.min_height = min_height
        self.max_height = max_height
        self.augment = augment
        self.smooth_target_sigma = smooth_target_sigma
        self.use_cir = use_cir
        self.use_ndvi = use_ndvi
        self.use_vegetation_mask = use_vegetation_mask
        self.vegetation_threshold = vegetation_threshold
        self.use_ndvi_mask = use_ndvi_mask
        self.ndvi_mask_threshold = ndvi_mask_threshold
        self.cir_nir_band = cir_nir_band
        self.cir_red_band = cir_red_band
        self.mask_close_radius = mask_close_radius


        if (self.use_cir or self.use_ndvi or self.use_ndvi_mask) and self.paths.cir is None:
            raise ValueError("RasterPaths.cir must be set when use_cir, use_ndvi or use_ndvi_mask is True.")
        if self.use_vegetation_mask and self.paths.vegetation is None:
            raise ValueError("RasterPaths.vegetation must be set when use_vegetation_mask is True.")

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        x, y = self.coords[idx]
        window = Window(x, y, self.patch_size, self.patch_size)

        # Load all arrays as numpy
        with rasterio.open(self.paths.summer) as src:
            summer = src.read([1, 2, 3], window=window).astype(np.float32) / 255.0  # (3, H, W)

        with rasterio.open(self.paths.winter) as src:
            winter = src.read([1, 2, 3], window=window).astype(np.float32) / 255.0  # (3, H, W)

        with rasterio.open(self.paths.ahn) as ahn_src:
            target = ahn_src.read(1, window=window).astype(np.float32)  # (H, W)

            vegetation = None
            if self.use_vegetation_mask:
                with rasterio.open(self.paths.vegetation) as vegetation_src:
                    vegetation = read_single_band_matching_window(
                        reference_src=ahn_src,
                        target_src=vegetation_src,
                        reference_window=window,
                        out_shape=(self.patch_size, self.patch_size),
                        fill_value=0.0,
                    ).astype(np.float32)  # (H, W)

        # Load CIR for model input (use_cir/use_ndvi) and/or the NDVI mask
        cir = None
        if self.use_cir or self.use_ndvi or self.use_ndvi_mask:
            with rasterio.open(self.paths.cir) as src:
                cir = scale_raster_channels(src.read([1, 2, 3], window=window))  # (3, H, W)

        if self.use_ndvi_mask and cir is not None:
            ndvi = ndvi_from_cir(cir, nir_band=self.cir_nir_band - 1, red_band=self.cir_red_band - 1)
            vegetation = (ndvi > self.ndvi_mask_threshold).astype(np.float32)

        # Tree mask: 1 = valid tree pixel, 0 = ignore
        mask = build_tree_mask(
                target,
                min_height=self.min_height,
                max_height=self.max_height,
                vegetation_array=vegetation,
                vegetation_threshold=self.vegetation_threshold,
                close_radius=self.mask_close_radius,
            )
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        if self.smooth_target_sigma > 0:
            target = smooth_target_within_mask(target, mask, sigma=self.smooth_target_sigma)

        # Random flips as augmentation
        if self.augment:
            flip_h = random.random() > 0.5  # horizontal (left-right)
            flip_v = random.random() > 0.5  # vertical (top-bottom)
            if flip_h or flip_v:
                summer = _flip_array_2d(summer, flip_h, flip_v)
                winter = _flip_array_2d(winter, flip_h, flip_v)
                target = _flip_array_2d(target, flip_h, flip_v)
                mask = _flip_array_2d(mask, flip_h, flip_v)
                if vegetation is not None:
                    vegetation = _flip_array_2d(vegetation, flip_h, flip_v)
                if cir is not None:
                    cir = _flip_array_2d(cir, flip_h, flip_v)

        # Build PyTorch tensors
        item: dict[str, torch.Tensor | int] = {
            "summer": torch.from_numpy(summer),                              # (3, H, W)
            "winter": torch.from_numpy(winter),                              # (3, H, W)
            "target": torch.from_numpy(target).unsqueeze(0),                # (1, H, W)
            "mask":   torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),  # (1, H, W)
            "x": x,
            "y": y,
        }

        if self.use_vegetation_mask and vegetation is not None:
            vegetation_mask = (vegetation > self.vegetation_threshold).astype(np.float32)
            item["vegetation"] = torch.from_numpy(vegetation_mask).unsqueeze(0)

        if cir is not None:
            if self.use_cir:
                item["cir"] = torch.from_numpy(cir)
            if self.use_ndvi:
                ndvi = ndvi_from_cir(
                    cir,
                    nir_band=self.cir_nir_band - 1,
                    red_band=self.cir_red_band - 1,
                )
                item["ndvi"] = torch.from_numpy(ndvi).unsqueeze(0)

        return item
