"""
Training loop for the Siamese DA-V2 height model.
"""

from __future__ import annotations

import time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import DataLoader
from dataclasses import dataclass
from pathlib import Path

from src.dataset import (
    RasterPaths,
    SiameseTreeDataset,
    build_tree_mask,
    check_raster_alignment,
    read_single_band_matching_window,
    spatial_train_val_test_split,
)
from src.depth_anything_v2_siamese import (
    load_depth_anything_v2_siamese_height_net,
)
from src.depth_anything_v2_utils import checkpoint_path, get_best_device


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimpleDAV2Config:
    """All settings for the experiment in one place.

    Change your options here; the rest of the code reads from this.
    """

    # Data paths
    summer_path:     Path = Path("data/processed/clipped_2022_end.tif")   # Summer RGB 2022
    winter_path:     Path = Path("data/processed/winter_clipped_lowres.tif")  # Winter RGB 2022 (fixed reference)
    ahn_path:        Path = Path("data/processed/ahn4_2022_ams.tif")      # AHN4 2022
    cir_path:        Path | None = None        # Path to CIR raster (required if use_cir=True)
    vegetation_path: Path | None = None        # Path to vegetation mask (required if use_vegetation_mask=True)
    repo_dir:        Path = Path("third_party/Depth-Anything-V2")
    output_dir:      Path = Path("outputs")

    # Data year (included in the checkpoint name)
    # This way a 2022 training never overwrites a 2024 checkpoint
    data_year: str = "2022"

    # Model
    encoder:     str = "vitb"   
    head_channels: int = 64 

    # Training data settings 
    patch_size:      int = 256
    stride:          int = 256
    batch_size:      int = 1
    num_workers:     int = 0  

    # Tree mask
    min_tree_height:    float = 4.0    
    max_tree_height:    float = 17.0   
    min_valid_ratio:    float = 0.05  

    # Train/val/test split
    train_ratio: float = 0.8           
    val_ratio:   float = 0.1                                 
        
    # Patch limit (None = use everything)
    max_train_patches: int | None = None
    max_val_patches:   int | None = None
    max_test_patches:  int | None = None
    seed: int = 42

    #  Augmentation 
    augment_train: bool = True
    smooth_target_sigma: float = 2.0

    # Optional extra input
    use_vegetation_mask: bool = False   
    vegetation_threshold: float = 0.5
    use_ndvi_mask: bool = False         
    ndvi_mask_threshold: float = 0.2   
    mask_close_radius: int = 0         
    display_mask_close_radius: int = 1  
    display_mask_fill_holes: bool = True
    use_cir:  bool = False   
    use_ndvi: bool = False 
    cir_nir_band: int = 1
    cir_red_band: int = 2

    # DA-V2 trainability
    train_da_depth_head:     bool = True    
    train_dino_backbone:     bool = False   
    share_depth_anything_weights: bool = True  
    include_rgb_in_head:     bool = True    

    depth_input_size: int = 518

    # Training hyperparameters
    epochs:         int   = 6
    head_lr:        float = 5e-4    # Learning rate for the height head 
    da_lr:          float = 3e-6    # Learning rate for the DA-V2 depth head
    weight_decay:   float = 1e-4
    clip_grad_norm: float | None = 1.0  # Gradient clipping

    # Target normalisation
    normalize_target:     bool  = True
    target_norm_min:      float = 4.0  
    target_norm_max:      float = 17.0  

    # LR scheduler
    use_lr_scheduler:   bool = True
    warmup_fraction:    float = 0.05  

    # Derived (do not change) 
    def paths(self) -> RasterPaths:
        return RasterPaths(
            summer=self.summer_path,
            winter=self.winter_path,
            ahn=self.ahn_path,
            cir=self.cir_path,
            vegetation=self.vegetation_path,
        )

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def patch_index_dir(self) -> Path:
        return self.output_dir / "patch_index"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_float(value: float) -> str:
    """Convert a float to a filename-friendly string (3.5 → '3p5')."""
    return str(value).replace(".", "p")


def patch_index_path(config: SimpleDAV2Config) -> Path:
    """Generate the path for cached patch coordinates.

    data_year is in the name so a 2022 cache is never accidentally
    reused for a 2024 run (or vice versa).
    """
    return config.patch_index_dir / (
        f"clean_dav2_yr{config.data_year}_"
        f"ps{config.patch_size}_st{config.stride}_"
        f"h{_safe_float(config.min_tree_height)}-{_safe_float(config.max_tree_height)}_"
        f"valid{_safe_float(config.min_valid_ratio)}_"
        f"veg{int(config.use_vegetation_mask)}_"
        f"vthr{_safe_float(config.vegetation_threshold)}_"
        f"ndvimask{int(config.use_ndvi_mask)}_"
        f"ndvithr{_safe_float(config.ndvi_mask_threshold)}_"
        f"train{_safe_float(config.train_ratio)}.npz"
    )


def legacy_patch_index_path(config: SimpleDAV2Config) -> Path:
    return config.patch_index_dir / (
        f"dav2_ps{config.patch_size}_st{config.stride}_"
        f"h{_safe_float(config.min_tree_height)}-{_safe_float(config.max_tree_height)}_"
        f"valid{_safe_float(config.min_valid_ratio)}_train{_safe_float(config.train_ratio)}.npz"
    )


def _load_patch_cache(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    payload = np.load(path)
    train = [tuple(map(int, row)) for row in payload["train_coords"]]
    val   = [tuple(map(int, row)) for row in payload["val_coords"]]
    # Old caches (before the test-set split) have no test_coords key
    test  = [tuple(map(int, row)) for row in payload["test_coords"]] if "test_coords" in payload else []
    print(f"Patch index loaded: {path}")
    print(f"  Train: {len(train)} patches  |  Val: {len(val)} patches  |  Test: {len(test)} patches")
    return train, val, test


def _patch_has_enough_valid_pixels(
    ahn_src: rasterio.io.DatasetReader,
    veg_src: rasterio.io.DatasetReader | None,
    cir_src: rasterio.io.DatasetReader | None,
    x: int,
    y: int,
    config: SimpleDAV2Config,
) -> bool:
    from src.dataset import ndvi_from_cir, scale_raster_channels
    window = Window(x, y, config.patch_size, config.patch_size)
    ahn_patch = ahn_src.read(1, window=window).astype(np.float32)

    vegetation_patch = None
    if config.use_ndvi_mask and cir_src is not None:
        cir = scale_raster_channels(
            cir_src.read([1, 2, 3], window=window).astype(np.float32)
        )
        ndvi = ndvi_from_cir(cir, nir_band=config.cir_nir_band - 1, red_band=config.cir_red_band - 1)
        vegetation_patch = (ndvi > config.ndvi_mask_threshold).astype(np.float32)
    elif veg_src is not None:
        vegetation_patch = read_single_band_matching_window(
            reference_src=ahn_src,
            target_src=veg_src,
            reference_window=window,
            out_shape=(config.patch_size, config.patch_size),
            fill_value=0.0,
        ).astype(np.float32)

    mask = build_tree_mask(
        ahn_patch,
        min_height=config.min_tree_height,
        max_height=config.max_tree_height,
        vegetation_array=vegetation_patch,
        vegetation_threshold=config.vegetation_threshold,
    )
    return float(mask.mean()) >= config.min_valid_ratio


def build_or_load_patch_coords(
    config: SimpleDAV2Config,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build or load the list of patch coordinates.

    First time: scans the whole raster and saves the result as a cache.
    After that: loads the cache (much faster).
    """
    config.patch_index_dir.mkdir(parents=True, exist_ok=True)
    cache_path = patch_index_path(config)

    if cache_path.exists():
        return _load_patch_cache(cache_path)

    legacy_path = legacy_patch_index_path(config)

    if not config.use_vegetation_mask and legacy_path.exists():
        train, val, *_rest = (*_load_patch_cache(legacy_path), [])
        test: list[tuple[int, int]] = []
        np.savez_compressed(
            cache_path,
            train_coords=np.asarray(train, dtype=np.int64),
            val_coords=np.asarray(val,   dtype=np.int64),
            test_coords=np.asarray(test,  dtype=np.int64),
        )
        return train, val, test

    if config.use_vegetation_mask and legacy_path.exists():
        legacy_train, legacy_val, *_ = (*_load_patch_cache(legacy_path), [])
        if config.vegetation_path is None:
            raise ValueError("vegetation_path is required when use_vegetation_mask=True.")

        with rasterio.open(config.ahn_path) as ahn_src, rasterio.open(config.vegetation_path) as veg_src:
            train = [
                (x, y) for x, y in legacy_train
                if _patch_has_enough_valid_pixels(ahn_src, veg_src, None, x, y, config)
            ]
            val = [
                (x, y) for x, y in legacy_val
                if _patch_has_enough_valid_pixels(ahn_src, veg_src, None, x, y, config)
            ]

        test: list[tuple[int, int]] = []
        np.savez_compressed(
            cache_path,
            train_coords=np.asarray(train, dtype=np.int64),
            val_coords=np.asarray(val,   dtype=np.int64),
            test_coords=np.asarray(test,  dtype=np.int64),
        )
        print(f"Vegetation-filtered patch index saved: {cache_path}")
        print(f"  Train: {len(train)} patches  |  Val: {len(val)} patches  |  Test: 0 (no legacy test set)")
        return train, val, test

    # Build completely from scratch
    coords: list[tuple[int, int]] = []
    with rasterio.open(config.ahn_path) as ahn_src:
        width = ahn_src.width
        height = ahn_src.height
        if config.use_vegetation_mask and config.vegetation_path is None:
            raise ValueError("vegetation_path is required when use_vegetation_mask=True.")
        if config.use_ndvi_mask and config.cir_path is None:
            raise ValueError("cir_path is required when use_ndvi_mask=True.")

        veg_src = rasterio.open(config.vegetation_path) if config.use_vegetation_mask else None
        cir_src = rasterio.open(config.cir_path)        if config.use_ndvi_mask        else None
        try:
            for y in range(0, height - config.patch_size + 1, config.stride):
                for x in range(0, width - config.patch_size + 1, config.stride):
                    if _patch_has_enough_valid_pixels(ahn_src, veg_src, cir_src, x, y, config):
                        coords.append((x, y))
        finally:
            if veg_src is not None:
                veg_src.close()
            if cir_src is not None:
                cir_src.close()

    train, val, test = spatial_train_val_test_split(
        coords, width,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    np.savez_compressed(
        cache_path,
        train_coords=np.asarray(train, dtype=np.int64),
        val_coords=np.asarray(val,   dtype=np.int64),
        test_coords=np.asarray(test,  dtype=np.int64),
    )
    print(f"Patch index saved: {cache_path}")
    print(f"  Total: {len(coords)} patches  |  Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    return train, val, test


def limit_coords(
    coords: list[tuple[int, int]],
    max_count: int | None,
    seed: int,
) -> list[tuple[int, int]]:
    """Limit the number of patches to max_count (random sample)."""
    if max_count is None or max_count >= len(coords):
        return list(coords)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(coords), size=max_count, replace=False)
    return [coords[int(i)] for i in selected]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def make_loaders(config: SimpleDAV2Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create the train, val and test DataLoaders.

    Checks raster alignment, builds/loads the patch index, and creates
    the PyTorch DataLoaders. Augmentation is only on for the train loader.
    The test loader is for the very final evaluation after training.
    """
    check_raster_alignment(config.paths())
    train_coords, val_coords, test_coords = build_or_load_patch_coords(config)

    train_coords = limit_coords(train_coords, config.max_train_patches, config.seed)
    val_coords   = limit_coords(val_coords,   config.max_val_patches,   config.seed + 1)
    test_coords  = limit_coords(test_coords,  config.max_test_patches,  config.seed + 2)

    shared_kwargs = dict(
        paths=config.paths(),
        patch_size=config.patch_size,
        min_height=config.min_tree_height,
        max_height=config.max_tree_height,
        smooth_target_sigma=config.smooth_target_sigma,
        use_vegetation_mask=config.use_vegetation_mask,
        vegetation_threshold=config.vegetation_threshold,
        use_ndvi_mask=config.use_ndvi_mask,
        ndvi_mask_threshold=config.ndvi_mask_threshold,
        use_cir=config.use_cir,
        use_ndvi=config.use_ndvi,
        cir_nir_band=config.cir_nir_band,
        cir_red_band=config.cir_red_band,
        mask_close_radius=config.mask_close_radius,
    )

    train_ds = SiameseTreeDataset(coords=train_coords, augment=config.augment_train, **shared_kwargs)
    val_ds   = SiameseTreeDataset(coords=val_coords,   augment=False, **shared_kwargs)
    test_ds  = SiameseTreeDataset(coords=test_coords,  augment=False, **shared_kwargs)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = _loader(train_ds, shuffle=True)
    val_loader   = _loader(val_ds,   shuffle=False)
    test_loader  = _loader(test_ds,  shuffle=False)

    print(f"Dataset ready  →  train: {len(train_ds)} patches  |  val: {len(val_ds)} patches  |  test: {len(test_ds)} patches")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model & optimizer
# ---------------------------------------------------------------------------

def make_model(config: SimpleDAV2Config, device: torch.device | None = None) -> torch.nn.Module:
    """Load the Siamese DA-V2 height model (with fresh DA-V2 weights)."""
    device = device or get_best_device()
    return load_depth_anything_v2_siamese_height_net(
        repo_dir=config.repo_dir,
        checkpoint=checkpoint_path(config.repo_dir, config.encoder),
        encoder=config.encoder,
        device=device,
        share_depth_anything_weights=config.share_depth_anything_weights,
        train_dino_backbone=config.train_dino_backbone,
        train_depth_head=config.train_da_depth_head,
        depth_input_size=config.depth_input_size,
        head_channels=config.head_channels,
        include_rgb_in_head=config.include_rgb_in_head,
        include_cir_in_head=config.use_cir,
        include_ndvi_in_head=config.use_ndvi,
        cir_channels=3,
    )


def load_model_from_checkpoint(
    saved_checkpoint: Path,
    config: SimpleDAV2Config,
    device: torch.device | None = None,
) -> torch.nn.Module:
    device = device or get_best_device()

    # First build the architecture
    model = make_model(config, device=device)

    # Load the saved weights
    checkpoint = torch.load(saved_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    history = checkpoint.get("history", [])
    if history:
        best = min(history, key=lambda r: r["val_mae"])
        print(f"Checkpoint loaded: {saved_checkpoint.name}")
        print(f"  Best val MAE: {best['val_mae']:.3f} m  (epoch {best['epoch']})")
        print(f"  Last epoch: {history[-1]['epoch']}  from saved history")
    else:
        print(f"Checkpoint loaded: {saved_checkpoint.name}")

    return model


def make_optimizer(model: torch.nn.Module, config: SimpleDAV2Config) -> torch.optim.Optimizer:
    """Create an AdamW optimizer with two learning rates:
      - height_head: high (5e-4) — this part is learned from scratch
      - DA-V2 depth head: low (3e-6) — fine-tune on pretrained weights
    """
    head_params = [p for p in model.height_head.parameters() if p.requires_grad]
    da_params   = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("height_head.")
    ]

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": config.head_lr})
    if da_params:
        groups.append({"params": da_params,   "lr": config.da_lr})
    if not groups:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check train_da_depth_head and train_dino_backbone."
        )
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def normalize_target(target: torch.Tensor, config: SimpleDAV2Config) -> torch.Tensor:
    t_min = config.target_norm_min
    t_max = config.target_norm_max
    return (target - t_min) / (t_max - t_min)


def denormalize_prediction(prediction: torch.Tensor, config: SimpleDAV2Config) -> torch.Tensor:
    """Convert a [0–1] prediction back to metres."""
    t_min = config.target_norm_min
    t_max = config.target_norm_max
    return prediction * (t_max - t_min) + t_min


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: SimpleDAV2Config,
) -> torch.Tensor:

    valid = mask > 0.5

    # Normalise the target to [0–1] if that is enabled
    target_for_loss = normalize_target(target, config) if config.normalize_target else target
    pred_for_loss   = prediction

    pred_valid   = pred_for_loss[valid]
    target_valid = target_for_loss[valid]

    return F.mse_loss(pred_valid, target_valid)


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: SimpleDAV2Config,
) -> dict[str, torch.Tensor]:
    """Move the relevant batch items to the correct device."""
    inputs = {
        "summer": batch["summer"].to(device, non_blocking=True),
        "winter": batch["winter"].to(device, non_blocking=True),
    }
    if config.use_cir:
        inputs["cir"]  = batch["cir"].to(device, non_blocking=True)
    if config.use_ndvi:
        inputs["ndvi"] = batch["ndvi"].to(device, non_blocking=True)
    return inputs


# ---------------------------------------------------------------------------
# Train or validate one epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: SimpleDAV2Config,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    
    is_training = optimizer is not None
    model.train(is_training)

    total_loss    = 0.0
    total_abs_err = 0.0
    total_sq_err  = 0.0
    total_pixels  = 0.0
    t_start = time.time()

    for batch in loader:
        inputs = _batch_to_device(batch, device, config)
        target = batch["target"].to(device, non_blocking=True)
        mask   = batch["mask"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**inputs)
            loss = compute_loss(prediction, target, mask, config)
            loss.backward()
            if config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                prediction = model(**inputs)
                loss = compute_loss(prediction, target, mask, config)

        # Metric accumulation, always report in metres
        valid = mask > 0.5
        pred_meters   = denormalize_prediction(prediction.detach(), config) if config.normalize_target else prediction.detach()
        diff_meters   = pred_meters[valid] - target[valid]
        total_loss    += float(loss.item())
        total_abs_err += float(torch.abs(diff_meters).sum().item())
        total_sq_err  += float((diff_meters ** 2).sum().item())
        total_pixels  += float(valid.sum().item())

    n_pixels = max(total_pixels, 1.0)
    n_batches = max(len(loader), 1)
    return {
        "loss":    total_loss / n_batches,
        "mae":     total_abs_err / n_pixels,
        "rmse":    (total_sq_err / n_pixels) ** 0.5,
        "pixels":  total_pixels,
        "seconds": time.time() - t_start,
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SimpleDAV2Config,
    device: torch.device | None = None,
) -> tuple[list[dict[str, float]], Path]:
    """Train the model for config.epochs epochs.

    - Saves the best model (lowest val MAE).
    - Uses a linear warmup + linear decay LR scheduler if use_lr_scheduler=True.
    - Prints the metrics every epoch.

    Returns
    -------
    history : list of metric dicts per epoch
    best_path : path to the saved best checkpoint
    """
    device = device or get_best_device()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = make_optimizer(model, config)

    scheduler = None
    if config.use_lr_scheduler:
        total_steps  = config.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * config.warmup_fraction))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Give the checkpoint a descriptive name
    # data_year is included so a 2022 run never overwrites a 2024 checkpoint
    norm_tag = "norm1" if config.normalize_target else "norm0"
    name = (
        f"clean_dav2_siamese_{config.encoder}_"
        f"yr{config.data_year}_"
        f"h{_safe_float(config.min_tree_height)}-{_safe_float(config.max_tree_height)}_"
        f"head{config.head_channels}_dahead{int(config.train_da_depth_head)}_"
        f"veg{int(config.use_vegetation_mask)}_"
        f"cir{int(config.use_cir)}_ndvi{int(config.use_ndvi)}_"
        f"{norm_tag}_mse"
    )
    best_path = config.checkpoint_dir / f"{name}_best.pt"
    history: list[dict[str, float]] = []
    best_val_mae = float("inf")

    norm_label = f"target [0-1] (min={config.target_norm_min}, max={config.target_norm_max})" if config.normalize_target else "raw metres"
    print(f"\nStarting training: {config.epochs} epochs  |  encoder: {config.encoder}  |  device: {device}")
    print(f"Loss: MSE (normalised)  |  Target: {norm_label}  |  Augmentation: {config.augment_train}")
    print(f"LR scheduler: warmup {config.warmup_fraction*100:.0f}% + linear decay  |  smooth σ={config.smooth_target_sigma}")
    print("-" * 70)

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, config, optimizer=optimizer,
                                  scheduler=scheduler)
        val_metrics   = run_epoch(model, val_loader,   device, config, optimizer=None)

        row = {
            "epoch":      epoch,
            "train_mae":  train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_mae":    val_metrics["mae"],
            "val_rmse":   val_metrics["rmse"],
            "seconds":    train_metrics["seconds"] + val_metrics["seconds"],
            "lr":         optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        marker = "  ← best" if row["val_mae"] < best_val_mae else ""
        print(
            f"Epoch {epoch:02d}/{config.epochs}  "
            f"train MAE={row['train_mae']:.3f}m  RMSE={row['train_rmse']:.3f}m  |  "
            f"val MAE={row['val_mae']:.3f}m  RMSE={row['val_rmse']:.3f}m  "
            f"({row['seconds']:.0f}s){marker}"
        )

        if row["val_mae"] < best_val_mae:
            best_val_mae = row["val_mae"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "config": config.__dict__,
                },
                best_path,
            )
            print(f"  → checkpoint saved: {best_path.name}")

    print("-" * 70)
    print(f"Done! Best val MAE: {best_val_mae:.3f} m  |  checkpoint: {best_path}")
    return history, best_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_history(history: list[dict[str, float]], save_path: Path | None = None) -> None:
    """Show MAE and RMSE curves for train and validation per epoch."""
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, [row["train_mae"]  for row in history], label="train")
    axes[0].plot(epochs, [row["val_mae"]    for row in history], label="validation")
    axes[0].set_title("Masked MAE on AHN tree pixels")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error (m)")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_rmse"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_rmse"]   for row in history], label="validation")
    axes[1].set_title("Masked RMSE on AHN tree pixels")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (m)")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {save_path}")
    plt.show()


def _refine_display_mask(
    mask: np.ndarray,
    close_radius: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:

    display_mask = np.asarray(mask, dtype=bool)

    if close_radius > 0:
        from scipy.ndimage import binary_closing

        yy, xx = np.ogrid[-close_radius : close_radius + 1, -close_radius : close_radius + 1]
        structure = (xx * xx + yy * yy) <= close_radius * close_radius
        display_mask = binary_closing(display_mask, structure=structure)

    if fill_holes:
        from scipy.ndimage import binary_fill_holes

        display_mask = binary_fill_holes(display_mask)

    return np.asarray(display_mask, dtype=bool)


def _plot_masked_crowns(
    ax,
    base_map: np.ndarray,
    value_map: np.ndarray,
    crown_mask: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
):
    """Show crown heights on a soft background so crown shapes stay readable."""
    background = np.where(np.isfinite(base_map), base_map, np.nan)
    ax.imshow(background, cmap="Greys", alpha=0.20)
    overlay = ax.imshow(
        np.ma.masked_where(~crown_mask, value_map),
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
    )
    if crown_mask.any():
        ax.contour(crown_mask.astype(float), levels=[0.5], colors="black", linewidths=0.35, alpha=0.45)
    ax.set_title(title)
    return overlay


@torch.no_grad()
def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: SimpleDAV2Config,
    device: torch.device,
    label: str,
) -> dict[str, float | np.ndarray]:

    model.eval()
    all_pred:  list[np.ndarray] = []
    all_truth: list[np.ndarray] = []

    for batch in loader:
        inputs = _batch_to_device(batch, device, config)
        pred = model(**inputs)
        if config.normalize_target:
            pred = denormalize_prediction(pred, config)

        pred_np   = pred.squeeze(1).cpu().numpy()
        target_np = batch["target"].squeeze(1).numpy()
        mask_np   = batch["mask"].squeeze(1).numpy().astype(bool)

        for p, t, m in zip(pred_np, target_np, mask_np):
            if not m.any():
                continue
            all_pred.append(p[m])
            all_truth.append(t[m])

    pred_vec  = np.concatenate(all_pred)
    truth_vec = np.concatenate(all_truth)
    residuals = pred_vec - truth_vec

    mae  = float(np.abs(residuals).mean())
    rmse = float(np.sqrt((residuals ** 2).mean()))
    bias = float(residuals.mean())
    p95  = float(np.percentile(np.abs(residuals), 95))

    # R² over all pixels together 
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((truth_vec - truth_vec.mean()) ** 2).sum())
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(
        f"{label} ({len(pred_vec):,} pixels)  →  "
        f"MAE = {mae:.3f} m  |  RMSE = {rmse:.3f} m  |  "
        f"R² = {r2:.3f}  |  bias = {bias:+.3f} m  |  p95 = {p95:.3f} m"
    )
    return {
        "mae":   mae,
        "rmse":  rmse,
        "r2":    r2,
        "bias":  bias,
        "p95":   p95,
        "pred":  pred_vec,
        "truth": truth_vec,
    }


@torch.no_grad()
def evaluate_val(
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: SimpleDAV2Config,
    device: torch.device,
) -> dict[str, float]:
    
    return _evaluate_loader(model, val_loader, config, device, label="Val set")


@torch.no_grad()
def evaluate_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    config: SimpleDAV2Config,
    device: torch.device,
) -> dict[str, float | np.ndarray]:

    return _evaluate_loader(model, test_loader, config, device, label="Test set")


def plot_evaluation(
    metrics: dict[str, float | np.ndarray],
    label: str = "Evaluation",
    max_scatter_points: int = 50_000,
    save_path: Path | None = None,
) -> None:

    import matplotlib.pyplot as plt

    pred_vec  = metrics["pred"]
    truth_vec = metrics["truth"]
    mae       = metrics["mae"]
    rmse      = metrics["rmse"]
    r2        = metrics["r2"]
    bias      = metrics["bias"]

    if len(pred_vec) > max_scatter_points:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(pred_vec), size=max_scatter_points, replace=False)
        pred_plot  = pred_vec[idx]
        truth_plot = truth_vec[idx]
    else:
        pred_plot, truth_plot = pred_vec, truth_vec

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(label, fontsize=13, fontweight="bold")

    # Left: scatter pred vs truth
    ax = axes[0]
    h  = ax.hexbin(truth_plot, pred_plot, gridsize=80, cmap="YlOrRd", mincnt=1)
    plt.colorbar(h, ax=ax, label="Number of pixels")

    lim_lo, lim_hi = 4, 17
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, label="ideal (y = x)")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Ground truth height (m)")
    ax.set_ylabel("Predicted height (m)")
    ax.set_title("Scatter: prediction vs ground truth")
    ax.legend(fontsize=9, loc="lower right")
    ax.text(
        0.05, 0.95,
        f"R²   = {r2:.3f}\nMAE  = {mae:.3f} m\nRMSE = {rmse:.3f} m\nbias = {bias:+.3f} m",
        transform=ax.transAxes, verticalalignment="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Right: residual plot (pred − truth vs truth)
    residuals_plot = pred_plot - truth_plot
    ax2 = axes[1]
    h2  = ax2.hexbin(truth_plot, residuals_plot, gridsize=80, cmap="coolwarm", mincnt=1)
    plt.colorbar(h2, ax=ax2, label="Number of pixels")
    ax2.axhline(0, color="black", linewidth=1.2, linestyle="--", label="residual = 0")
    ax2.set_xlabel("Ground truth height (m)")
    ax2.set_ylabel("Residual: prediction − truth (m)")
    ax2.set_title("Residual plot")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {save_path}")
    plt.show()


def make_temporal_test_loader(
    train_config: SimpleDAV2Config,
    summer_path: Path,
    cir_path: Path | None = None,
    max_patches: int | None = None,
) -> DataLoader:

    # Load the patch coordinates from the original training
    _, _, test_coords = build_or_load_patch_coords(train_config)
    test_coords = limit_coords(test_coords, max_patches, train_config.seed + 2)

    # Replace only the summer image and CIR, winter and AHN stay the same
    orig = train_config.paths()
    new_paths = RasterPaths(
        summer     = Path(summer_path),
        winter     = orig.winter,
        ahn        = orig.ahn,
        cir        = Path(cir_path) if cir_path is not None else orig.cir,
        vegetation = orig.vegetation,
    )

    ds = SiameseTreeDataset(
        coords              = test_coords,
        paths               = new_paths,
        patch_size          = train_config.patch_size,
        min_height          = train_config.min_tree_height,
        max_height          = train_config.max_tree_height,
        smooth_target_sigma = train_config.smooth_target_sigma,
        use_vegetation_mask = train_config.use_vegetation_mask,
        vegetation_threshold= train_config.vegetation_threshold,
        use_ndvi_mask       = train_config.use_ndvi_mask,
        ndvi_mask_threshold = train_config.ndvi_mask_threshold,
        use_cir             = train_config.use_cir,
        use_ndvi            = train_config.use_ndvi,
        cir_nir_band        = train_config.cir_nir_band,
        cir_red_band        = train_config.cir_red_band,
        mask_close_radius   = train_config.mask_close_radius,
        augment             = False,
    )

    loader = DataLoader(
        ds,
        batch_size  = train_config.batch_size,
        shuffle     = False,
        num_workers = train_config.num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    print(f"Temporal test loader ready  →  {len(ds)} patches  |  summer image: {Path(summer_path).name}")
    return loader


def make_generalization_loader(
    summer_path: Path,
    winter_path: Path,
    ahn_path: Path,
    cir_path: Path,
    ref_config: SimpleDAV2Config,
    max_patches: int | None = None,
) -> DataLoader:

    city_paths = RasterPaths(
        summer     = Path(summer_path),
        winter     = Path(winter_path),
        ahn        = Path(ahn_path),
        cir        = Path(cir_path),
        vegetation = None,
    )

    check_raster_alignment(city_paths)

    # Build the patch index over the full raster 
    coords: list[tuple[int, int]] = []
    with rasterio.open(city_paths.ahn) as ahn_src:
        width  = ahn_src.width
        height = ahn_src.height
        cir_src = rasterio.open(city_paths.cir) if ref_config.use_ndvi_mask else None
        try:
            for y in range(0, height - ref_config.patch_size + 1, ref_config.stride):
                for x in range(0, width - ref_config.patch_size + 1, ref_config.stride):
                    if _patch_has_enough_valid_pixels(ahn_src, None, cir_src, x, y, ref_config):
                        coords.append((x, y))
        finally:
            if cir_src is not None:
                cir_src.close()

    coords = limit_coords(coords, max_patches, ref_config.seed)
    print(f"Generalization patch index: {len(coords)} patches  |  city: {Path(ahn_path).stem}")

    ds = SiameseTreeDataset(
        coords               = coords,
        paths                = city_paths,
        patch_size           = ref_config.patch_size,
        min_height           = ref_config.min_tree_height,
        max_height           = ref_config.max_tree_height,
        smooth_target_sigma  = ref_config.smooth_target_sigma,
        use_vegetation_mask  = False,
        vegetation_threshold = ref_config.vegetation_threshold,
        use_ndvi_mask        = ref_config.use_ndvi_mask,
        ndvi_mask_threshold  = ref_config.ndvi_mask_threshold,
        use_cir              = ref_config.use_cir,
        use_ndvi             = ref_config.use_ndvi,
        cir_nir_band         = ref_config.cir_nir_band,
        cir_red_band         = ref_config.cir_red_band,
        mask_close_radius    = ref_config.mask_close_radius,
        augment              = False,
    )

    return DataLoader(
        ds,
        batch_size  = ref_config.batch_size,
        shuffle     = False,
        num_workers = ref_config.num_workers,
        pin_memory  = torch.cuda.is_available(),
    )


@torch.no_grad()
def show_prediction(
    model: torch.nn.Module,
    dataset: SiameseTreeDataset,
    config: SimpleDAV2Config,
    device: torch.device,
    idx: int = 0,
) -> None:
    """Visualise one validation patch: summer, winter, AHN, prediction, error."""
    import matplotlib.pyplot as plt

    model.eval()
    item = dataset[idx]

    batch = {
        "summer": item["summer"].unsqueeze(0),
        "winter": item["winter"].unsqueeze(0),
    }
    if config.use_cir:
        batch["cir"]  = item["cir"].unsqueeze(0)
    if config.use_ndvi:
        batch["ndvi"] = item["ndvi"].unsqueeze(0)

    pred_tensor = model(**_batch_to_device(batch, device, config))

    if config.normalize_target:
        pred_tensor = denormalize_prediction(pred_tensor, config)
    prediction = pred_tensor.squeeze().cpu().numpy()
    target     = item["target"].squeeze().numpy()
    mask       = item["mask"].squeeze().numpy().astype(bool)
    display_mask = _refine_display_mask(
        mask,
        close_radius=config.display_mask_close_radius,
        fill_holes=config.display_mask_fill_holes,
    )

    valid_target = target[mask]
    valid_pred   = prediction[mask]
    mae  = float(np.mean(np.abs(valid_pred - valid_target)))
    rmse = float(np.sqrt(np.mean((valid_pred - valid_target) ** 2)))
    vmin = float(min(valid_target.min(), valid_pred.min()))
    vmax = float(max(valid_target.max(), valid_pred.max()))

    _, axes = plt.subplots(1, 6, figsize=(26, 4))
    axes[0].imshow(np.moveaxis(item["summer"].numpy(), 0, -1))
    axes[0].set_title("Summer RGB")
    axes[1].imshow(np.moveaxis(item["winter"].numpy(), 0, -1))
    axes[1].set_title("Winter RGB")

    im_full = axes[2].imshow(target, cmap="terrain")
    axes[2].set_title("AHN full (DSM)\n(own scale: 0 m – max)")
    plt.colorbar(im_full, ax=axes[2], fraction=0.046, pad=0.04, label="Height (m)")

    im_tree = _plot_masked_crowns(
        axes[3],
        base_map=target,
        value_map=target,
        crown_mask=display_mask,
        title="AHN tree crowns\n(display mask)",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im_tree, ax=axes[3], fraction=0.046, pad=0.04, label="Height (m)")

    im_pred = axes[4].imshow(np.where(display_mask, prediction, np.nan), cmap="terrain", vmin=vmin, vmax=vmax)
    axes[4].set_title("Prediction (m)")
    plt.colorbar(im_pred, ax=axes[4], fraction=0.046, pad=0.04, label="Height (m)")

    im_err = axes[5].imshow(np.where(display_mask, prediction - target, np.nan), cmap="coolwarm", vmin=-3, vmax=3)
    axes[5].set_title("Error: pred − AHN (m)")
    plt.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04, label="Error (m)")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Patch {idx}  |  MAE = {mae:.2f} m  |  RMSE = {rmse:.2f} m")
    plt.tight_layout()
    plt.show()

    print(f"AHN range  : {valid_target.min():.1f} – {valid_target.max():.1f} m")
    print(f"Pred range : {valid_pred.min():.1f} – {valid_pred.max():.1f} m")
    print(f"MAE={mae:.3f} m  |  RMSE={rmse:.3f} m")
