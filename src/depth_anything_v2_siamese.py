from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.depth_anything_v2_utils import load_official_depth_anything_v2


# ImageNet normalisation that Depth Anything V2 expects
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_depth_anything_v2(images: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensors in [0,1] to ImageNet-normalised values."""
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std  = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set all parameters of a module to trainable or frozen."""
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def set_depth_anything_trainability(
    model: nn.Module,
    train_dino_backbone: bool = False,
    train_depth_head: bool = False,
) -> None:
    
    set_requires_grad(model, False)
    if train_dino_backbone and hasattr(model, "pretrained"):
        set_requires_grad(model.pretrained, True)
    if train_depth_head and hasattr(model, "depth_head"):
        set_requires_grad(model.depth_head, True)


def standardize_depth_per_patch(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = depth.mean(dim=(2, 3), keepdim=True)
    std  = depth.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (depth - mean) / std


# ---------------------------------------------------------------------------
# The height head: translates combined features into height in metres
# ---------------------------------------------------------------------------

class DepthFusionHead(nn.Module):
    """
    Input: stacked channels (depths + optional RGB/CIR/NDVI).
    Output: (1, H, W) map with predicted height in metres.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        nonnegative_output: bool = True,
    ) -> None:
        super().__init__()
        self.nonnegative_output = nonnegative_output
        mid = hidden_channels // 2

        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.GELU(),
            # Layer 2
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.GELU(),
            # Layer 3
            nn.Conv2d(hidden_channels, mid, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid),
            nn.GELU(),
            # Layer 4
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        prediction = self.net(features)
        if self.nonnegative_output:
            # Softplus for always positive output
            prediction = F.softplus(prediction)
        return prediction


# ---------------------------------------------------------------------------
# The full Siamese model
# ---------------------------------------------------------------------------

class DepthAnythingV2SiameseHeightNet(nn.Module):
    """
    Step 1: both images through (shared) DA-V2 → depth maps.
    Step 2: combine depth maps + optionally add CIR/NDVI.
    Step 3: DepthFusionHead predicts height in metres.
    """

    def __init__(
        self,
        summer_model: nn.Module,
        winter_model: nn.Module | None = None,
        depth_input_size: int = 252,
        head_channels: int = 64,
        include_rgb_in_head: bool = True,
        include_cir_in_head: bool = False,
        include_ndvi_in_head: bool = False,
        cir_channels: int = 3,
        standardize_depth: bool = True,
        nonnegative_output: bool = True,
        keep_frozen_depth_anything_eval: bool = True,
    ) -> None:
        super().__init__()

        self.summer_model = summer_model
        self.winter_model = winter_model if winter_model is not None else summer_model
        self.shared_depth_anything = (self.summer_model is self.winter_model)

        self.depth_input_size = depth_input_size
        self.include_rgb_in_head = include_rgb_in_head
        self.include_cir_in_head = include_cir_in_head
        self.include_ndvi_in_head = include_ndvi_in_head
        self.standardize_depth = standardize_depth
        self.keep_frozen_depth_anything_eval = keep_frozen_depth_anything_eval

        fusion_channels = 4
        if include_rgb_in_head:
            fusion_channels += 6 
        if include_cir_in_head:
            fusion_channels += cir_channels
        if include_ndvi_in_head:
            fusion_channels += 1

        self.height_head = DepthFusionHead(
            in_channels=fusion_channels,
            hidden_channels=head_channels,
            nonnegative_output=nonnegative_output,
        )

    def train(self, mode: bool = True) -> "DepthAnythingV2SiameseHeightNet":
        """Keep frozen DA-V2 parts in eval mode at all times."""
        super().train(mode)
        if mode and self.keep_frozen_depth_anything_eval:
            self._set_frozen_parts_to_eval()
        return self

    def _set_frozen_parts_to_eval(self) -> None:
        """Put frozen submodules in eval() so BatchNorm behaves correctly."""
        for depth_model in self._get_depth_models():
            if not any(p.requires_grad for p in depth_model.parameters()):
                depth_model.eval()
                continue
            for part_name in ("pretrained", "depth_head"):
                part = getattr(depth_model, part_name, None)
                if part is not None and not any(p.requires_grad for p in part.parameters()):
                    part.eval()

    def _get_depth_models(self) -> list[nn.Module]:
        if self.shared_depth_anything:
            return [self.summer_model]
        return [self.summer_model, self.winter_model]

    def _run_depth_anything(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        original_size = images.shape[-2:]

        resized = F.interpolate(
            images,
            size=(self.depth_input_size, self.depth_input_size),
            mode="bilinear",
            align_corners=False,
        )
        normalized = normalize_for_depth_anything_v2(resized)

        model_is_frozen = not any(p.requires_grad for p in model.parameters())
        if model_is_frozen:
            with torch.no_grad():
                depth = model(normalized)
        else:
            depth = model(normalized)

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        # Back to the original patch size
        depth = F.interpolate(depth, size=original_size, mode="bilinear", align_corners=False)

        if self.standardize_depth:
            depth = standardize_depth_per_patch(depth)

        return depth

    def forward(
        self,
        summer: torch.Tensor,
        winter: torch.Tensor,
        cir: torch.Tensor | None = None,
        ndvi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        # Step 1: depth maps from DA-V2
        summer_depth = self._run_depth_anything(self.summer_model, summer)
        winter_depth = self._run_depth_anything(self.winter_model, winter)

        # Step 2: combine depth maps
        depth_mean = 0.5 * (summer_depth + winter_depth)
        depth_diff = summer_depth - winter_depth         

        features = [summer_depth, winter_depth, depth_mean, depth_diff]

        if self.include_rgb_in_head:
            features.extend([summer, winter])

        if self.include_cir_in_head:
            if cir is None:
                raise ValueError("CIR input is required but was not provided.")
            features.append(cir)

        if self.include_ndvi_in_head:
            if ndvi is None:
                raise ValueError("NDVI input is required but was not provided.")
            features.append(ndvi)

        # Step 3: stack all channels and predict height
        fused = torch.cat(features, dim=1)
        return self.height_head(fused)


# ---------------------------------------------------------------------------
# Factory function: load the full model
# ---------------------------------------------------------------------------

def load_depth_anything_v2_siamese_height_net(
    repo_dir: Path,
    checkpoint: Path,
    encoder: str,
    device: torch.device,
    share_depth_anything_weights: bool = True,
    train_dino_backbone: bool = False,
    train_depth_head: bool = False,
    depth_input_size: int = 252,
    head_channels: int = 64,
    include_rgb_in_head: bool = True,
    include_cir_in_head: bool = False,
    include_ndvi_in_head: bool = False,
    cir_channels: int = 3,
) -> DepthAnythingV2SiameseHeightNet:
    
    summer_model = load_official_depth_anything_v2(
        repo_dir=repo_dir,
        checkpoint=checkpoint,
        encoder=encoder,
        device=device,
    )
    set_depth_anything_trainability(
        summer_model,
        train_dino_backbone=train_dino_backbone,
        train_depth_head=train_depth_head,
    )

    winter_model = None
    if not share_depth_anything_weights:
        winter_model = load_official_depth_anything_v2(
            repo_dir=repo_dir,
            checkpoint=checkpoint,
            encoder=encoder,
            device=device,
        )
        set_depth_anything_trainability(
            winter_model,
            train_dino_backbone=train_dino_backbone,
            train_depth_head=train_depth_head,
        )

    model = DepthAnythingV2SiameseHeightNet(
        summer_model=summer_model,
        winter_model=winter_model,
        depth_input_size=depth_input_size,
        head_channels=head_channels,
        include_rgb_in_head=include_rgb_in_head,
        include_cir_in_head=include_cir_in_head,
        include_ndvi_in_head=include_ndvi_in_head,
        cir_channels=cir_channels,
    )
    return model.to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of parameters that are actually trained."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Total number of parameters (including frozen)."""
    return sum(p.numel() for p in model.parameters())
