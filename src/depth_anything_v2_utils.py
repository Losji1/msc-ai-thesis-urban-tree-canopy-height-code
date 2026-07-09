from __future__ import annotations
import sys
from pathlib import Path
import torch


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_depth_anything_v2_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "depth_anything_v2" / "dpt.py").exists():
        raise FileNotFoundError(
            f"Could not find official Depth Anything V2 repo at {repo_dir}. "
            "Clone https://github.com/DepthAnything/Depth-Anything-V2 first."
        )

    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def checkpoint_path(repo_dir: Path, encoder: str) -> Path:
    return repo_dir / "checkpoints" / f"depth_anything_v2_{encoder}.pth"


def load_official_depth_anything_v2(
    repo_dir: Path,
    checkpoint: Path,
    encoder: str,
    device: torch.device,
) -> torch.nn.Module:
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder {encoder!r}. Choose from {sorted(MODEL_CONFIGS)}.")

    add_depth_anything_v2_repo_to_path(repo_dir)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model