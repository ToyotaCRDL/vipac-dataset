"""CLIPIQA wrapper for VIPAC quality evaluation.

Uses PIQ's CLIPIQA (CLIP-based Image Quality Assessment) which leverages
CLIP-ViT-B/16 features. The model compares image embeddings against
"Good photo" / "Bad photo" text anchors. Scores range 0-1 (higher = better).

Paper: https://arxiv.org/abs/2207.12396
"""

import warnings
from typing import Optional

import numpy as np
import torch


def build_model(device: Optional[str] = None) -> "CLIPIQA":
    """Build and initialize CLIPIQA model with pre-trained weights.

    Parameters
    ----------
    device : str or None
        Device to place the model on. Defaults to 'cuda' if available.

    Returns
    -------
    CLIPIQA model in eval mode.
    """
    from piq import CLIPIQA

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # PIQ's CLIPIQA expects input in [0, data_range]. Default data_range=1.0
    # expects [0,1] input. We set data_range=255.0 so we can pass uint8 values
    # directly (as float tensor) without manual normalization.
    # piq 0.8.0 calls torch.load without weights_only, which torch 2.5.x
    # flags with a FutureWarning. The loaded file is piq's official token
    # file from its GitHub release (trusted source), so suppress it here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*weights_only.*",
            module=r"piq\.utils\.common",
        )
        model = CLIPIQA(data_range=255.0)
    model.eval()

    # Move model buffers to device. The feature_extractor is moved inside
    # the forward() method by PIQ, so we only need to move the buffers.
    if device == "cuda":
        model = model.cuda()

    return model


def _get_device(model):
    """Get device from model buffers."""
    return model.default_mean.device


def predict_quality(model, img: np.ndarray) -> float:
    """Compute CLIPIQA quality score for a single image.

    Parameters
    ----------
    model : CLIPIQA
        Model returned by build_model().
    img : np.ndarray
        RGB image (H, W, 3, uint8).

    Returns
    -------
    float
        Quality score in 0-1 range (higher = better).
    """
    device = _get_device(model)
    img_t = (
        torch.from_numpy(img)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        score = model(img_t)
    return float(score.item())


def predict_batch(model, images: list[np.ndarray]) -> list[float]:
    """Compute CLIPIQA scores for a batch of images.

    Parameters
    ----------
    model : CLIPIQA
    images : list of np.ndarray (H, W, 3, uint8)

    Returns
    -------
    List[float] of quality scores (0-1 range).
    """
    device = _get_device(model)
    tensors = []
    for img in images:
        t = torch.from_numpy(img).float().permute(2, 0, 1)
        tensors.append(t)
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        scores = model(batch)
    return scores.squeeze(-1).tolist()
