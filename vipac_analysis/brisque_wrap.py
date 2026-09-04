"""BRISQUE wrapper for VIPAC quality evaluation.

Uses PIQ's BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
which computes hand-crafted spatial features and regresses quality via SVM.
Scores range 0-100 (lower = better quality).

No model parameters required (runs on CPU).
"""

import numpy as np
import torch


def predict_quality(img: np.ndarray) -> float:
    """Compute BRISQUE quality score for a single image.

    Parameters
    ----------
    img : np.ndarray
        RGB image (H, W, 3, uint8).

    Returns
    -------
    float
        Quality score in 0-100 range (lower = better).
    """
    from piq import brisque

    img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return float(brisque(img_t).item())
