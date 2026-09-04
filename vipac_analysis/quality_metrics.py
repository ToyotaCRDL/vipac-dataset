"""CPU-based image quality metrics: blur, noise, exposure, contrast, BRISQUE."""

import cv2
import numpy as np
from PIL import Image


def compute_all_metrics(img: np.ndarray, include_brisque: bool = False) -> dict:
    """Compute all CPU metrics for a single RGB image (H, W, 3, uint8)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = {
        "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise_sigma": _noise_sigma(gray),
        **_exposure_stats(gray),
        "contrast_rms": _rms_contrast(gray),
    }
    if include_brisque:
        result["brisque_quality"] = _brisque(img)
    return result


def load_cpu_metrics(image_path: str, include_brisque: bool = False) -> dict:
    """Load one image from disk and compute all CPU metrics."""
    img = np.array(Image.open(image_path).convert("RGB"))
    return compute_all_metrics(img, include_brisque=include_brisque)


def _noise_sigma(gray: np.ndarray) -> float:
    """Noise level: std of residual after Gaussian blur."""
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    residual = gray.astype(np.float64) - blurred.astype(np.float64)
    return float(np.std(residual))


def _exposure_stats(gray: np.ndarray) -> dict:
    """Exposure: mean brightness + histogram percentiles + peak bin."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    cum = np.cumsum(hist.astype(np.float64))
    total = cum[-1]
    percentiles = [
        (np.searchsorted(cum, total * p / 100, side="right") / 255.0 * 255.0)
        for p in (10, 50, 90)
    ]
    return {
        "exposure_mean": float(np.mean(gray)),
        "exposure_p10": percentiles[0],
        "exposure_p50": percentiles[1],
        "exposure_p90": percentiles[2],
        "exposure_peak_bin": int(np.argmax(hist)),
    }


def _rms_contrast(gray: np.ndarray) -> float:
    """RMS contrast: std / sqrt(mean^2) * 255."""
    g = gray.astype(np.float64)
    rms = np.sqrt(np.mean(g ** 2))
    if rms < 1e-6:
        return 0.0
    return float(np.std(g) / rms * 255.0)


def _brisque(img: np.ndarray) -> float:
    """BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator."""
    from piq import brisque
    import torch

    img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return float(brisque(img_t).item())
