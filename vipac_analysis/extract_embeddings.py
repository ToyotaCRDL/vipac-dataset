"""Extract CLIP-ViT-B/16 embeddings for all VIPAC images."""

import os
import warnings

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from vipac_analysis.config import all_image_ids, image_id_str


def _build_clip(device):
    """Load CLIP ViT-B/16 model and preprocessing function."""
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    model.eval()
    model.requires_grad_(False)
    return model, preprocess


def extract_embeddings(
    image_dir,
    metadata_path,
    output_path,
    index_path,
    batch_size=64,
    resume=False,
):
    """Extract 512-dim CLIP embeddings for all VIPAC images.

    Args:
        image_dir: Path to VIPAC/images/ directory.
        metadata_path: Path to metadata-images.csv.
        output_path: Output .npy file path for embeddings.
        index_path: Output CSV path for image index.
        batch_size: GPU batch size.
        resume: Skip if output already exists at correct size.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = _build_clip(device)

    img_ids = list(all_image_ids())
    total = len(img_ids)

    if resume and os.path.exists(output_path):
        existing = np.load(output_path)
        if existing.shape[0] == total:
            print(f"Resuming: embeddings already complete ({existing.shape})")
            metadata = pd.read_csv(metadata_path)
            index_df = metadata.rename(columns={
                "image-id": "image_id",
                "vehicle-id": "vehicle_id",
            })
            index_df[["image_id", "vehicle_id", "color", "attribute", "seed"]].to_csv(
                index_path, index=False
            )
            return

    embeddings = np.zeros((total, 512), dtype=np.float32)
    processed = 0
    batch_tensors = []

    with torch.no_grad():
        # cuDNN emits a benign UserWarning ("Plan failed with a cudnnException
        # ... CUDNN_STATUS_NOT_SUPPORTED") when its conv heuristics fall back
        # to another algorithm. It does not affect embeddings, so suppress it here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r".*cudnnException.*",
            )
            for idx, img_id in enumerate(tqdm(img_ids, desc="Extracting")):
                img_path = os.path.join(image_dir, image_id_str(img_id) + ".png")
                if not os.path.exists(img_path):
                    print(f"\nWarning: missing {img_path}")
                    continue

                img = Image.open(img_path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)

                if len(batch_tensors) == batch_size:
                    batch = torch.stack(batch_tensors).to(device)
                    try:
                        features = clip_model.encode_image(batch)
                    except torch.cuda.OutOfMemoryError:
                        batch_size = max(1, batch_size // 2)
                        print(f"\nOOM: reduced batch_size to {batch_size}")
                        # Process accumulated tensors in smaller batches
                        for b_start in range(0, len(batch_tensors), batch_size):
                            chunk_tensors = batch_tensors[b_start : b_start + batch_size]
                            chunk = torch.stack(chunk_tensors).to(device)
                            chunk_features = clip_model.encode_image(chunk)
                            chunk_start = processed + b_start
                            embeddings[chunk_start : chunk_start + chunk_features.shape[0]] = (
                                chunk_features.cpu().numpy()
                            )
                        processed += len(batch_tensors)
                        batch_tensors = []
                        continue

                    embeddings[processed : processed + batch_size] = features.cpu().numpy()
                    processed += batch_size
                    batch_tensors = []

            # Handle remaining images
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(device)
                features = clip_model.encode_image(batch)
                embeddings[processed : processed + len(batch_tensors)] = features.cpu().numpy()
                processed += len(batch_tensors)

    np.save(output_path, embeddings)
    print(f"\nSaved {processed}/{total} embeddings -> {output_path}")

    metadata = pd.read_csv(metadata_path)
    index_df = metadata.rename(columns={
        "image-id": "image_id",
        "vehicle-id": "vehicle_id",
    })
    index_df[["image_id", "vehicle_id", "color", "attribute", "seed"]].to_csv(
        index_path, index=False
    )
    print(f"Saved index -> {index_path}")
