"""Score all 120,000 images with trained RSS-CNN models."""

import argparse
import glob
import joblib
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from vipac_analysis.dataset import ImageDataset
from vipac_analysis.model import RSSCNN
from vipac_analysis.transforms import eval_transform


def load_models(model_dir):
    """Find all .pth model files in a directory."""
    pth_files = sorted(glob.glob(os.path.join(model_dir, '*.pth')))
    return pth_files


def score_model(model, image_loader, device):
    """Run inference on all images, return dict {img_id_6digit: score}."""
    model.eval()
    scores = {}
    with torch.no_grad():
        # cuDNN emits a benign UserWarning ("Plan failed with a cudnnException
        # ... CUDNN_STATUS_NOT_SUPPORTED") when its conv heuristics fall back
        # to another algorithm. It does not affect scores, so suppress it here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r".*cudnnException.*",
            )
            for batch_idx, (imgs, img_ids) in enumerate(image_loader):
                imgs = imgs.to(device)
                outputs = model(imgs, None)
                for i in range(outputs.shape[0]):
                    img_id = int(img_ids[i])
                    scores[f"{img_id:06d}"] = outputs[i][0].item()
                if (batch_idx + 1) % 50 == 0:
                    sys.stderr.write(
                        f'\rscored {batch_idx + 1}/{len(image_loader)} batches')
                    sys.stderr.flush()
            total = batch_idx + 1
            sys.stderr.write(
                f'\rscored {total}/{len(image_loader)} batches\n')
            sys.stderr.flush()
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='Score images with RSS-CNN')
    parser.add_argument('--model-dir', required=True,
                        help='Directory containing .pth model files')
    parser.add_argument('--images', required=True,
                        help='Path to VIPAC/images/ directory')
    parser.add_argument('--metadata', default=None,
                        help='Path to metadata-images.csv (optional)')
    parser.add_argument('--output', required=True,
                        help='Output path for scores.joblib')
    parser.add_argument('--cnn', default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--activation', default='sigmoid',
                        choices=['softrelu', 'sigmoid'])
    parser.add_argument('--catdim', type=int, default=2,
                        choices=[1, 2])
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load all model weights
    pth_files = load_models(args.model_dir)
    if not pth_files:
        print(f'No .pth files found in {args.model_dir}')
        return
    print(f'Loading {len(pth_files)} model(s) from {args.model_dir}')

    # Image dataset
    image_dataset = ImageDataset(
        args.images, metadata_path=args.metadata, transform=eval_transform)
    image_loader = DataLoader(
        image_dataset, batch_size=args.batch_size,
        num_workers=2, shuffle=False)
    print(f'Scoring {len(image_dataset)} images')

    # Score with each model
    all_scores = []
    for pth_fn in pth_files:
        print(f'  Scoring with {os.path.basename(pth_fn)}')
        model = RSSCNN(
            cnn_name=args.cnn, fdb_path=None, gap=True,
            activation=args.activation, catdim=args.catdim)
        model.load_state_dict(torch.load(pth_fn, map_location='cpu', weights_only=True))
        model = model.to(device)
        scores = score_model(model, image_loader, device)
        all_scores.append(scores)

    # Average across models
    img_ids = image_dataset.img_ids
    averaged = {}
    for img_id in img_ids:
        key = f"{img_id:06d}"
        averaged[key] = float(np.mean([s[key] for s in all_scores]))

    # Sort by score descending
    sorted_imgids = sorted(averaged, key=lambda k: averaged[k], reverse=True)

    # Save joblib
    result = {
        'lscored': averaged,
        'sorted_imgids': sorted_imgids,
        'n_models': len(pth_files),
        'model_files': [os.path.basename(f) for f in pth_files],
    }
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    joblib.dump(result, args.output, compress=3)
    print(f'Saved scores to {args.output}')

    # Save CSV
    csv_output = args.output.rsplit('.', 1)[0] + '.csv'
    with open(csv_output, 'w') as f:
        f.write('image_id,score\n')
        for img_id in img_ids:
            f.write(f'{img_id},{averaged[f"{img_id:06d}"]:.6f}\n')
    print(f'Saved CSV to {csv_output}')


if __name__ == '__main__':
    main()
