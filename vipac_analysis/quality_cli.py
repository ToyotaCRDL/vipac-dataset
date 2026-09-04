"""CLI entry point for VIPAC image quality evaluation.

Usage:
    python -m vipac_analysis.quality_cli \
        --images VIPAC/images/ \
        --output output/image_quality/image_quality_scores.csv
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="VIPAC Image Quality Evaluation Pipeline"
    )
    parser.add_argument(
        "--images", default="VIPAC/images/",
        help="Path to image directory (default: VIPAC/images/)",
    )
    parser.add_argument(
        "--metadata", default="VIPAC/metadata-images.csv",
        help="Path to metadata CSV (default: VIPAC/metadata-images.csv)",
    )
    parser.add_argument(
        "--output", default="output/image_quality/image_quality_scores.csv",
        help="Output CSV path (default: output/image_quality/image_quality_scores.csv)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="GPU batch size (default: 64)",
    )
    parser.add_argument(
        "--cpu-workers", type=int, default=4,
        help="CPU multiprocessing workers (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=5000,
        help="Save checkpoint every N images (default: 5000)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip GPU pass (MUSIQ)",
    )
    parser.add_argument(
        "--skip-clipiqa", action="store_true",
        help="Skip CLIPIQA GPU pass",
    )
    parser.add_argument(
        "--clipiqa-batch-size", type=int, default=32,
        help="CLIPIQA batch size (default: 32)",
    )
    parser.add_argument(
        "--skip-cpu", action="store_true",
        help="Skip CPU pass (blur/noise/exposure/contrast)",
    )
    parser.add_argument(
        "--include-brisque", action="store_true",
        help="Include BRISQUE in CPU pass (adds ~1h for 120K images)",
    )

    args = parser.parse_args()

    # Resolve relative paths
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.images):
        args.images = os.path.join(base, args.images)
    if not os.path.isabs(args.metadata):
        args.metadata = os.path.join(base, args.metadata)
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)

    from vipac_analysis.quality_pipeline import QualityPipeline

    pipeline = QualityPipeline(
        image_dir=args.images,
        metadata_path=args.metadata,
        output_path=args.output,
        batch_size=args.batch_size,
        cpu_workers=args.cpu_workers,
        checkpoint_interval=args.checkpoint_interval,
        skip_gpu=args.skip_gpu,
        skip_cpu=args.skip_cpu,
        skip_clipiqa=args.skip_clipiqa,
        clipiqa_batch_size=args.clipiqa_batch_size,
        include_brisque=args.include_brisque,
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
