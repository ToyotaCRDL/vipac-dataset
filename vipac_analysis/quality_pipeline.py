"""GPU + CPU quality evaluation pipeline for VIPAC images."""

import functools
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vipac_analysis.config import COLORS, KWORDS, all_image_ids, image_id_str
from vipac_analysis.dataset import ImageDataset
from vipac_analysis.musiq import (
    MUSIQNet,
    build_model,
    download_checkpoint,
    prepare_multiscale_input,
)
from vipac_analysis.quality_metrics import load_cpu_metrics


class QualityPipeline:
    def __init__(
        self,
        image_dir: str,
        metadata_path: str,
        output_path: str,
        batch_size: int = 64,
        cpu_workers: int = 4,
        checkpoint_interval: int = 5000,
        skip_gpu: bool = False,
        skip_cpu: bool = False,
        skip_clipiqa: bool = False,
        clipiqa_batch_size: int = 32,
        include_brisque: bool = False,
        resume: bool = False,
    ):
        self.image_dir = image_dir.rstrip("/")
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.cpu_workers = cpu_workers
        self.chkpt_interval = checkpoint_interval
        self.skip_gpu = skip_gpu
        self.skip_cpu = skip_cpu
        self.skip_clipiqa = skip_clipiqa
        self.clipiqa_batch_size = clipiqa_batch_size
        self.include_brisque = include_brisque
        self.resume = resume

        self.chkpt_path = os.path.splitext(output_path)[0] + ".chkpt"

    def run(self):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        # Image list
        img_ids = list(all_image_ids())

        # Checkpoint / resume
        completed_ids, results = set(), {}
        if self.resume and os.path.exists(self.chkpt_path):
            data = joblib.load(self.chkpt_path)
            completed_ids = set(data["completed_image_ids"])
            results = data["results"]
            print(f"Resuming from checkpoint: {len(completed_ids)} already done")

        # Load existing scores from output CSV (preserves skipped passes)
        if os.path.exists(self.output_path):
            existing = pd.read_csv(self.output_path)
            for _, row in existing.iterrows():
                key = str(row.get("image_id", ""))
                if key and key not in results:
                    results[key] = {}
                    for col in [
                        "musiq_quality", "clipiqa_quality", "brisque_quality",
                        "blur_laplacian_var", "noise_sigma",
                        "exposure_mean", "exposure_p10", "exposure_p50",
                        "exposure_p90", "exposure_peak_bin", "contrast_rms",
                    ]:
                        if col in row and not pd.isna(row[col]):
                            results[key][col] = float(row[col])
            print(f"Loaded existing scores for {len(results)} images from {self.output_path}")

            # Only mark an image as fully completed if it has ALL scores for ALL
            # passes that will run in this invocation. This prevents a pass from
            # being skipped because another pass already has scores.
            required_cols = []
            if not self.skip_gpu:
                required_cols.append("musiq_quality")
            if not self.skip_clipiqa:
                required_cols.append("clipiqa_quality")
            if not self.skip_cpu:
                required_cols.append("blur_laplacian_var")
            for key, vals in results.items():
                if all(col in vals for col in required_cols):
                    completed_ids.add(key)

        if not self.skip_gpu:
            results = self._run_gpu(img_ids, completed_ids, results)
        if not self.skip_clipiqa:
            results = self._run_clipiqa(img_ids, completed_ids, results)
        if not self.skip_cpu:
            results = self._run_cpu(img_ids, completed_ids, results)

        # Build DataFrame
        df = self._build_df(img_ids, results)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Saved {len(df)} rows -> {self.output_path}")

        # Summary
        self._save_summary(df)

        # Clean checkpoint
        if os.path.exists(self.chkpt_path):
            os.remove(self.chkpt_path)

    def _run_gpu(self, img_ids, completed_ids, results):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== GPU pass (MUSIQ) on {device} ===")

        ckpt_dir = os.path.join(os.path.dirname(self.output_path) or ".", "models")
        ckpt = download_checkpoint(ckpt_dir)
        model = build_model(ckpt, device)

        # Process in batches
        remaining = [i for i in img_ids if image_id_str(i) not in completed_ids]
        total = len(remaining)

        for start in range(0, total, self.batch_size):
            batch_ids = remaining[start : start + self.batch_size]
            batch_imgs = []
            for img_id in batch_ids:
                path = f"{self.image_dir}/{image_id_str(img_id)}.png"
                from PIL import Image
                batch_imgs.append(np.array(Image.open(path).convert("RGB")))

            # Find max patches for padding
            all_inputs = [prepare_multiscale_input(img) for img in batch_imgs]
            max_len = max(p.shape[0] for p, _, _, _ in all_inputs)

            # Pad to max_len
            patches_list, spatial_list, scale_list, mask_list = [], [], [], []
            for p, s, sc, m in all_inputs:
                cur_len = p.shape[0]
                pad = max_len - cur_len
                if pad > 0:
                    p = torch.cat([p, torch.zeros(pad, p.shape[1])], dim=0)
                    s = torch.cat([s, torch.zeros(pad, dtype=s.dtype)])
                    sc = torch.cat([sc, torch.zeros(pad, dtype=sc.dtype)])
                    m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)])
                patches_list.append(p)
                spatial_list.append(s)
                scale_list.append(sc)
                mask_list.append(m)

            batch_patches = torch.stack(patches_list).to(device)
            batch_spatial = torch.stack(spatial_list).to(device)
            batch_scale = torch.stack(scale_list).to(device)
            batch_mask = torch.stack(mask_list).to(device)

            try:
                with torch.no_grad():
                    scores = model(batch_patches, batch_spatial, batch_scale, batch_mask)
                for img_id, score in zip(batch_ids, scores.flatten()):
                    results[image_id_str(img_id)] = {"musiq_quality": float(score)}
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise RuntimeError("OOM - reduce batch size") from e
                raise

            done = start + len(batch_ids)
            if done % self.chkpt_interval < self.batch_size:
                self._save_checkpoint(
                    {image_id_str(i) for i in img_ids[:done]}, results
                )
            if done % 50000 < self.batch_size:
                print(f"  GPU: {done}/{total} images")

        print(f"  GPU pass done: {total} images\n")
        return results

    def _run_clipiqa(self, img_ids, completed_ids, results):
        from vipac_analysis.clipiqa import build_model as build_clipiqa, predict_batch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== GPU pass (CLIPIQA) on {device} ===")

        model = build_clipiqa(device=device)
        remaining = [i for i in img_ids if image_id_str(i) not in completed_ids]
        total = len(remaining)

        for start in range(0, total, self.clipiqa_batch_size):
            batch_ids = remaining[start : start + self.clipiqa_batch_size]
            batch_imgs = []
            for img_id in batch_ids:
                path = f"{self.image_dir}/{image_id_str(img_id)}.png"
                from PIL import Image
                batch_imgs.append(np.array(Image.open(path).convert("RGB")))

            scores = predict_batch(model, batch_imgs)
            for img_id, score in zip(batch_ids, scores):
                key = image_id_str(img_id)
                if key in results:
                    results[key]["clipiqa_quality"] = float(score)
                else:
                    results[key] = {"clipiqa_quality": float(score)}

            done = start + len(batch_ids)
            if done % self.chkpt_interval < self.clipiqa_batch_size:
                self._save_checkpoint(
                    {image_id_str(i) for i in img_ids[:done]}, results
                )
            if done % 50000 < self.clipiqa_batch_size:
                print(f"  CLIPIQA: {done}/{total} images")

        print(f"  CLIPIQA pass done: {total} images\n")
        return results

    def _run_cpu(self, img_ids, completed_ids, results):
        extra = " + BRISQUE" if self.include_brisque else ""
        print(f"\n=== CPU pass (blur/noise/exposure/contrast{extra}) ===")
        remaining = [i for i in img_ids if image_id_str(i) not in completed_ids]
        total = len(remaining)
        paths = [f"{self.image_dir}/{image_id_str(i)}.png" for i in remaining]

        worker = functools.partial(load_cpu_metrics, include_brisque=self.include_brisque)
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.cpu_workers, mp_context=mp_context) as pool:
            it = tqdm(pool.map(worker, paths), total=total, desc="CPU")
            for img_id, metrics in zip(remaining, it):
                key = image_id_str(img_id)
                if key in results:
                    results[key].update(metrics)
                else:
                    results[key] = metrics

        done_count = total
        if done_count % self.chkpt_interval < 1:
            self._save_checkpoint(
                {image_id_str(i) for i in img_ids}, results
            )
        print(f"  CPU pass done: {total} images\n")
        return results

    def _build_df(self, img_ids, results):
        rows = []
        for img_id in img_ids:
            key = image_id_str(img_id)
            row = {"image_id": img_id}
            if key in results:
                row.update(results[key])
            else:
                row["musiq_quality"] = float("nan")
                row["clipiqa_quality"] = float("nan")
                row.update({k: float("nan") for k in [
                    "blur_laplacian_var", "noise_sigma",
                    "exposure_mean", "exposure_p10", "exposure_p50",
                    "exposure_p90", "exposure_peak_bin", "contrast_rms",
                ]})
                if self.include_brisque:
                    row["brisque_quality"] = float("nan")
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add metadata
        if self.metadata_path and os.path.exists(self.metadata_path):
            meta = pd.read_csv(self.metadata_path)
            df = df.merge(meta, left_on="image_id", right_on="image-id", how="left")
        else:
            # Derive from image_id formula
            for img_id in df["image_id"]:
                ic = img_id // 20000
                ik = (img_id % 20000) // 2000
                iv = (img_id % 2000) // 100
                is_ = img_id % 100
                df.loc[df["image_id"] == img_id, "vehicle_id"] = iv
                df.loc[df["image_id"] == img_id, "color"] = COLORS[ic]
                df.loc[df["image_id"] == img_id, "attribute"] = KWORDS[ik]
                df.loc[df["image_id"] == img_id, "seed"] = 1000 + is_

        return df

    def _save_summary(self, df):
        metric_cols = [
            "musiq_quality", "clipiqa_quality", "blur_laplacian_var",
            "noise_sigma", "exposure_mean", "exposure_p10", "exposure_p50",
            "exposure_p90", "exposure_peak_bin", "contrast_rms",
        ]
        if self.include_brisque:
            metric_cols.insert(2, "brisque_quality")
        metric_cols = [c for c in metric_cols if c in df.columns]
        summary = []
        for col in metric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            summary.append({
                "metric": col,
                "min": s.min(),
                "p5": np.percentile(s, 5),
                "median": s.median(),
                "mean": s.mean(),
                "p95": np.percentile(s, 95),
                "max": s.max(),
            })
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.splitext(self.output_path)[0] + "_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved -> {summary_path}")
        print(summary_df.to_string(index=False))

    def _save_checkpoint(self, completed_ids, results):
        joblib.dump(
            {"completed_image_ids": list(completed_ids), "results": results},
            self.chkpt_path,
            compress=3,
        )
