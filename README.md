# VIPAC dataset and validation code

This repository contains a dataset of vehicle impressions from pairwise assessments across cultures (VIPAC) and validation codes for the dataset.

## Data access and the omitted image archive

The dataset is stored inside `VIPAC/`.
`VIPAC/images/` is intentionally not stored in this GitHub repository because of its size. The complete Figshare dataset release includes the images as `images.zip`. After downloading the Figshare package, extract `images.zip` inside `VIPAC/` (this creates `VIPAC/images/`) so that the analysis commands can resolve the images without code changes.

## Data documentation

See [`VIPAC/README.md`](VIPAC/README.md) for file-level descriptions and table columns.

## File checksums

`VIPAC/CHECKSUMS_SHA256.txt` records SHA-256 hashes for the dataset and documentation stored under `VIPAC/`.

## Environment setup

This repository uses [uv](https://docs.astral.sh/uv/) for environment management. If you do not have uv installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternatives: brew install uv, pipx install uv
```

From the repository root:

```bash
uv sync
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

All `python -m vipac_analysis...` commands below assume the venv is activated (equivalently, prefix them with `uv run`).

## Verified environment

All commands in this repository were executed and verified on:

| Component | Value |
|---|---|
| OS | Ubuntu 20.04.1 LTS |
| GPU | 1x NVIDIA A100-PCIE-40GB (driver 575.57.08) |
| Python | 3.12.13, managed with uv 0.11.14 |
| PyTorch | 2.5.1+cu121 (CUDA 12.1) |

## Validation analyses

Run commands from the repository root unless a command states otherwise. The image-dependent analyses require the Figshare image archive at `VIPAC/images/`.

### 1. Image quality evaluation

```bash
# CLIP embeddings
python -m vipac_analysis.embedding_cli --extract --output output/image_quality/

# Per-vehicle hierarchical clustering on CLIP embeddings
python -m vipac_analysis.embedding_cli \
  --hierarchical --per-vehicle-hierarchical \
  --output output/image_quality/

# Image quality metrics (GPU)
python -m vipac_analysis.quality_cli \
  --images VIPAC/images/ \
  --metadata VIPAC/metadata-images.csv \
  --output output/image_quality/image_quality_scores.csv

# Image quality metrics (CPU only)
python -m vipac_analysis.quality_cli \
  --images VIPAC/images/ \
  --metadata VIPAC/metadata-images.csv \
  --output output/image_quality/image_quality_scores_cpu.csv \
  --skip-gpu --skip-clipiqa \
  --cpu-workers 8
```

The GPU command downloads the MUSIQ checkpoint (~163 MB) to
`output/image_quality/models/paq2piq_ckpt.npz` on the first run and reuses
it on subsequent runs. The CLIP and CLIPIQA weights are cached by their
libraries in `~/.cache/clip/` outside the repository.

Relevant implementation files: `vipac_analysis/clipiqa.py`, `vipac_analysis/embedding_cli.py`, `vipac_analysis/extract_embeddings.py`, `vipac_analysis/hierarchical_clustering.py`, `vipac_analysis/musiq.py`, `vipac_analysis/quality_cli.py`, `vipac_analysis/quality_metrics.py`, `vipac_analysis/quality_pipeline.py`, `vipac_analysis/visualize_hierarchical.py`.

### 2. Image usage statistics

```bash
# US
python -m vipac_analysis.image_usage \
  --responses VIPAC/pairwise-responses-us.csv \
  --output output/image_usage/

# JP
python -m vipac_analysis.image_usage \
  --responses VIPAC/pairwise-responses-jp.csv \
  --output output/image_usage/
```

Relevant implementation files: `vipac_analysis/image_usage.py`.

### 3. Participant demographics

```bash
python -m vipac_analysis.demographics \
  --participants-us VIPAC/participants-us.csv \
  --participants-jp VIPAC/participants-jp.csv \
  --output output/demographics/
```

Relevant implementation files: `vipac_analysis/demographics.py`.

### 4. Dummy-trial reliability

```bash
python -m vipac_analysis.dummy_hgram \
  --responses-us VIPAC/pairwise-responses-us.csv \
  --responses-jp VIPAC/pairwise-responses-jp.csv \
  --output output/dummy/
```

Relevant implementation files: `vipac_analysis/dataset.py`, `vipac_analysis/dummy_hgram.py`.

### 5. Baseline predictive modeling and attribute relationships

Download FractalDB pre-trained ResNet34 weights (FractalDB-1000_resnet34.pth) from the link (Pre-trained models) in https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch.

```bash
# US models
for attr in modern sporty sleek elegant stylish dynamic aerodynamic sophisticated luxurious aggressive; do
  python -m vipac_analysis.train --country us --attribute $attr \
    --responses VIPAC/pairwise-responses-us.csv \
    --images VIPAC/images/ \
    --output output/models/us/$attr/ \
    --repetitions 10 \
    --ncorrect-thre 10 \
    --fdb FractalDB-1000_res34.pth
done

# JP models
for attr in modern sporty sleek elegant stylish dynamic aerodynamic sophisticated luxurious aggressive; do
  python -m vipac_analysis.train --country jp --attribute $attr \
    --responses VIPAC/pairwise-responses-jp.csv \
    --images VIPAC/images/ \
    --output output/models/jp/$attr/ \
    --repetitions 10 \
    --ncorrect-thre 10 \
    --fdb FractalDB-1000_res34.pth
done

# US scores for 120,000 images
for attr in modern sporty sleek elegant stylish dynamic aerodynamic sophisticated luxurious aggressive; do
  python -m vipac_analysis.score \
    --model-dir output/models/us/$attr/ \
    --images VIPAC/images/ \
    --output output/models/scores/us/$attr.joblib
done

# JP scores for 120,000 images
for attr in modern sporty sleek elegant stylish dynamic aerodynamic sophisticated luxurious aggressive; do
  python -m vipac_analysis.score \
    --model-dir output/models/jp/$attr/ \
    --images VIPAC/images/ \
    --output output/models/scores/jp/$attr.joblib
done

# US correlation matrix
python -m vipac_analysis.correlation --scores-dir output/models/scores/us/ --output output/models/scores/corr_matrix_us.csv

# JP correlation matrix
python -m vipac_analysis.correlation --scores-dir output/models/scores/jp/ --output output/models/scores/corr_matrix_jp.csv
```

Relevant implementation files: `vipac_analysis/__init__.py`, `vipac_analysis/correlation.py`, `vipac_analysis/model.py`, `vipac_analysis/report.py`, `vipac_analysis/score.py`, `vipac_analysis/train.py`.

### GPU vs CPU

We checked whether the same results can be obtained without a GPU by
re-running the device-dependent code paths on fixed subsets with CUDA
disabled (`CUDA_VISIBLE_DEVICES=""`), using identical inputs and seeds, and
comparing against the GPU runs.

Analysis 1 (500-image subset):

| Output | GPU vs CPU difference |
|---|---|
| Blur / noise / exposure / contrast columns | bit-identical (no GPU code path) |
| MUSIQ (`musiq_quality`) | max abs diff 8.6e-3 (mean 1.9e-3) |
| CLIPIQA (`clipiqa_quality`) | max abs diff 9.3e-4 (mean 2.1e-4) |
| CLIP 512-D embeddings | max abs diff 1.6e-2; min cosine similarity 0.999995 |

Analysis 5 smoke test (US / `modern`, 10,710 pairs, 1 repetition, seed 0):
test accuracy 71.06% (GPU) vs 70.21% (CPU); ranking accuracy 71.99% vs
71.43%; both runs early-stopped at the same epoch.

Practical consequences for CPU-only machines:

- Analyses 2-4 use only pandas / numpy / matplotlib and give identical
  results on CPU.
- Analysis 1: the CPU command documented below (`--skip-gpu
  --skip-clipiqa`) omits the MUSIQ and CLIPIQA columns by design, so those
  two columns cannot be obtained with it. Running these passes on CPU is
  possible but roughly 30-100x slower (500 images on 4 cores: MUSIQ 75 s
  vs 2.6 s, CLIPIQA 50 s vs 0.5 s, CLIP embeddings 138 s vs 1.7 s), i.e.
  about a day for the full 120,000-image set.
- Analysis 5: individual trainings run on CPU (smoke test above) and agree
  with GPU to within ~1 point of test accuracy, but are ~40x slower (one
  10.7k-pair run: ~2.7 h on 4 CPU cores vs ~4 min on GPU). The full sweep
  (10 attributes x 2 countries x 10 repetitions, ~98k pairs each) is on the
  order of months on CPU and not practical; trained models also differ at
  the floating-point level, so bit-identical reproduction requires the same
  device.

## Citation and licences

The final Data Descriptor citation, Figshare DOI to be added.

The source code (`vipac_analysis/`) is released under the MIT licence (see [`LICENSE`](LICENSE)). The VIPAC dataset (`VIPAC/`) is released under CC BY 4.0 (see [`VIPAC/README.md`](VIPAC/README.md)).

