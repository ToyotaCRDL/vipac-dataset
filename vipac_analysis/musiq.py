"""PyTorch implementation of MUSIQ (Multi-Scale Image Quality Transformer).

Based on: https://arxiv.org/abs/2108.05997
Checkpoint: gs://gresearch/musiq/musiq/paq2piq_ckpt.npz
"""

import collections
import io
import math
import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Preprocessing (NumPy equivalent of the TensorFlow preprocessing)
# ---------------------------------------------------------------------------

def resize_preserve_aspect(img: np.ndarray, longer_side: int) -> np.ndarray:
    """Resize image preserving aspect ratio so longer side = longer_side."""
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= longer_side:
        return img
    ratio = longer_side / longer
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img_pil)


def extract_patches(
    img: np.ndarray,
    patch_size: int,
    patch_stride: int,
    hse_grid_size: int = 10,
    scale_id: int = 0,
    max_seq_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract patches and compute position annotations.

    Returns patches (num_patches, P, P, 3), spatial_pos, scale_pos, mask.
    """
    h, w = img.shape[:2]
    count_h = int(math.ceil(h / patch_stride))
    count_w = int(math.ceil(w / patch_stride))

    # Pad image to be divisible by stride
    pad_h = count_h * patch_stride - h
    pad_w = count_w * patch_stride - w
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    # Extract patches
    patches = []
    for i in range(count_h):
        for j in range(count_w):
            y1, y2 = i * patch_stride, i * patch_stride + patch_size
            x1, x2 = j * patch_stride, j * patch_stride + patch_size
            patches.append(img[y1:y2, x1:x2])
    patches = np.array(patches)

    # Hashed spatial position embedding index (matches TF nearest-neighbor resize)
    pos_w = (np.arange(count_w) * hse_grid_size) // max(count_w, 1)
    pos_h = (np.arange(count_h) * hse_grid_size) // max(count_h, 1)
    grid_h, grid_w = np.meshgrid(pos_h, pos_w)
    spatial_pos = (grid_h * hse_grid_size + grid_w).reshape(-1)

    scale_pos = np.full_like(spatial_pos, scale_id)
    mask = np.ones_like(spatial_pos, dtype=bool)

    # Cap to max_seq_len
    if max_seq_len is not None and len(patches) > max_seq_len:
        patches = patches[:max_seq_len]
        spatial_pos = spatial_pos[:max_seq_len]
        scale_pos = scale_pos[:max_seq_len]
        mask = mask[:max_seq_len]

    return patches, spatial_pos, scale_pos, mask


def prepare_multiscale_input(
    img: np.ndarray,
    patch_size: int = 32,
    patch_stride: int = 32,
    hse_grid_size: int = 10,
    longer_side_lengths: List[int] = (224, 384),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare multi-scale patch representation for MUSIQ.

    Returns tensors: patches, spatial_pos, scale_pos, mask (all stacked).
    """
    all_patches, all_spatial, all_scale, all_mask = [], [], [], []

    longer_side_lengths = sorted(longer_side_lengths)
    for scale_id, longer_size in enumerate(longer_side_lengths):
        resized = resize_preserve_aspect(img, longer_size)
        max_seq = int(math.ceil(longer_size / patch_stride)) ** 2
        p, sp, scp, mk = extract_patches(
            resized, patch_size, patch_stride, hse_grid_size, scale_id, max_seq
        )
        all_patches.append(p)
        all_spatial.append(sp)
        all_scale.append(scp)
        all_mask.append(mk)

    # Native resolution (scale_id = len(longer_side_lengths))
    scale_id = len(longer_side_lengths)
    p, sp, scp, mk = extract_patches(
        img, patch_size, patch_stride, hse_grid_size, scale_id, None
    )
    all_patches.append(p)
    all_spatial.append(sp)
    all_scale.append(scp)
    all_mask.append(mk)

    patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    patches = patches / 127.5 - 1.0  # Normalize to [-1, 1]
    p_shape = patches.shape  # (L, P, P, 3)
    patches = patches.reshape(p_shape[0], -1)  # (L, P*P*3)

    spatial = np.concatenate(all_spatial, axis=0).astype(np.int64)
    scale = np.concatenate(all_scale, axis=0).astype(np.int64)
    mask = np.concatenate(all_mask, axis=0).astype(bool)

    return (
        torch.from_numpy(patches),
        torch.from_numpy(spatial),
        torch.from_numpy(scale),
        torch.from_numpy(mask),
    )


# ---------------------------------------------------------------------------
# Weight-standardized Conv2d
# ---------------------------------------------------------------------------

def _std_conv(x, weight, bias=None, stride=1, padding=0):
    """Conv2d with weight standardization along [out_c, in_c, h, w]."""
    w = weight - weight.mean(dim=[1, 2, 3], keepdim=True)
    w = w / (w.std(dim=[1, 2, 3], keepdim=True) + 1e-5)
    return F.conv2d(x, w, bias, stride=stride, padding=padding)


# ---------------------------------------------------------------------------
# ResNet patch embedding
# ---------------------------------------------------------------------------

# StdConv applies weight standardization at runtime (matches JAX training).
# Checkpoint stores raw weights; std is applied in forward pass.

class StdConv(nn.Module):
    """Weight-standardized Conv2d (no bias)."""

    def __init__(self, in_c, out_c, k, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return _std_conv(x, self.weight, stride=self.stride, padding=self.padding)


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block with StdConv + GroupNorm."""

    def __init__(self, inch, nout, stride=1, bottleneck=True):
        super().__init__()
        features, outc = nout, nout * 4 if bottleneck else (nout, nout)

        self.has_proj = inch != outc or stride != 1
        if self.has_proj:
            self.proj = StdConv(inch, outc, 1, stride=stride)
            self.proj_gn = nn.GroupNorm(32, outc, eps=1e-4)

        if bottleneck:
            self.conv1 = StdConv(inch, features, 1)
            self.gn1 = nn.GroupNorm(32, features, eps=1e-4)
            self.conv2 = StdConv(features, features, 3, padding=1)
            self.gn2 = nn.GroupNorm(32, features, eps=1e-4)
            self.conv3 = StdConv(features, outc, 1)
        else:
            self.conv1 = StdConv(inch, features, 3, padding=1)
            self.gn1 = nn.GroupNorm(32, features, eps=1e-4)
            self.conv2 = StdConv(features, outc, 3, padding=1)

        self.gn3 = nn.GroupNorm(32, outc, eps=1e-4)
        self.bottleneck = bottleneck

    def forward(self, x):
        residual = x
        if self.has_proj:
            residual = self.proj_gn(self.proj(residual))
        if self.bottleneck:
            x = F.relu(self.gn1(self.conv1(x)))
            x = F.relu(self.gn2(self.conv2(x)))
            x = self.gn3(self.conv3(x))
        else:
            x = F.relu(self.gn1(self.conv1(x)))
            x = self.gn3(self.conv2(x))
        return F.relu(residual + x)


class ResnetPatchEmbedding(nn.Module):
    """ResNet-based patch embedding for MUSIQ.

    num_layers=5: blocks=[1], bottleneck=True, total stride 4.
    Input: (N, 3, P, P) -> Output: (N, 256, H', W').
    """

    def __init__(self, num_layers=5):
        super().__init__()
        self.conv_root = StdConv(3, 64, 7, stride=2, padding=3)
        self.gn_root = nn.GroupNorm(32, 64, eps=1e-4)

        blocks_cfg, bottleneck = {
            5: ([1], True), 8: ([1, 1], True), 11: ([1, 1, 1], True),
            14: ([1, 1, 1, 1], True), 9: ([1, 1, 1, 1], False),
        }.get(num_layers, ([1], True))

        self.num_stages = len(blocks_cfg)
        inch = 64
        for i, bs in enumerate(blocks_cfg):
            nout = 64 * (2 ** i)
            s = 1 if i == 0 else 2
            outc = nout * 4 if bottleneck else nout
            units = []
            for j in range(bs):
                units.append(ResidualUnit(
                    inch if j == 0 else outc, nout, s if j == 0 else 1, bottleneck
                ))
            setattr(self, f"block{i + 1}", nn.Sequential(*units))
            inch = outc

    def forward(self, patches):
        x = self.gn_root(self.conv_root(patches))
        x = F.relu(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        for i in range(self.num_stages):
            x = getattr(self, f"block{i + 1}")(x)
        return x


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer encoder layer: MultiHeadAttn + MLP."""

    def __init__(self, hidden, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden),
        )

    def forward(self, x, mask):
        a, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            key_padding_mask=~mask if mask is not None else None,
        )
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with [CLS] token and position embeddings."""

    def __init__(self, hidden_size, num_heads, mlp_dim, num_layers,
                 num_spatial_positions, num_scales):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.spatial_embed = nn.Embedding(num_spatial_positions, hidden_size)
        self.scale_embed = nn.Embedding(num_scales, hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads, mlp_dim)
            for _ in range(num_layers)
        )
        self.ln_final = nn.LayerNorm(hidden_size)

    def forward(self, x, spatial_pos, scale_pos, mask):
        batch, seq, _ = x.shape
        x = x + self.spatial_embed(spatial_pos) + self.scale_embed(scale_pos)
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln_final(x)


# ---------------------------------------------------------------------------
# Full MUSIQ model
# ---------------------------------------------------------------------------

class MUSIQNet(nn.Module):
    """MUSIQ: Multi-Scale Image Quality Transformer."""

    def __init__(self, num_classes=1, hidden_size=384,
                 transformer_layers=14, transformer_heads=6,
                 transformer_mlp=1152, resnet_layers=5,
                 num_spatial_positions=100, num_scales=3):
        super().__init__()
        self.patch_embed = ResnetPatchEmbedding(resnet_layers)
        self.embedding = nn.Linear(16384, hidden_size)
        self.transformer = TransformerEncoder(
            hidden_size, transformer_heads, transformer_mlp,
            transformer_layers, num_spatial_positions, num_scales,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, patches, spatial_pos, scale_pos, mask):
        b, l, c = patches.shape
        patch_size = int(math.sqrt(c // 3))
        x = patches.view(b * l, patch_size, patch_size, 3).permute(0, 3, 1, 2)
        x = self.patch_embed(x)
        x = x.flatten(1)
        x = self.embedding(x)
        x = x.view(b, l, -1)
        x = self.transformer(x, spatial_pos, scale_pos, mask)
        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _recover_tree(key_list, val_list):
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(key_list, val_list):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = _recover_tree(k_subtree, v_subtree)
    return tree


def _squeeze(arr):
    """Convert JAX GN params (1,1,1,C) -> PyTorch (C,)."""
    return arr.reshape(-1) if arr.ndim == 4 else arr


def _load_npz_to_state_dict(npz_path):
    """Load Google Research npz checkpoint and convert to PyTorch state_dict."""
    with open(npz_path, "rb") as f:
        data = f.read()
    vals = np.load(io.BytesIO(data), allow_pickle=True)
    keys = list(vals.keys())
    params = _recover_tree(keys, [vals[k] for k in keys])["opt"]["target"]
    p = params

    sd = {}

    # conv_root: JAX (7,7,3,64) -> PT (64,3,7,7)
    sd["patch_embed.conv_root.weight"] = torch.from_numpy(
        p["conv_root"]["kernel"].transpose(3, 2, 0, 1)
    )
    sd["patch_embed.gn_root.weight"] = torch.from_numpy(_squeeze(p["gn_root"]["scale"]))
    sd["patch_embed.gn_root.bias"] = torch.from_numpy(_squeeze(p["gn_root"]["bias"]))

    # block1/unit1 - single bottleneck unit (num_layers=5)
    unit = p["block1"]["unit1"]
    for name in ("conv1", "conv2", "conv3"):
        k = unit[name]["kernel"]  # (kh, kw, ic, oc)
        sd[f"patch_embed.block1.0.{name}.weight"] = torch.from_numpy(k.transpose(3, 2, 0, 1))
    for i in range(1, 4):
        sd[f"patch_embed.block1.0.gn{i}.weight"] = torch.from_numpy(_squeeze(unit[f"gn{i}"]["scale"]))
        sd[f"patch_embed.block1.0.gn{i}.bias"] = torch.from_numpy(_squeeze(unit[f"gn{i}"]["bias"]))
    # projection
    pk = unit["conv_proj"]["kernel"]
    sd["patch_embed.block1.0.proj.weight"] = torch.from_numpy(pk.transpose(3, 2, 0, 1))
    sd["patch_embed.block1.0.proj_gn.weight"] = torch.from_numpy(_squeeze(unit["gn_proj"]["scale"]))
    sd["patch_embed.block1.0.proj_gn.bias"] = torch.from_numpy(_squeeze(unit["gn_proj"]["bias"]))

    # embedding: JAX (16384, 384) -> PT (384, 16384)
    sd["embedding.weight"] = torch.from_numpy(p["embedding"]["kernel"].T)
    sd["embedding.bias"] = torch.from_numpy(p["embedding"]["bias"])

    # Transformer
    t = p["Transformer"]
    sd["transformer.cls"] = torch.from_numpy(t["cls"])
    sd["transformer.spatial_embed.weight"] = torch.from_numpy(
        t["posembed_input"]["pos_embedding"]
    ).squeeze(0)
    sd["transformer.scale_embed.weight"] = torch.from_numpy(
        t["scaleembed_input"]["scale_embedding"]
    ).squeeze(0)

    for i in range(14):
        block = t[f"encoderblock_{i}"]
        pf = f"transformer.layers.{i}"
        sd[f"{pf}.ln1.weight"] = torch.from_numpy(block["LayerNorm_0"]["scale"])
        sd[f"{pf}.ln1.bias"] = torch.from_numpy(block["LayerNorm_0"]["bias"])

        attn = block["MultiHeadDotProductAttention_1"]
        # JAX qkv kernel: (384, 6, 64) -> PT in_proj: stack to (1152, 384)
        q_w = attn["query"]["kernel"].reshape(384, 384).T
        k_w = attn["key"]["kernel"].reshape(384, 384).T
        v_w = attn["value"]["kernel"].reshape(384, 384).T
        sd[f"{pf}.attn.in_proj_weight"] = torch.from_numpy(
            np.concatenate([q_w, k_w, v_w], axis=0)
        )
        q_b = attn["query"]["bias"].reshape(384)
        k_b = attn["key"]["bias"].reshape(384)
        v_b = attn["value"]["bias"].reshape(384)
        sd[f"{pf}.attn.in_proj_bias"] = torch.from_numpy(
            np.concatenate([q_b, k_b, v_b], axis=0)
        )
        o_w = attn["out"]["kernel"].reshape(384, 384).T  # PT: (384, 384)
        sd[f"{pf}.attn.out_proj.weight"] = torch.from_numpy(o_w)
        sd[f"{pf}.attn.out_proj.bias"] = torch.from_numpy(attn["out"]["bias"])

        sd[f"{pf}.ln2.weight"] = torch.from_numpy(block["LayerNorm_2"]["scale"])
        sd[f"{pf}.ln2.bias"] = torch.from_numpy(block["LayerNorm_2"]["bias"])

        mlp = block["MlpBlock_3"]
        sd[f"{pf}.mlp.0.weight"] = torch.from_numpy(mlp["Dense_0"]["kernel"].T)
        sd[f"{pf}.mlp.0.bias"] = torch.from_numpy(mlp["Dense_0"]["bias"])
        sd[f"{pf}.mlp.2.weight"] = torch.from_numpy(mlp["Dense_1"]["kernel"].T)
        sd[f"{pf}.mlp.2.bias"] = torch.from_numpy(mlp["Dense_1"]["bias"])

    # Final LayerNorm
    sd["transformer.ln_final.weight"] = torch.from_numpy(t["encoder_norm"]["scale"])
    sd["transformer.ln_final.bias"] = torch.from_numpy(t["encoder_norm"]["bias"])

    # Head: JAX (384, 1) -> PT (1, 384)
    sd["head.weight"] = torch.from_numpy(p["head"]["kernel"].T)
    sd["head.bias"] = torch.from_numpy(p["head"]["bias"])

    return sd


# ---------------------------------------------------------------------------
# Download + build helpers
# ---------------------------------------------------------------------------

def download_checkpoint(checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "paq2piq_ckpt.npz")
    if os.path.exists(path):
        return path
    print("Downloading MUSIQ checkpoint (163 MB) ...")
    url = (
        "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/"
        "musiq%2Fpaq2piq_ckpt.npz?generation=1638397896105603&alt=media"
    )
    with urllib.request.urlopen(url, timeout=120) as resp, open(path, "wb") as f:
        data = resp.read()
        f.write(data)
    print(f"Saved to {path} ({len(data)} bytes)")
    return path


def build_model(checkpoint_path: str, device: str = "cuda") -> MUSIQNet:
    model = MUSIQNet(num_classes=1).to(device)
    sd = _load_npz_to_state_dict(checkpoint_path)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict_quality(model: MUSIQNet, img: np.ndarray,
                    device: str = "cuda") -> float:
    """MUSIQ quality score for a single RGB image (H, W, 3, uint8)."""
    patches, spatial, scale, mask = prepare_multiscale_input(img)
    with torch.no_grad():
        score = model(
            patches.unsqueeze(0).to(device),
            spatial.unsqueeze(0).to(device),
            scale.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
        )
    return float(score.item())
