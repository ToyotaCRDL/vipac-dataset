"""Hierarchical clustering on image embeddings.

Two-phase approach for 120K images:
1. Stratified subsample → scipy linkage (on raw 512D CLIP embeddings)
2. Cut dendrogram → assign all images to cluster centroids → find 512D representatives
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage


def stratified_subsample(
    index: pd.DataFrame,
    groupby: str = "vehicle_id",
    n_per_group: int = 150,
    random_state: int = 42,
) -> np.ndarray:
    """Return global indices of a stratified subsample.

    Samples up to `n_per_group` points from each group. Returns indices
    into the full embedding arrays.
    """
    rng = np.random.RandomState(random_state)
    indices = []

    for _, group_id in index.groupby(groupby):
        group_idx = group_id.index.values.astype(int)
        sample_size = min(n_per_group, len(group_idx))
        chosen = rng.choice(group_idx, size=sample_size, replace=False)
        indices.extend(chosen)

    return np.array(indices, dtype=int)


def build_linkage(subsample: np.ndarray, method: str = "average") -> np.ndarray:
    """Build scipy linkage matrix from subsampled embedding vectors.

    Uses observation-vector mode (default for scipy linkage) which computes
    pairwise distances internally.
    """
    return linkage(subsample, method=method)


def cut_dendrogram(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut linkage matrix into `n_clusters` flat clusters."""
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def assign_all_to_clusters(
    full_embeddings: np.ndarray,
    subsample_embeddings: np.ndarray,
    subsample_labels: np.ndarray,
    batch_size: int = 20000,
) -> np.ndarray:
    """Assign all images to the nearest cluster centroid.

    Returns an array of 0-based cluster labels with length equal to
    full_embeddings.shape[0].
    """
    unique_labels = np.unique(subsample_labels)
    n_clusters = len(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    dim = subsample_embeddings.shape[1]
    centroids = np.empty((n_clusters, dim))
    for label, idx in label_to_idx.items():
        centroids[idx] = subsample_embeddings[subsample_labels == label].mean(axis=0)

    labels = np.empty(full_embeddings.shape[0], dtype=np.int32)
    for start in range(0, full_embeddings.shape[0], batch_size):
        batch = full_embeddings[start:start + batch_size]
        dists = np.linalg.norm(
            batch[:, None, :] - centroids[None, :, :], axis=-1
        )
        labels[start:start + batch_size] = np.argmin(dists, axis=1)

    return labels


def find_representatives(
    embeddings_512: np.ndarray,
    index: pd.DataFrame,
    all_labels: np.ndarray,
) -> list[dict]:
    """For each cluster, find the image closest to the cluster centroid in 512D.

    Returns a list of dicts (one per cluster) with keys:
      - cluster_id: int (0-based)
      - image_id: int (VIPAC image ID)
      - global_index: int (index into the full embedding arrays)
    """
    representatives = []

    for c in np.unique(all_labels):
        mask = all_labels == c
        cluster_embeddings = embeddings_512[mask]
        centroid = cluster_embeddings.mean(axis=0)
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        nearest_local = np.argmin(dists)
        global_idx = int(np.flatnonzero(mask)[nearest_local])
        image_id = int(index.iloc[global_idx]["image_id"])
        representatives.append({
            "cluster_id": int(c),
            "image_id": image_id,
            "global_index": global_idx,
        })

    return representatives


def relabel_by_size(
    all_labels: np.ndarray,
    representatives: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """Relabel clusters so the largest cluster gets ID 0.

    Clusters are sorted by descending member count. Ties are broken
    by original cluster_id (ascending) for determinism.

    Returns (relabeled_all_labels, relabeled_representatives) where each
    representative dict gains a "size" key with the cluster member count.
    """
    unique, counts = np.unique(all_labels, return_counts=True)
    # Sort by (-count, original_id) so largest cluster → ID 0
    order = np.argsort(-counts, kind="stable")
    sorted_original_ids = unique[order]
    old_to_new = {int(old): int(new) for new, old in enumerate(sorted_original_ids)}
    size_map = {int(old): int(c) for old, c in zip(unique, counts)}

    new_labels = np.array(
        [old_to_new[int(l)] for l in all_labels], dtype=np.int32
    )
    new_reps = [
        {**r, "cluster_id": old_to_new[r["cluster_id"]], "size": size_map[r["cluster_id"]]}
        for r in representatives
    ]
    return new_labels, new_reps


def sample_cluster_images(
    all_labels: np.ndarray,
    index: pd.DataFrame,
    n_per_cluster: int = 20,
    random_state: int = 42,
) -> list[dict]:
    """For each cluster, randomly sample up to n_per_cluster image IDs.

    Returns a list of dicts (one per cluster):
      - cluster_id: int
      - image_ids: list[int]  (sorted by image_id)
    """
    rng = np.random.RandomState(random_state)
    result = []
    for c in np.unique(all_labels):
        member_indices = np.flatnonzero(all_labels == c)
        sample_size = min(n_per_cluster, len(member_indices))
        chosen = rng.choice(member_indices, size=sample_size, replace=False)
        image_ids = sorted(int(index.iloc[i]["image_id"]) for i in chosen)
        result.append({
            "cluster_id": int(c),
            "image_ids": image_ids,
            "total_size": int(len(member_indices)),
        })
    return result


def print_cluster_summary(
    all_labels: np.ndarray,
    title: str = "",
) -> None:
    """Print a table of cluster sizes and verify the total matches."""
    unique, counts = np.unique(all_labels, return_counts=True)
    total = int(counts.sum())
    expected = len(all_labels)

    header = f"=== {title} ===" if title else "=== Cluster Summary ==="
    print(f"\n  {header}")
    for c, cnt in zip(unique, counts):
        print(f"    Cluster {int(c):3d}: {int(cnt):6d} images")
    print(f"    {'Total':11s}: {total:6d} images (expected: {expected})")

    if total != expected:
        print(f"    WARNING: Total {total} != expected {expected}")
    else:
        print(f"    OK: All {total} images assigned.")
