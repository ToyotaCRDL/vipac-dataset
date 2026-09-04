"""Hierarchical clustering visualizations for image embeddings.

Produces dendrograms with image thumbnails, tree layouts,
and representative/cluster image grids.
"""

import os
from os import path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def _save_figure(fig, output_path, dpi=300, save_format="png", **kwargs):
    """Save a figure in the specified format(s).

    Parameters
    ----------
    save_format : str
        "png" (default), "svg", or "both".  "both" writes .png and .svg.

    When saving as SVG, ``svg.fonttype`` is temporarily set to ``"none"``
    so that text is emitted as editable ``<text>`` elements rather than
    being converted to path outlines.
    """
    base, _ = path.splitext(output_path)
    os.makedirs(path.dirname(path.abspath(base)), exist_ok=True)

    formats = ("png", "svg") if save_format == "both" else (save_format,)
    for fmt in formats:
        if fmt == "svg":
            with matplotlib.rc_context({
                "svg.fonttype": "none",
                "font.family": "Liberation Sans",
                "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "sans-serif"],
                "font.size": 18,
            }):
                fig.savefig(f"{base}.{fmt}", dpi=dpi, **kwargs)
        else:
            fig.savefig(f"{base}.{fmt}", dpi=dpi, **kwargs)
        print(f"  {base}.{fmt}")


def plot_dendrogram_with_images(
    Z: np.ndarray,
    representatives: list[dict],
    subsample_indices: np.ndarray,
    index: pd.DataFrame,
    images_dir: str,
    output_path: str,
    n_clusters: int,
    save_format: str = "png",
):
    """Dendrogram (top) with representative image strip (bottom).

    The dendrogram shows the full hierarchical structure on the subsample.
    Representative images for each flat cluster are placed below, ordered
    by the dendrogram's leaf positions.
    """
    from vipac_analysis.hierarchical_clustering import cut_dendrogram

    reps_by_cluster = {r["cluster_id"]: r for r in representatives}
    n_clusters_present = len(reps_by_cluster)
    cluster_colors = sns.color_palette("husl", n_clusters_present)
    n = len(subsample_indices)

    fig = plt.figure(figsize=(max(30, n_clusters * 2), 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.05)
    ax_dendro = fig.add_subplot(gs[0])
    ax_images = fig.add_subplot(gs[1])

    # --- Dendrogram ---
    color_threshold = 0.7 * Z[:, 2].max()
    R = dendrogram(
        Z,
        ax=ax_dendro,
        color_threshold=color_threshold,
        labels=None,
        show_leaf_counts=False,
        leaf_font_size=0,
    )

    ax_dendro.set_title(
        f"Hierarchical Clustering ({n} images, "
        f"{n_clusters} clusters)",
    )
    ax_dendro.set_ylabel("Distance")
    ax_dendro.set_xlabel(f"Cluster (n={n_clusters})")

    # --- Leaf ordering: dendrogram places leaves at x = 0, 1, ..., n-1 ---
    leaves = R["leaves"]  # subsample indices in leaf order

    # Get color list from dendrogram (one color per leaf)
    color_list = R.get("color_list", [])

    # Assign each leaf position to a flat cluster (fcluster is 1-based)
    subsample_labels = cut_dendrogram(Z, n_clusters)
    leaf_clusters = np.array([subsample_labels[l] - 1 for l in leaves])  # 0-based

    # For each cluster, find the center x position of its leaves
    cluster_x = {}
    for c in range(n_clusters_present):
        positions = np.where(leaf_clusters == c)[0]
        if len(positions) > 0:
            cluster_x[c] = (positions.min() + positions.max()) / 2.0

    sorted_clusters = sorted(cluster_x.keys(), key=lambda c: cluster_x[c])

    # --- Image strip at bottom ---
    ax_images.set_xlim(-0.5, n - 0.5)
    ax_images.set_ylim(-1, 1)

    # Draw cluster background bands
    for i, c in enumerate(sorted_clusters):
        positions = np.where(leaf_clusters == c)[0]
        x_left = positions.min() - 0.5
        x_right = positions.max() + 0.5
        ax_images.axvspan(x_left, x_right, alpha=0.1, color=cluster_colors[c])

    # Place representative images
    img_width = min(1.0, (n - 1) / max(n_clusters_present, 1) * 0.8)
    for c in sorted_clusters:
        rep = reps_by_cluster[c]
        x_pos = cluster_x[c]
        img_path = path.join(images_dir, f"{rep['image_id']:06d}.png")
        if not path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg")
        if not path.exists(img_path):
            continue

        try:
            img = mpimg.imread(img_path)
            ax_images.imshow(
                img,
                extent=[x_pos - img_width / 2, x_pos + img_width / 2, -0.9, 0.9],
                aspect="equal",
                zorder=2,
            )
            # Color frame
            rect = plt.Rectangle(
                (x_pos - img_width / 2, -0.9),
                img_width, 1.8,
                fill=False, edgecolor=cluster_colors[c],
                linewidth=2, zorder=3,
            )
            ax_images.add_patch(rect)
        except Exception:
            pass

    ax_images.set_xticks(
        [cluster_x[c] for c in sorted_clusters],
        labels=[f"#{c}" for c in sorted_clusters],
    )
    ax_images.set_xlabel("Cluster ID")
    ax_images.set_yticks([])

    _save_figure(fig, output_path, bbox_inches="tight", save_format=save_format)
    plt.close(fig)


def plot_tree_layout(
    Z: np.ndarray,
    representatives: list[dict],
    subsample_labels: np.ndarray,
    images_dir: str,
    output_path: str,
    n_clusters: int,
    save_format: str = "png",
):
    """Tree layout: show the cluster-level hierarchy with representative images.

    Builds a reduced linkage tree where leaves are the flat clusters (not
    individual observations). Each internal node shows the merge distance.
    Representative images are placed below their cluster leaf.
    """
    reps_by_cluster = {r["cluster_id"]: r for r in representatives}
    n_clusters_present = len(reps_by_cluster)
    cluster_colors = sns.color_palette("husl", n_clusters_present)

    # Build the cluster-level hierarchy from Z and subsample_labels.
    # Trace which clusters merge at each step of Z.
    n_original = len(subsample_labels)
    # Each original point belongs to a cluster (0-based)
    point_to_cluster = np.array([int(l) - 1 for l in subsample_labels])

    # Track which clusters each node contains
    node_clusters = {}  # node_id -> set of cluster ids
    for i in range(n_original):
        c = point_to_cluster[i]
        node_clusters.setdefault(i, set()).add(c)

    # Build merge info: for each Z row, find the set of clusters in each child
    all_nodes = []  # (left_node, right_node, dist, node_id)
    for i, row in enumerate(Z):
        left = int(row[0])
        right = int(row[1])
        dist = row[2]
        node_id = n_original + i
        clusters = node_clusters.get(left, set()) | node_clusters.get(right, set())
        node_clusters[node_id] = clusters
        all_nodes.append((left, right, dist, node_id, clusters))

    # Build a reduced tree with one leaf per cluster
    sorted_reps = sorted(reps_by_cluster.keys())
    cluster_map = {c: i for i, c in enumerate(sorted_reps)}

    # Map original node_id -> reduced tree node_id
    reduced_id_map = {}  # original_node_id -> reduced_node_id
    next_reduced = n_clusters_present

    # For leaf nodes (individual points), map to their cluster reduced id
    for i in range(n_original):
        c = point_to_cluster[i]
        reduced_id_map[i] = cluster_map[c]

    reduced_z_rows = []
    for left, right, dist, node_id, clusters in all_nodes:
        lid = reduced_id_map[left]
        rid = reduced_id_map[right]
        if lid != rid:
            reduced_z_rows.append([lid, rid, dist, len(clusters)])
            reduced_id_map[node_id] = next_reduced
            next_reduced += 1
        else:
            # Same cluster - still need to map this node to the existing reduced id
            reduced_id_map[node_id] = lid

    if len(reduced_z_rows) < 1:
        # Degenerate: not enough merges, just show a simple layout
        fig, ax = plt.subplots(figsize=(max(10, n_clusters_present * 1.5), 4))
        for i, c in enumerate(sorted_reps):
            ax.scatter([i], [0], s=500, color=cluster_colors[c], alpha=0.7)
            ax.set_xticks(range(n_clusters_present))
            ax.set_xticklabels([f"#{c}" for c in sorted_reps])
            ax.set_title(f"Tree Layout ({n_clusters} clusters)")
        _save_figure(fig, output_path, bbox_inches="tight", save_format=save_format)
        plt.close(fig)
        return

    reduced_Z = np.array(reduced_z_rows)

    fig = plt.figure(figsize=(max(16, n_clusters_present * 2), 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.1)
    ax_tree = fig.add_subplot(gs[0])
    ax_images = fig.add_subplot(gs[1], sharex=ax_tree)

    # Draw dendrogram of the reduced tree (no labels - we'll add our own)
    R = dendrogram(
        reduced_Z,
        ax=ax_tree,
        color_threshold=0.7 * reduced_Z[:, 2].max(),
        no_labels=True,
    )

    ax_tree.set_title(f"Cluster Tree Layout ({n_clusters} clusters)")
    ax_tree.set_ylabel("Merge Distance")

    # Get cluster labels in leaf order from dendrogram
    # R['ivl'] maps leaf position → label string. R['leaves'] gives the
    # subsample indices in leaf display order.
    leaves_order = R["leaves"]  # indices into sorted_reps

    # The dendrogram internally places leaves at positions 0, 1, ..., n-1.
    # After rendering the actual xlim may differ. Read text positions.
    # We need to draw first, then read positions.
    fig.canvas.draw()

    # Get x positions from the dendrogram's internal coordinate mapping
    # Each leaf's x position can be derived from the line objects
    xlim = ax_tree.get_xlim()
    n_leaves = len(leaves_order)
    x_per_unit = (xlim[1] - xlim[0]) / n_leaves

    # Place images at evenly-spaced positions matching dendrogram leaves
    for i, leaf_pos in enumerate(leaves_order):
        c = sorted_reps[leaf_pos]
        x = xlim[0] + x_per_unit * (i + 0.5)
        rep = reps_by_cluster.get(c)
        if rep is None:
            continue
        img_path = path.join(images_dir, f"{rep['image_id']:06d}.png")
        if not path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg")
        if not path.exists(img_path):
            continue
        try:
            img = mpimg.imread(img_path)
            img_size = min(1.2, x_per_unit * 0.8)
            ax_images.imshow(
                img,
                extent=[x - img_size, x + img_size, -0.9, 0.9],
                aspect="equal", zorder=2,
            )
            rect = plt.Rectangle(
                (x - img_size, -0.9), img_size * 2, 1.8,
                fill=False, edgecolor=cluster_colors[c],
                linewidth=2, zorder=3,
            )
            ax_images.add_patch(rect)
            ax_images.text(x, 1.15, f"#{c}", ha="center", va="bottom",
                           fontweight="bold")
        except Exception:
            pass

    ax_images.set_ylim(-1, 1.5)
    ax_images.set_yticks([])
    ax_images.set_xlabel("Cluster")

    # Add cluster labels to dendrogram x-axis
    ax_tree.set_xticks(
        [xlim[0] + x_per_unit * (i + 0.5) for i in range(n_leaves)],
        [f"#{sorted_reps[leaves_order[i]]}" for i in range(n_leaves)],
    )

    _save_figure(fig, output_path, bbox_inches="tight", save_format=save_format)
    plt.close(fig)


def plot_representative_grid(
    representatives: list[dict],
    images_dir: str,
    output_path: str,
    n_clusters: int,
    cell_inches: float = 2.5,
    dpi: int = 150,
    save_format: str = "png",
    label_bg: bool = True,
    vehicle_id: int | None = None,
):
    """Arrange representative images in a tight grid with no gaps.

    Images are placed in a rectangular grid close to square aspect ratio.
    Each cell is cell_inches × cell_inches.
    """
    reps_sorted = sorted(representatives, key=lambda r: r["cluster_id"])
    n = len(reps_sorted)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, ax = plt.subplots(figsize=(cols * cell_inches, rows * cell_inches))
    ax.set_xlim(0, cols * cell_inches)
    ax.set_ylim(0, rows * cell_inches)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, rep in enumerate(reps_sorted):
        col = i % cols
        row = i // cols
        x0 = col * cell_inches
        y0 = (rows - 1 - row) * cell_inches  # Row 0 at top
        img_path = path.join(images_dir, f"{rep['image_id']:06d}.png")
        if not path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg")
        if path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                ax.imshow(
                    img,
                    extent=[x0, x0 + cell_inches, y0, y0 + cell_inches],
                    aspect="equal",
                )
            except Exception:
                ax.add_patch(plt.Rectangle(
                    (x0, y0), cell_inches, cell_inches,
                    facecolor="#ccc", edgecolor="none",
                ))
                ax.text(
                    x0 + cell_inches / 2, y0 + cell_inches / 2,
                    "?", ha="center", va="center",
                    fontsize=cell_inches * 8, color="#888",
                )
        else:
            ax.add_patch(plt.Rectangle(
                (x0, y0), cell_inches, cell_inches,
                facecolor="#ccc", edgecolor="none",
            ))

        ax.text(
            x0 + cell_inches / 2, y0 + 0.15,
            f"#{rep['cluster_id']} ({rep.get('size', '?')})",
            ha="center", va="bottom",
            fontsize=max(18, cell_inches * 8),
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8) if label_bg else None,
        )

    if vehicle_id is None:
        title = f"Representative Images ({n} clusters)"
    else:
        title = (
            f"Representative image per cluster #0-{n_clusters - 1} "
            f"({n_clusters} clusters) for vehicle-{vehicle_id}"
        )
    ax.set_title(title, pad=10)
    _save_figure(fig, output_path, dpi=dpi, bbox_inches="tight", save_format=save_format)
    plt.close(fig)


def plot_cluster_samples(
    cluster_samples: list[dict],
    images_dir: str,
    output_dir: str,
    cell_inches: float = 2.5,
    dpi: int = 150,
    save_format: str = "png",
    label_bg: bool = True,
):
    """For each cluster, render a 5-column grid of sampled images with ID labels."""
    os.makedirs(output_dir, exist_ok=True)
    cols = 5

    for cs in sorted(cluster_samples, key=lambda x: x["cluster_id"]):
        image_ids = cs["image_ids"]
        n = len(image_ids)
        rows = int(np.ceil(n / cols))

        fig, ax = plt.subplots(figsize=(cols * cell_inches, rows * cell_inches))
        ax.set_xlim(0, cols * cell_inches)
        ax.set_ylim(0, rows * cell_inches)
        ax.set_aspect("equal")
        ax.axis("off")

        for i, image_id in enumerate(image_ids):
            col = i % cols
            row = i // cols
            x0 = col * cell_inches
            y0 = (rows - 1 - row) * cell_inches

            img_path = path.join(images_dir, f"{image_id:06d}.png")
            if not path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
            if path.exists(img_path):
                try:
                    img = mpimg.imread(img_path)
                    ax.imshow(
                        img,
                        extent=[x0, x0 + cell_inches, y0, y0 + cell_inches],
                        aspect="equal",
                    )
                except Exception:
                    ax.add_patch(plt.Rectangle(
                        (x0, y0), cell_inches, cell_inches,
                        facecolor="#ccc", edgecolor="none",
                    ))
            else:
                ax.add_patch(plt.Rectangle(
                    (x0, y0), cell_inches, cell_inches,
                    facecolor="#ccc", edgecolor="none",
                ))

            ax.text(
                x0 + cell_inches / 2, y0 + 0.15,
                f"{image_id:06d}",
                ha="center", va="bottom",
                fontsize=max(18, cell_inches * 6),
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8) if label_bg else None,
            )

        ax.set_title(
            f"Cluster #{cs['cluster_id']} ({n}/{cs['total_size']} samples)",
            pad=10,
        )
        output_path = path.join(output_dir, f"cluster_{cs['cluster_id']}_samples")
        _save_figure(fig, output_path, dpi=dpi, bbox_inches="tight", save_format=save_format)
        plt.close(fig)

    print(f"  Saved cluster sample grids to {output_dir}/")


def plot_largest_cluster_grid(
    image_ids: list[int],
    total_size: int,
    images_dir: str,
    output_path: str,
    vehicle_id: int,
    cell_inches: float = 2.5,
    dpi: int = 150,
    save_format: str = "png",
    label_bg: bool = True,
):
    """Render a 5-column grid of sampled images from the largest cluster."""
    cols = 5
    n = len(image_ids)
    rows = int(np.ceil(n / cols))

    fig, ax = plt.subplots(figsize=(cols * cell_inches, rows * cell_inches))
    ax.set_xlim(0, cols * cell_inches)
    ax.set_ylim(0, rows * cell_inches)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, image_id in enumerate(image_ids):
        col = i % cols
        row = i // cols
        x0 = col * cell_inches
        y0 = (rows - 1 - row) * cell_inches

        img_path = path.join(images_dir, f"{image_id:06d}.png")
        if not path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg")
        if path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                ax.imshow(
                    img,
                    extent=[x0, x0 + cell_inches, y0, y0 + cell_inches],
                    aspect="equal",
                )
            except Exception:
                ax.add_patch(plt.Rectangle(
                    (x0, y0), cell_inches, cell_inches,
                    facecolor="#ccc", edgecolor="none",
                ))
        else:
            ax.add_patch(plt.Rectangle(
                (x0, y0), cell_inches, cell_inches,
                facecolor="#ccc", edgecolor="none",
            ))

        ax.text(
            x0 + cell_inches / 2, y0 + 0.15,
            f"{image_id:06d}",
            ha="center", va="bottom",
            fontsize=max(18, cell_inches * 6),
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8) if label_bg else None,
        )

    ax.set_title(
        f"Randomly sampled images of largest cluster (vehicle-{vehicle_id}, {n}/{total_size} samples)",
        pad=10,
    )
    _save_figure(fig, output_path, dpi=dpi, bbox_inches="tight", save_format=save_format)
    plt.close(fig)


def plot_per_vehicle_largest_grid(
    vehicle_reps: list[tuple[int, int, int]],
    images_dir: str,
    output_path: str,
    cell_inches: float = 2.5,
    dpi: int = 150,
    save_format: str = "png",
    label_bg: bool = True,
):
    """Render a 5-column grid of largest-cluster representatives, one per vehicle."""
    cols = 5
    n = len(vehicle_reps)
    rows = int(np.ceil(n / cols))

    fig, ax = plt.subplots(figsize=(cols * cell_inches, rows * cell_inches))
    ax.set_xlim(0, cols * cell_inches)
    ax.set_ylim(0, rows * cell_inches)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (vid, image_id, size) in enumerate(vehicle_reps):
        col = i % cols
        row = i // cols
        x0 = col * cell_inches
        y0 = (rows - 1 - row) * cell_inches

        img_path = path.join(images_dir, f"{image_id:06d}.png")
        if not path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg")
        if path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                ax.imshow(
                    img,
                    extent=[x0, x0 + cell_inches, y0, y0 + cell_inches],
                    aspect="equal",
                )
            except Exception:
                ax.add_patch(plt.Rectangle(
                    (x0, y0), cell_inches, cell_inches,
                    facecolor="#ccc", edgecolor="none",
                ))
        else:
            ax.add_patch(plt.Rectangle(
                (x0, y0), cell_inches, cell_inches,
                facecolor="#ccc", edgecolor="none",
            ))

        ax.text(
            x0 + cell_inches / 2, y0 + 0.15,
            f"vehicle-{vid}\n({size})",
            ha="center", va="bottom",
            fontsize=max(18, cell_inches * 8),
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8) if label_bg else None,
        )

    ax.set_title(f"Representative image of largest cluster per vehicle ({n} vehicles)", pad=10)
    _save_figure(fig, output_path, dpi=dpi, bbox_inches="tight", save_format=save_format)
    plt.close(fig)
