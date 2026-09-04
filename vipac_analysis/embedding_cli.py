"""CLI entry point for VIPAC image embedding analysis.

Usage:
    python -m vipac_analysis.embedding_cli --extract
    python -m vipac_analysis.embedding_cli --hierarchical
    python -m vipac_analysis.embedding_cli --hierarchical --per-vehicle-hierarchical
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd


def _run_global_hierarchical(
    embeddings_512,
    index_df,
    images_dir,
    output_dir,
    subsample_size,
    n_clusters,
    method,
    save_format: str = "png",
    label_bg: bool = True,
):
    """Run hierarchical clustering on all images globally (512D CLIP embeddings)."""
    from vipac_analysis.hierarchical_clustering import (
        stratified_subsample,
        build_linkage,
        cut_dendrogram,
        assign_all_to_clusters,
        find_representatives,
        relabel_by_size,
        sample_cluster_images,
        print_cluster_summary,
    )
    from vipac_analysis.visualize_hierarchical import (
        plot_cluster_samples,
        plot_dendrogram_with_images,
        plot_tree_layout,
        plot_representative_grid,
    )

    hier_dir = os.path.join(output_dir, "hierarchical")
    os.makedirs(hier_dir, exist_ok=True)

    n_per_group = max(1, subsample_size // len(index_df["vehicle_id"].unique()))
    print(f"  Stratified subsample: {subsample_size} total ({n_per_group} per vehicle)")
    subsample_indices = stratified_subsample(
        index_df, groupby="vehicle_id", n_per_group=n_per_group
    )

    subsample_data = embeddings_512[subsample_indices]
    full_data = embeddings_512

    print(f"  Building linkage ({len(subsample_data)} points, 512D, method={method})")
    Z = build_linkage(subsample_data, method=method)
    subsample_labels = cut_dendrogram(Z, n_clusters)
    print(f"  Cutting into {n_clusters} clusters")

    all_labels = assign_all_to_clusters(full_data, subsample_data, subsample_labels)
    representatives = find_representatives(embeddings_512, index_df, all_labels)
    all_labels, representatives = relabel_by_size(all_labels, representatives)
    print(f"  Found {len(representatives)} representative images")
    print_cluster_summary(all_labels, title=f"Global (K={n_clusters})")

    # Sample and render cluster image grids
    cluster_samples = sample_cluster_images(all_labels, index_df)
    plot_cluster_samples(
        cluster_samples, images_dir,
        os.path.join(hier_dir, "cluster_samples"),
        save_format=save_format,
        label_bg=label_bg,
    )

    # Save cluster assignments
    assign_df = index_df.copy()
    assign_df["hierarchical_cluster"] = all_labels
    assign_df.to_csv(os.path.join(hier_dir, "assignments.csv"), index=False)
    print(f"  Saved assignments.csv")

    # Save clustering results for later visualization
    results = {
        "Z": Z,
        "subsample_indices": subsample_indices,
        "subsample_labels": subsample_labels,
        "all_labels": all_labels,
        "representatives": representatives,
        "cluster_samples": cluster_samples,
        "n_clusters": n_clusters,
    }
    with open(os.path.join(hier_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved results.pkl")

    # Visualize
    plot_dendrogram_with_images(
        Z, representatives, subsample_indices, index_df, images_dir,
        os.path.join(hier_dir, f"dendrogram_K{n_clusters}.png"),
        n_clusters,
        save_format=save_format,
    )
    plot_tree_layout(
        Z, representatives, subsample_labels,
        images_dir,
        os.path.join(hier_dir, f"tree_layout_K{n_clusters}.png"),
        n_clusters,
        save_format=save_format,
    )
    plot_representative_grid(
        representatives, images_dir,
        os.path.join(hier_dir, f"grid_K{n_clusters}.png"),
        n_clusters,
        save_format=save_format,
        label_bg=label_bg,
    )


def _run_per_vehicle_hierarchical(
    embeddings_512,
    index_df,
    images_dir,
    output_dir,
    subsample_size,
    n_clusters,
    method,
    save_format: str = "png",
    label_bg: bool = True,
):
    """Run hierarchical clustering separately for each vehicle (512D CLIP embeddings)."""
    from vipac_analysis.hierarchical_clustering import (
        build_linkage,
        cut_dendrogram,
        assign_all_to_clusters,
        find_representatives,
        relabel_by_size,
        sample_cluster_images,
        print_cluster_summary,
    )
    from vipac_analysis.visualize_hierarchical import (
        plot_cluster_samples,
        plot_dendrogram_with_images,
        plot_largest_cluster_grid,
        plot_tree_layout,
        plot_representative_grid,
    )

    hier_dir = os.path.join(output_dir, "hierarchical_per_vehicle")
    os.makedirs(hier_dir, exist_ok=True)

    n_per_vehicle = subsample_size // 20
    all_assignments = []
    all_largest_reps = []
    all_results = {}

    for vid in sorted(index_df["vehicle_id"].unique()):
        mask = index_df["vehicle_id"].values == vid
        v_512 = embeddings_512[mask]
        v_index = index_df[mask].reset_index(drop=True)

        if len(v_512) < 10:
            print(f"  Skipping vehicle {vid}: too few points")
            continue

        # Subsample within vehicle
        n_sub = min(n_per_vehicle, len(v_512))
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(v_512), size=n_sub, replace=False)

        subsample_data = v_512[sub_idx]
        full_data = v_512

        print(f"  Vehicle {vid}: linkage on {len(subsample_data)} points (512D) → {n_clusters} clusters")
        Z = build_linkage(subsample_data, method=method)
        subsample_labels = cut_dendrogram(Z, n_clusters)

        # Assign all vehicle points to cluster centroids
        v_labels = assign_all_to_clusters(full_data, subsample_data, subsample_labels)

        # Find representatives in 512D space
        representatives = find_representatives(v_512, v_index, v_labels)
        v_labels, representatives = relabel_by_size(v_labels, representatives)
        print(f"  Vehicle {vid}: {len(representatives)} clusters found")
        print_cluster_summary(v_labels, title=f"Vehicle {vid} (K={n_clusters})")

        # Collect largest cluster representative
        largest_rep = next((r for r in representatives if r["cluster_id"] == 0), None)
        if largest_rep:
            all_largest_reps.append((vid, largest_rep["image_id"], largest_rep["size"]))

        # Sample and render cluster image grids per vehicle
        v_cluster_samples = sample_cluster_images(v_labels, v_index)
        print(f"  Vehicle {vid}: {len(v_cluster_samples)} cluster samples generated")
        all_results[vid] = {
            "Z": Z,
            "sub_idx": sub_idx,
            "subsample_labels": subsample_labels,
            "v_labels": v_labels,
            "representatives": representatives,
            "v_cluster_samples": v_cluster_samples,
            "n_clusters": n_clusters,
        }
        plot_cluster_samples(
            v_cluster_samples, images_dir,
            os.path.join(hier_dir, "cluster_samples", f"vehicle-{vid}"),
            save_format=save_format,
            label_bg=label_bg,
        )

        # Render largest cluster grid
        try:
            largest = next(cs for cs in v_cluster_samples if cs["cluster_id"] == 0)
            output_path = os.path.join(hier_dir, f"vehicle-{vid}_largest_cluster.png")
            plot_largest_cluster_grid(
                largest["image_ids"], largest["total_size"],
                images_dir, output_path,
                vehicle_id=vid,
                save_format=save_format,
                label_bg=label_bg,
            )
        except Exception as e:
            print(f"  WARNING: Failed to render largest cluster for vehicle {vid}: {e}")

        # Save assignments
        assign_df = index_df[mask].copy()
        assign_df["hierarchical_cluster"] = v_labels
        all_assignments.append(assign_df)

        # Visualize
        plot_dendrogram_with_images(
            Z, representatives, sub_idx, v_index, images_dir,
            os.path.join(hier_dir, f"vehicle-{vid}_dendrogram_K{n_clusters}.png"),
            n_clusters,
            save_format=save_format,
        )
        plot_tree_layout(
            Z, representatives, subsample_labels,
            images_dir,
            os.path.join(hier_dir, f"vehicle-{vid}_tree_K{n_clusters}.png"),
            n_clusters,
            save_format=save_format,
        )
        plot_representative_grid(
            representatives, images_dir,
            os.path.join(hier_dir, f"vehicle-{vid}_grid_K{n_clusters}.png"),
            n_clusters,
            save_format=save_format,
            vehicle_id=vid,
        )

    if all_assignments:
        pd.concat(all_assignments, ignore_index=True).to_csv(
            os.path.join(hier_dir, "assignments.csv"), index=False
        )
        print(f"  Saved per-vehicle assignments.csv")

    if all_largest_reps:
        from vipac_analysis.visualize_hierarchical import plot_per_vehicle_largest_grid
        plot_per_vehicle_largest_grid(
            all_largest_reps, images_dir,
            os.path.join(hier_dir, "largest_cluster.png"),
            save_format=save_format,
            label_bg=label_bg,
        )

    # Save clustering results for later visualization
    if all_results:
        with open(os.path.join(hier_dir, "results.pkl"), "wb") as f:
            pickle.dump(all_results, f)
        print(f"  Saved results.pkl")


def main():
    parser = argparse.ArgumentParser(
        description="VIPAC Image Embedding Analysis Pipeline"
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Run embedding extraction",
    )
    parser.add_argument(
        "--images", default="VIPAC/images/",
        help="Path to image directory",
    )
    parser.add_argument(
        "--metadata", default="VIPAC/metadata-images.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--output", default="output/image_quality/",
        help="Output directory for embedding files and figures "
             "(default: output/image_quality/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="GPU batch size for extraction (default: 64)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip extraction if embeddings already exist",
    )
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical clustering and visualization (on 512D CLIP embeddings)",
    )
    parser.add_argument(
        "--hierarchical-subsample", type=int, default=3000,
        help="Subsample size for linkage (default: 3000)",
    )
    parser.add_argument(
        "--hierarchical-clusters", type=int, default=20,
        help="Number of flat clusters to cut from the dendrogram (default: 20)",
    )
    parser.add_argument(
        "--hierarchical-method",
        choices=["average", "complete", "ward", "single"],
        default="average",
        help="Linkage method for hierarchical clustering (default: average)",
    )
    parser.add_argument(
        "--per-vehicle-hierarchical", action="store_true",
        help="Run hierarchical clustering per vehicle instead of globally",
    )
    parser.add_argument(
        "--svg", action="store_true",
        help="Also save hierarchical clustering figures as SVG (editable text, embedded images)",
    )
    parser.add_argument(
        "--load-clustering", action="store_true",
        help="Load clustering results from results.pkl and regenerate images without recomputing",
    )
    parser.add_argument(
        "--font", default="Liberation Sans",
        help="Font family for figures (default: Liberation Sans)",
    )
    parser.add_argument(
        "--font-size", type=int, default=18,
        help="Default font size in points (default: 18)",
    )
    parser.add_argument(
        "--no-label-bg", action="store_true",
        help="Omit white background rectangles behind text labels",
    )

    args = parser.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isabs(args.images):
        args.images = os.path.join(base, args.images)
    if not os.path.isabs(args.metadata):
        args.metadata = os.path.join(base, args.metadata)
    if not os.path.isabs(args.output):
        args.output = os.path.join(base, args.output)

    import matplotlib
    matplotlib.rcParams.update({
        "font.family": args.font,
        "font.sans-serif": [args.font, "DejaVu Sans", "sans-serif"],
        "font.size": args.font_size,
    })

    if not (args.extract or args.hierarchical or args.load_clustering):
        parser.print_help()
        return

    os.makedirs(args.output, exist_ok=True)

    embeddings_path = os.path.join(args.output, "embeddings_clip_512.npy")
    index_path = os.path.join(args.output, "embeddings_clip_512_index.csv")

    if args.extract:
        print("=== Extract CLIP embeddings ===")
        from vipac_analysis.extract_embeddings import extract_embeddings

        extract_embeddings(
            image_dir=args.images,
            metadata_path=args.metadata,
            output_path=embeddings_path,
            index_path=index_path,
            batch_size=args.batch_size,
            resume=args.resume,
        )

    if args.hierarchical and not args.load_clustering:
        print("\n=== Hierarchical clustering ===")
        save_format = "both" if args.svg else "png"
        label_bg = not args.no_label_bg

        index_df = pd.read_csv(index_path)
        embeddings_512 = np.load(embeddings_path)

        if args.per_vehicle_hierarchical:
            _run_per_vehicle_hierarchical(
                embeddings_512=embeddings_512,
                index_df=index_df,
                images_dir=args.images,
                output_dir=args.output,
                subsample_size=args.hierarchical_subsample,
                n_clusters=args.hierarchical_clusters,
                method=args.hierarchical_method,
                save_format=save_format,
                label_bg=label_bg,
            )
        else:
            _run_global_hierarchical(
                embeddings_512=embeddings_512,
                index_df=index_df,
                images_dir=args.images,
                output_dir=args.output,
                subsample_size=args.hierarchical_subsample,
                n_clusters=args.hierarchical_clusters,
                method=args.hierarchical_method,
                save_format=save_format,
                label_bg=label_bg,
            )

    if args.load_clustering:
        print("\n=== Reload clustering and regenerate images ===")
        save_format = "both" if args.svg else "png"
        label_bg = not args.no_label_bg
        from vipac_analysis.hierarchical_clustering import print_cluster_summary
        from vipac_analysis.visualize_hierarchical import (
            plot_cluster_samples,
            plot_dendrogram_with_images,
            plot_largest_cluster_grid,
            plot_per_vehicle_largest_grid,
            plot_representative_grid,
            plot_tree_layout,
        )

        # Try global and per-vehicle paths
        global_pkl = os.path.join(args.output, "hierarchical", "results.pkl")
        per_vehicle_pkl = os.path.join(args.output, "hierarchical_per_vehicle", "results.pkl")

        if os.path.exists(per_vehicle_pkl):
            with open(per_vehicle_pkl, "rb") as f:
                all_results = pickle.load(f)
            print(f"  Loaded per-vehicle results for {len(all_results)} vehicles")
            index_df = pd.read_csv(index_path)
            hier_dir = os.path.join(args.output, "hierarchical_per_vehicle")
            all_largest_reps = []
            for vid, res in sorted(all_results.items()):
                mask = index_df["vehicle_id"].values == vid
                v_index = index_df[mask].reset_index(drop=True)
                Z = res["Z"]
                sub_idx = res["sub_idx"]
                subsample_labels = res["subsample_labels"]
                v_labels = res["v_labels"]
                representatives = res["representatives"]
                v_cluster_samples = res["v_cluster_samples"]
                n_clusters = res["n_clusters"]
                print_cluster_summary(v_labels, title=f"Vehicle {vid} (K={n_clusters})")
                plot_dendrogram_with_images(
                    Z, representatives, sub_idx, v_index, args.images,
                    os.path.join(hier_dir, f"vehicle-{vid}_dendrogram_K{n_clusters}.png"),
                    n_clusters,
                    save_format=save_format,
                )
                plot_tree_layout(
                    Z, representatives, subsample_labels,
                    args.images,
                    os.path.join(hier_dir, f"vehicle-{vid}_tree_K{n_clusters}.png"),
                    n_clusters,
                    save_format=save_format,
                )
                plot_representative_grid(
                    representatives, args.images,
                    os.path.join(hier_dir, f"vehicle-{vid}_grid_K{n_clusters}.png"),
                    n_clusters,
                    save_format=save_format,
                    label_bg=label_bg,
                    vehicle_id=vid,
                )
                plot_cluster_samples(
                    v_cluster_samples, args.images,
                    os.path.join(hier_dir, "cluster_samples", f"vehicle-{vid}"),
                    save_format=save_format,
                    label_bg=label_bg,
                )
                largest = next((cs for cs in v_cluster_samples if cs["cluster_id"] == 0), None)
                largest_rep = next((r for r in representatives if r["cluster_id"] == 0), None)
                if largest:
                    if largest_rep:
                        all_largest_reps.append((vid, largest_rep["image_id"], largest_rep["size"]))
                    plot_largest_cluster_grid(
                        largest["image_ids"], largest["total_size"],
                        args.images,
                        os.path.join(hier_dir, f"vehicle-{vid}_largest_cluster.png"),
                        vehicle_id=vid,
                        save_format=save_format,
                        label_bg=label_bg,
                    )
            if all_largest_reps:
                plot_per_vehicle_largest_grid(
                    all_largest_reps, args.images,
                    os.path.join(hier_dir, "largest_cluster.png"),
                    save_format=save_format,
                    label_bg=label_bg,
                )
            print(f"  Regenerated per-vehicle images")
            total_images = sum(r["v_labels"].size for r in all_results.values())
            print(f"  Total images across vehicles: {total_images} (expected: {len(index_df)})")

        elif os.path.exists(global_pkl):
            with open(global_pkl, "rb") as f:
                res = pickle.load(f)
            print(f"  Loaded global results ({res['n_clusters']} clusters)")
            index_df = pd.read_csv(index_path)
            hier_dir = os.path.join(args.output, "hierarchical")
            Z = res["Z"]
            subsample_indices = res["subsample_indices"]
            subsample_labels = res["subsample_labels"]
            all_labels = res["all_labels"]
            representatives = res["representatives"]
            cluster_samples = res["cluster_samples"]
            n_clusters = res["n_clusters"]
            print_cluster_summary(all_labels, title=f"Global loaded (K={n_clusters})")
            plot_dendrogram_with_images(
                Z, representatives, subsample_indices, index_df, args.images,
                os.path.join(hier_dir, f"dendrogram_K{n_clusters}.png"),
                n_clusters,
                save_format=save_format,
            )
            plot_tree_layout(
                Z, representatives, subsample_labels,
                args.images,
                os.path.join(hier_dir, f"tree_layout_K{n_clusters}.png"),
                n_clusters,
                save_format=save_format,
            )
            plot_representative_grid(
                representatives, args.images,
                os.path.join(hier_dir, f"grid_K{n_clusters}.png"),
                n_clusters,
                save_format=save_format,
                label_bg=label_bg,
            )
            plot_cluster_samples(
                cluster_samples, args.images,
                os.path.join(hier_dir, "cluster_samples"),
                save_format=save_format,
                label_bg=label_bg,
            )
            print(f"  Regenerated global images")
        else:
            print("  ERROR: No results.pkl found in hierarchical or hierarchical_per_vehicle")

    print("\nDone.")


if __name__ == "__main__":
    main()
