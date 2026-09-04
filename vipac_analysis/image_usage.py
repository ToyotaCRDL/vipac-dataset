"""Image usage statistics from pairwise comparison responses.

Compute how many of the 120,000 images appear in participant-answered pairs,
the maximum single-image usage count, and enumerate all image pairs presented
two or more times (left-right mirror treated as the same pair).
"""

import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_responses(csv_path, exclude_dummy=True):
    """Load pairwise responses CSV, optionally filtering out dummy trials."""
    df = pd.read_csv(csv_path)
    if exclude_dummy:
        df = df[df['dummy-flag'] == 0].reset_index(drop=True)
    return df


def compute_image_usage(df, total_images):
    """Compute image usage statistics.

    Args:
        df: DataFrame with left-image-id and right-image-id columns.
        total_images: Total number of images in the dataset (120,000).

    Returns:
        Dict with usage_rate, max_usage, max_usage_image, repeated_pairs.
    """
    left_ids = df['left-image-id'].values
    right_ids = df['right-image-id'].values
    all_used = np.concatenate([left_ids, right_ids])

    # Distinct images used
    unique_images = len(np.unique(all_used))

    # Per-image usage counts
    image_counts = Counter(all_used)
    max_usage_image, max_usage = image_counts.most_common(1)[0]

    # Distribution: usage_count -> number of images with that count
    usage_dist = Counter(image_counts.values())

    # Canonical pairs: (min_id, max_id) to treat mirrors as identical
    pair_left = np.minimum(left_ids, right_ids)
    pair_right = np.maximum(left_ids, right_ids)
    pair_df = pd.DataFrame({'image_a': pair_left, 'image_b': pair_right})
    pair_counts = pair_df.value_counts().reset_index(name='count')

    repeated = pair_counts[pair_counts['count'] >= 2].sort_values('count', ascending=False)

    # Original (non-canonical) pair orientation counts
    orig_df = pd.DataFrame({'left': left_ids, 'right': right_ids})
    orig_counts = orig_df.value_counts().reset_index(name='count')

    # Collect contributor and orientation info for repeated pairs
    if len(repeated) > 0:
        repeat_cols = ['image_a', 'image_b']

        # Contributors: (orientation, participant_id, attribute) per appearance
        contrib_df = pair_df.copy()
        contrib_df['participant_id'] = df['participant-id'].values
        contrib_df['attribute'] = df['attribute'].values
        contrib_df['orig_left'] = left_ids
        contrib_df['orient'] = np.where(
            contrib_df['orig_left'] == contrib_df['image_a'], 'A-B', 'B-A'
        )
        contrib_df['contrib'] = (
            '(' + contrib_df['orient'] + ', ' +
            contrib_df['participant_id'].astype(str) + ', ' +
            contrib_df['attribute'] + ')'
        )
        filtered = contrib_df.merge(repeated[repeat_cols], on=repeat_cols, how='inner')
        contrib_agg = (
            filtered.groupby(repeat_cols)['contrib']
            .apply(lambda s: ';'.join(s.unique()))
            .reset_index(name='contributors')
        )
        repeated = repeated.merge(contrib_agg, on=repeat_cols, how='left')
        repeated['contributor_count'] = repeated['contributors'].str.count(';') + 1

        # Same-side: count of the dominant orientation
        canon = orig_counts.rename(
            columns={'left': 'image_a', 'right': 'image_b'})
        canon = canon[(canon['image_a'] < canon['image_b'])].reset_index(drop=True)
        mirror = orig_counts.rename(
            columns={'left': 'image_b', 'right': 'image_a'})
        mirror = mirror[(mirror['image_a'] < mirror['image_b'])].reset_index(drop=True)

        canon_max = (
            canon.merge(repeated[repeat_cols], on=repeat_cols, how='inner')
            .groupby(repeat_cols)['count'].max()
        )
        mirror_max = (
            mirror.merge(repeated[repeat_cols], on=repeat_cols, how='inner')
            .groupby(repeat_cols)['count'].max()
        )
        same_side = pd.DataFrame({
            'canon_count': canon_max,
            'mirror_count': mirror_max,
        }).fillna(0).reset_index()
        same_side['same_side'] = same_side[['canon_count', 'mirror_count']].max(axis=1).astype(int) - 1
        repeated = repeated.merge(
            same_side[repeat_cols + ['same_side']], on=repeat_cols, how='left')
        repeated['same_side'] = repeated['same_side'].astype(int)
    else:
        repeated['contributors'] = ''
        repeated['contributor_count'] = 0
        repeated['same_side'] = 0

    return {
        'total_images': total_images,
        'unique_used': unique_images,
        'usage_rate': unique_images / total_images,
        'total_trials': len(df),
        'max_usage': max_usage,
        'max_usage_image': int(max_usage_image),
        'usage_dist': usage_dist,
        'repeated_pairs': repeated,
    }


def print_summary(stats, label, top_n=50):
    """Print usage statistics to stdout."""
    repeated = stats['repeated_pairs']

    print(f'\n=== {label} Image Usage Statistics ===')
    print(f'Total trials analysed:      {stats["total_trials"]:>10,}')
    print(f'Total images in dataset:     {stats["total_images"]:>10,}')
    print(f'Images used in pairs:        {stats["unique_used"]:>10,}')
    print(f'Usage rate:                  {stats["usage_rate"]:>9.1%}')
    print(f'Max single-image usage:      {stats["max_usage"]:>10,}  (image {stats["max_usage_image"]:06d})')

    # Distribution table
    usage_dist = stats['usage_dist']
    print(f'\nUsage count distribution:')
    print(f'  {"Usage count":>12s}  {"Images":>12s}')
    print(f'  {"-"*12}  {"-"*12}')
    for count in sorted(usage_dist):
        print(f'  {count:>12,}  {usage_dist[count]:>12,}')
    print(f'  {"-"*12}  {"-"*12}')
    total_images_dist = sum(usage_dist.values())
    print(f'  {"Total":>12s}  {total_images_dist:>12,}')

    # Verification
    total_appearances = sum(c * n for c, n in usage_dist.items())
    expected_appearances = 2 * stats['total_trials']
    if total_images_dist != stats['unique_used']:
        print(f'  WARNING: distribution image sum ({total_images_dist:,}) '
              f'!= unique images ({stats["unique_used"]:,})')
    if total_appearances != expected_appearances:
        print(f'  WARNING: total appearances ({total_appearances:,}) '
              f'!= 2 * trials ({expected_appearances:,})')
    print(f'  Verified: {total_images_dist:,} images, '
          f'{total_appearances:,} total appearances (2 x {stats["total_trials"]:,} trials)')

    print(f'\nRepeated pairs (count >= 2): {len(repeated):>10,} pairs')
    if len(repeated) > 0:
        same_side_count = len(repeated[repeated['same_side'] >= 1])
        print(f'Same-side repeated:              {same_side_count:>10,} pairs')
        display = repeated.head(top_n)
        print(f'\n  {"Image A":>10s}  {"Image B":>10s}  {"Count":>8s}  {"SameSide":>8s}  {"Contributors":>64s}')
        print(f'  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*64}')
        for _, row in display.iterrows():
            print(f'  {int(row["image_a"]):>10d}  {int(row["image_b"]):>10d}  '
                  f'{int(row["count"]):>8,}  {int(row["same_side"]):>8,}  {row["contributors"]}')
        if len(repeated) > top_n:
            print(f'  ... ({len(repeated) - top_n} more)')


def plot_usage_distribution(stats, label, output_dir, no_title=False):
    """Plot usage count distribution as a bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    usage_dist = stats['usage_dist']
    max_count = stats['max_usage']

    counts = list(range(1, max_count + 1))
    nums = [usage_dist.get(c, 0) for c in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts, nums, color='#2c7fb8')
    ax.set_xlabel('Image usage count')
    ax.set_ylabel('Number of images')
    if not no_title:
        ax.set_title(f'{label} Image Usage Distribution', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    output_path = os.path.join(
        output_dir, f'image_usage_distribution_{label.lower().replace(" ", "_")}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    svg_path = output_path.rsplit('.', 1)[0] + '.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')
    print(f'Saved {svg_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Image usage statistics from pairwise comparison responses')
    parser.add_argument('--responses', nargs='+', required=True,
                        help='Path(s) to pairwise-responses CSV (US and/or JP)')
    parser.add_argument('--include-dummy', action='store_true',
                        help='Include dummy trials (default: exclude)')
    parser.add_argument('--output', default=None,
                        help='Output directory for figures '
                             '(default: output/image_usage/)')
    parser.add_argument('--no-title', action='store_true',
                        help='Suppress plot title')
    args = parser.parse_args()

    # Load and concatenate all response files
    dfs = []
    for path in args.responses:
        label = 'us' if 'us' in os.path.basename(path) else 'jp' if 'jp' in os.path.basename(path) else ''
        print(f'Loading {path} ...')
        dfs.append(load_responses(path, exclude_dummy=not args.include_dummy))

    df = pd.concat(dfs, ignore_index=True)
    total_images = 120_000

    # Derive label from files provided
    labels = []
    for path in args.responses:
        base = os.path.basename(path)
        if 'us' in base:
            labels.append('US')
        elif 'jp' in base:
            labels.append('JP')
    label = ' + '.join(labels) if labels else 'Combined'

    stats = compute_image_usage(df, total_images)
    print_summary(stats, label)

    fig_dir = args.output if args.output else 'output/image_usage/'
    plot_usage_distribution(stats, label, fig_dir, args.no_title)


if __name__ == '__main__':
    main()
