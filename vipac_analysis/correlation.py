"""Compute impression score correlation matrix and plot its heatmap."""

import argparse
import joblib
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from vipac_analysis.config import KWORDS


def load_scores(scores_dir, attribute):
    """Load averaged scores for one attribute."""
    joblib_path = os.path.join(scores_dir, f'{attribute}.joblib')
    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f'Score file not found: {joblib_path}')
    data = joblib.load(joblib_path)
    return data['lscored']


def compute_correlation_matrix(scores_dir):
    """Build 10x10 correlation matrix from per-attribute scores."""
    all_ids = None
    score_dicts = {}

    for kw in KWORDS:
        scores = load_scores(scores_dir, kw)
        ids = set(scores.keys())

        if all_ids is None:
            all_ids = ids
        else:
            all_ids &= ids

        score_dicts[kw] = scores

    # Align all attributes to common image IDs
    common_ids = sorted(all_ids)
    print(f'Common image IDs across all attributes: {len(common_ids)}')

    matrix = np.zeros((len(KWORDS), len(common_ids)))
    for i, kw in enumerate(KWORDS):
        matrix[i] = [score_dicts[kw][id_] for id_ in common_ids]

    # Pearson correlation
    corr = np.corrcoef(matrix)
    return corr, KWORDS


def save_matrix_csv(corr, path):
    """Save the full (no-header) correlation matrix CSV to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for row in corr:
            f.write(','.join(f'{v:.4f}' for v in row) + '\n')
    print(f'Saved full matrix to {path}')


def save_heatmap(corr_matrix, keywords, output_dir, label='us', fontsize=10):
    """Save the correlation matrix heatmap as PNG and SVG."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(keywords)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(keywords, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticklabels(keywords, fontsize=fontsize)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=fontsize,
                    color='white' if corr_matrix[i, j] > 0.8 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'{label.upper()} Impression Score Correlation Matrix')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'corr_heatmap_{label}.png'), dpi=150)
    fig.savefig(os.path.join(output_dir, f'corr_heatmap_{label}.svg'))
    plt.close(fig)
    print(f'Saved heatmap to {output_dir}/corr_heatmap_{label}.png')
    print(f'Saved heatmap to {output_dir}/corr_heatmap_{label}.svg')


def main():
    parser = argparse.ArgumentParser(
        description='Compute correlation matrix from RSS-CNN scores')
    parser.add_argument('--scores-dir', required=True,
                        help='Directory containing {attribute}.joblib files')
    parser.add_argument('--output', required=True,
                        help='Output path for the full correlation matrix CSV')
    parser.add_argument('--font-size', type=int, default=10,
                        help='Font size for heatmap axis labels (default: 10)')
    args = parser.parse_args()

    corr, keywords = compute_correlation_matrix(args.scores_dir)

    # Print matrix
    print('\nCorrelation matrix:')
    header = f'{"":>14s}' + ''.join(f'{kw:>12s}' for kw in keywords)
    print(header)
    for i, kw in enumerate(keywords):
        row = f'{kw:>14s}' + ''.join(f'{corr[i, j]:12.4f}' for j in range(len(keywords)))
        print(row)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    label = os.path.basename(os.path.normpath(args.scores_dir))

    save_matrix_csv(corr, args.output)

    # Find max/min off-diagonal pairs
    max_corr, min_corr = -np.inf, np.inf
    max_pair, min_pair = None, None
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            if corr[i, j] > max_corr:
                max_corr = corr[i, j]
                max_pair = (keywords[i], keywords[j])
            if corr[i, j] < min_corr:
                min_corr = corr[i, j]
                min_pair = (keywords[i], keywords[j])

    print(f'\nMax correlation: {max_pair[0]} & {max_pair[1]} = {max_corr:.4f}')
    print(f'Min correlation: {min_pair[0]} & {min_pair[1]} = {min_corr:.4f}')

    save_heatmap(corr, keywords, output_dir, label=label, fontsize=args.font_size)


if __name__ == '__main__':
    main()
