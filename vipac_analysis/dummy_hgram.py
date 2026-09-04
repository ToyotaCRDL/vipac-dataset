"""Histogram of dummy-trial correctness per impression attribute.

Each participant answers 10 dummy trials (trial index 9,19,...,99) for
a single assigned attribute.  Per-participant correct counts range 0-10,
producing an 11-bin histogram for each attribute.

Expected answer alternates by trial position:
    expected = ((trial_index - 9) // 10) % 2
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vipac_analysis.config import KWORDS


def compute_attribute_correctness(df, ncorrect_thre=0):
    """Return a dict mapping attribute -> array of per-participant correct counts.

    Each participant belongs to one attribute and answers 10 dummy trials.
    The returned counts range 0-10.
    """
    dummy = df[df['dummy-flag'] == 1]

    # Per-participant correct count and attribute
    participant_correct = {}
    participant_attr = {}
    for _, row in dummy.iterrows():
        sid = int(row['participant-id'])
        attr = row['attribute']
        expected = ((int(row['trial-index']) - 9) // 10) % 2
        participant_attr[sid] = attr
        participant_correct[sid] = participant_correct.get(sid, 0) + (
            1 if int(row['chosen-side']) == expected else 0
        )

    # Filter by total correctness threshold
    if ncorrect_thre > 0:
        sids = {sid for sid, nc in participant_correct.items() if nc >= ncorrect_thre}
        participant_correct = {sid: nc for sid, nc in participant_correct.items() if sid in sids}
        participant_attr = {sid: attr for sid, attr in participant_attr.items() if sid in sids}

    # Group correct counts by attribute
    attr_counts = {kw: [] for kw in KWORDS}
    for sid, attr in participant_attr.items():
        attr_counts[attr].append(participant_correct[sid])
    return {kw: np.array(counts) for kw, counts in attr_counts.items()}


def plot_hgram(attr_counts, output_path, label):
    """Plot a 2x5 grid of histograms (0-10 bins), one per attribute."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()

    y_max = max(np.histogram(attr_counts[kw], bins=np.arange(12) - 0.5)[0].max()
                for kw in KWORDS)
    y_lim = int(np.ceil(y_max / 10) * 10 + 10)

    for i, kw in enumerate(KWORDS):
        ax = axes[i]
        counts = attr_counts[kw]
        bins = np.arange(12) - 0.5  # 11 bins for 0-10
        ax.hist(counts, bins=bins, rwidth=0.8, color='#2c7fb8')
        ax.set_xlabel('Correct')
        ax.set_ylabel('Participants' if i == 0 else '')
        ax.set_xticks(range(11))
        mean_val = counts.mean()
        ax.set_title(f'{kw}  (mean={mean_val:.2f})')
        ax.set_ylim(0, y_lim)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    svg_path = output_path.rsplit('.', 1)[0] + '.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')
    print(f'Saved {svg_path}')


def print_summary(attr_counts, label):
    """Print per-attribute statistics to stdout."""
    print(f'\n=== {label.upper()} Dummy Correctness per Attribute ===')
    print(f'{"Attribute":>14s}  {"Mean":>7s}  {"Median":>7s}  {"Std":>7s}  {"N":>7s}  {"0":>5s} {"1":>5s} {"2":>5s} {"3":>5s} {"4":>5s} {"5":>5s} {"6":>5s} {"7":>5s} {"8":>5s} {"9":>5s} {"10":>5s}')
    for kw in KWORDS:
        counts = attr_counts[kw]
        hist, _ = np.histogram(counts, bins=np.arange(12) - 0.5)
        hist_str = ''.join(f'{h:>5d}' for h in hist)
        print(f'{kw:>14s}  {counts.mean():>7.2f}  {np.median(counts):>7.1f}  {counts.std():>7.2f}  '
              f'{len(counts):>7d}{hist_str}')


def main():
    parser = argparse.ArgumentParser(
        description='Histogram of dummy-trial correctness per attribute')
    parser.add_argument('--responses-us', required=True,
                        help='Path to pairwise-responses-us.csv')
    parser.add_argument('--responses-jp', required=True,
                        help='Path to pairwise-responses-jp.csv')
    parser.add_argument('--output', required=True,
                        help='Output directory for histogram PNGs')
    parser.add_argument('--ncorrect-thre', type=int, default=0,
                        help='Filter participants by total dummy correctness (0=disable)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for path, label in [(args.responses_us, 'us'), (args.responses_jp, 'jp')]:
        df = pd.read_csv(path)
        attr_counts = compute_attribute_correctness(df, args.ncorrect_thre)
        print_summary(attr_counts, label)

        out_path = os.path.join(args.output, f'dummy_hgram_{label}.png')
        plot_hgram(attr_counts, out_path, label)


if __name__ == '__main__':
    main()
