"""View training results from summary files."""

import argparse
import glob
import os
import sys

import joblib

from vipac_analysis.config import KWORDS


def find_summaries(output_dir, attribute=None):
    """Find summary joblib files, sorted by attribute name."""
    pattern = os.path.join(output_dir, '**/*_summary.joblib')
    paths = sorted(glob.glob(pattern, recursive=True))
    if attribute:
        paths = [p for p in paths if f'_{attribute}_' in os.path.basename(p)]
    return paths


def print_summary(path):
    """Print one summary file in a formatted table."""
    data = joblib.load(path)
    attribute = os.path.basename(path).split('_')[-2]  # e.g. 'modern' from ..._modern_summary.joblib
    test_accs = data['test_accs']
    test_accs_r = data['test_accs_r']
    min_val_losses = data['min_val_losses']

    print(f'\n=== {attribute} ===')
    print(f'  {"Rep":>3}  {"Test Acc":>8}  {"Test Acc (Rank)":>14}  {"Val Loss":>9}')

    for rep in range(len(test_accs)):
        val = f'{min_val_losses[rep]:>9.4f}' if rep < len(min_val_losses) else ''
        print(f'  {rep:>3}  {test_accs[rep]:>8.2%}  {test_accs_r[rep]:>14.2%}  {val}')

    # Mean line
    print(f'  {"":>3}  {"─" * 8:>8}  {"─" * 14:>14}  {"─" * 9:>9}')
    print(f'  {"Mean":>4}  {data["test_acc_mean"]:>8.2%}  {data["test_acc_r_mean"]:>14.2%}')
    print(f'  {"Std":>4}   {data["test_acc_std"]:>8.2%}  {data["test_acc_r_std"]:>14.2%}')


def main():
    p = argparse.ArgumentParser(description='Show RSS-CNN training results')
    p.add_argument('output_dir',
                   help='Directory containing *_summary.joblib files')
    p.add_argument('--attribute', choices=KWORDS,
                   help='Show only this attribute (default: all)')
    args = p.parse_args()

    paths = find_summaries(args.output_dir, args.attribute)
    if not paths:
        print(f'No summary files found in {args.output_dir}', file=sys.stderr)
        sys.exit(1)

    summaries = []

    for path in paths:
        try:
            print_summary(path)
            data = joblib.load(path)
            attr = os.path.basename(path).split('_')[-2]
            summaries.append((attr, data['test_acc_mean'], data['test_acc_r_mean']))
        except KeyError as e:
            print(f'  (skipped {os.path.basename(path)}: missing {e})',
                  file=sys.stderr)

    if not args.attribute and summaries:
        print(f'\n({len(summaries)} attribute(s) shown)')
        best_acc = max(summaries, key=lambda x: x[1])
        worst_acc = min(summaries, key=lambda x: x[1])
        best_r = max(summaries, key=lambda x: x[2])
        worst_r = min(summaries, key=lambda x: x[2])
        print(f'Best/Worst (Test Acc):      {best_acc[0]} ({best_acc[1]:.2%}) / {worst_acc[0]} ({worst_acc[1]:.2%})')
        print(f'Best/Worst (Test Acc Rank):  {best_r[0]} ({best_r[2]:.2%}) / {worst_r[0]} ({worst_r[2]:.2%})')


if __name__ == '__main__':
    main()
