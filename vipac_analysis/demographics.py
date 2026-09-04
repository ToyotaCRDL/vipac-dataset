#!/usr/bin/env python3
"""Visualize demographic composition ratios for US and JP participants.

Generates horizontal bar charts (percentage composition) for age, gender,
ethnicity, residence country, longest residence country, income, and education.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vipac_analysis.config import US_INCOME_MAP, US_EDUCATION_MAP


# === Constants ===

AGE_BINS = [17, 24, 34, 44, 54, 64, 74, 100]
AGE_LABELS = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+', 'Missing']

DEMOGRAPHIC_FIELDS = [
    'age_group', 'gender', 'ethnicity', 'residence_country',
    'longest_residence_country', 'income', 'education',
]

FIELD_TITLES = {
    'age_group': 'Age',
    'gender': 'Gender',
    'ethnicity': 'Ethnicity',
    'residence_country': 'Residence Country',
    'longest_residence_country': 'Longest Residence Country',
    'income': 'Income',
    'education': 'Education',
}

LABEL_DISPLAY = {
    'Prefer not to answer': 'Prefer not\nto answer',
    'Hispanic/Latino': 'Hispanic/\nLatino',
    'Indigenous/Pacific': 'Indigenous/\nPacific',
}

US_STATE_CODES = frozenset({
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY',
    'AS', 'GU', 'MP', 'PR', 'VI',  # territories
    'AA', 'AE', 'AP',  # military
})

US_STATE_NAMES = frozenset({
    'CALIFORNIA', 'NEW YORK', 'TEXAS', 'FLORIDA', 'ILLINOIS', 'PENNSYLVANIA',
    'OHIO', 'GEORGIA', 'NORTH CAROLINA', 'MICHIGAN', 'NEW JERSEY',
})

COUNTRY_CODE_NAMES = {
    'AU': 'Australia', 'JP': 'Japan', 'NZ': 'New Zealand', 'MX': 'Mexico',
    'GB': 'United Kingdom', 'IE': 'Ireland', 'DE': 'Germany', 'FR': 'France',
    'IT': 'Italy', 'ES': 'Spain', 'PT': 'Portugal', 'NL': 'Netherlands',
    'BE': 'Belgium', 'CH': 'Switzerland', 'AT': 'Austria', 'SE': 'Sweden',
    'NO': 'Norway', 'DK': 'Denmark', 'FI': 'Finland', 'PL': 'Poland',
    'CZ': 'Czech Republic', 'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria',
    'HR': 'Croatia', 'SI': 'Slovenia', 'SK': 'Slovakia', 'LT': 'Lithuania',
    'LV': 'Latvia', 'EE': 'Estonia', 'RU': 'Russia', 'UA': 'Ukraine',
    'TR': 'Turkey', 'IL': 'Israel', 'ZA': 'South Africa', 'KE': 'Kenya',
    'NG': 'Nigeria', 'GH': 'Ghana', 'EG': 'Egypt', 'MA': 'Morocco',
    'TN': 'Tunisia', 'DZ': 'Algeria', 'AO': 'Angola', 'MZ': 'Mozambique',
    'ZW': 'Zimbabwe', 'UG': 'Uganda', 'TZ': 'Tanzania', 'RW': 'Rwanda',
    'SN': 'Senegal', 'CI': 'Ivory Coast', 'BF': 'Burkina Faso', 'ML': 'Mali',
    'NE': 'Niger', 'TD': 'Chad', 'CF': 'Central African Republic',
    'CM': 'Cameroon', 'CD': 'DR Congo', 'CG': 'Congo', 'GA': 'Gabon',
    'BJ': 'Benin', 'GW': 'Guinea-Bissau', 'GN': 'Guinea', 'SL': 'Sierra Leone',
    'LR': 'Liberia', 'GM': 'Gambia', 'MR': 'Mauritania', 'SZ': 'Eswatini',
    'BW': 'Botswana', 'NA': 'Namibia', 'KM': 'Comoros', 'SC': 'Seychelles',
    'MU': 'Mauritius', 'MG': 'Madagascar', 'DJ': 'Djibouti', 'SO': 'Somalia',
    'ER': 'Eritrea', 'ET': 'Ethiopia', 'BI': 'Burundi',
    'KR': 'South Korea', 'KP': 'North Korea', 'CN': 'China', 'MN': 'Mongolia',
    'IN': 'India', 'PK': 'Pakistan', 'BD': 'Bangladesh', 'LK': 'Sri Lanka',
    'NP': 'Nepal', 'BT': 'Bhutan', 'MV': 'Maldives',
    'ID': 'Indonesia', 'MY': 'Malaysia', 'TH': 'Thailand', 'VN': 'Vietnam',
    'PH': 'Philippines', 'SG': 'Singapore', 'MM': 'Myanmar', 'KH': 'Cambodia',
    'LA': 'Laos', 'TW': 'Taiwan', 'HK': 'Hong Kong',
    'BN': 'Brunei', 'TL': 'Timor-Leste', 'FJ': 'Fiji', 'PG': 'Papua New Guinea',
    'BR': 'Brazil', 'AR': 'Argentina', 'CL': 'Chile', 'CO': 'Colombia',
    'PE': 'Peru', 'VE': 'Venezuela', 'EC': 'Ecuador', 'BO': 'Bolivia',
    'PY': 'Paraguay', 'UY': 'Uruguay', 'GY': 'Guyana', 'SR': 'Suriname',
    'CR': 'Costa Rica', 'PA': 'Panama', 'GT': 'Guatemala', 'HN': 'Honduras',
    'SV': 'El Salvador', 'NI': 'Nicaragua', 'CU': 'Cuba', 'DO': 'Dominican Republic',
    'JM': 'Jamaica', 'TT': 'Trinidad and Tobago', 'BS': 'Bahamas', 'BZ': 'Belize',
    'AF': 'Afghanistan', 'IQ': 'Iraq', 'IR': 'Iran', 'SY': 'Syria', 'JO': 'Jordan',
    'LB': 'Lebanon', 'KW': 'Kuwait', 'QA': 'Qatar', 'AE': 'UAE', 'BH': 'Bahrain',
    'OM': 'Oman', 'SA': 'Saudi Arabia', 'YE': 'Yemen',
    'AX': 'Aland Islands', 'IS': 'Iceland', 'KY': 'Cayman Islands',
    'CR': 'Costa Rica', 'QA': 'Qatar',
}

US_VARIANT_KEYWORDS = frozenset({'US', 'USA', 'UNITED STATES', 'UNITED STATES OF AMERICA'})

ANSWER_KEYWORDS = frozenset({'NO-ANSWER', 'NO ANSWER', 'PREFER NOT TO ANSWER', 'N/A', 'NA', 'PREFER NOT TO ANSW'})


# === Data Cleaning Functions ===

def clean_age(series):
    numeric = pd.to_numeric(series, errors='coerce')
    valid_mask = (numeric >= 18) & (numeric <= 100)
    result = pd.Series(index=series.index, dtype=object)
    result[valid_mask] = pd.cut(numeric[valid_mask], bins=AGE_BINS,
                                labels=AGE_LABELS[:-1], right=False).astype(str)
    result[~valid_mask] = 'Missing'
    return result


def normalize_gender(series):
    mapping = {
        'male': 'Male', 'female': 'Female',
        'non-binary': 'Non-binary', 'non_binary': 'Non-binary', 'nonbinary': 'Non-binary',
        'prefer not to answer': 'Prefer not to answer', 'other': 'Other',
    }
    return series.astype(str).str.strip().str.lower().map(mapping).dropna()


def normalize_ethnicity(series, region):
    if region == 'us':
        mapping = {
            'White': 'White', 'Black': 'Black',
            'Asian Indian': 'Asian', 'Chinese': 'Asian', 'Filipino': 'Asian',
            'Japanese': 'Asian', 'Korean': 'Asian', 'Other Asian': 'Asian',
            'Vietnamese': 'Asian',
            'American Indian or Alaska native': 'Indigenous/Pacific',
            'Other Pacific Islander': 'Indigenous/Pacific',
            'Native Hawaiian': 'Indigenous/Pacific',
            'Prefer not to answer': 'Prefer not to answer',
            'Other race': 'Other',
        }
    else:
        mapping = {
            'asian': 'Asian', 'white': 'White',
            'black/african american': 'Black',
            'hispanic/latino': 'Hispanic/Latino',
            'other': 'Other',
            'prefer not to answer': 'Prefer not to answer',
        }
    return series.astype(str).str.strip().map(mapping).dropna()


def normalize_country(series):
    def map_value(val):
        s = str(val).strip().upper()
        if not s or s == 'NAN' or s in ANSWER_KEYWORDS:
            return np.nan
        if s in US_VARIANT_KEYWORDS:
            return 'US'
        if len(s) == 2 and s in US_STATE_CODES:
            return 'US'
        if s in US_STATE_NAMES:
            return 'US'
        if len(s) == 2 and s in COUNTRY_CODE_NAMES:
            return COUNTRY_CODE_NAMES[s]
        if len(s) == 2:
            return s  # unknown 2-letter code
        return s  # full name

    result = series.map(map_value)
    result = result.dropna()

    # Collapse rare categories (< 0.3% of total) to "Other"
    counts = result.value_counts()
    total = counts.sum()
    threshold = max(3, total * 0.003)
    rare = counts[counts < threshold].index.tolist()
    result = result.replace(rare, 'Other')

    return result


def map_income(series, region):
    if region == 'us':
        return series.astype(int).map(US_INCOME_MAP).dropna()
    return series.astype(pd.Int64Dtype()).dropna().apply(lambda x: f'Income {x}')


def map_education(series, region):
    if region == 'us':
        return series.astype(int).map(US_EDUCATION_MAP).dropna()
    return series.astype(pd.Int64Dtype()).dropna().apply(lambda x: f'Education {x}')


# === High-Level Pipeline ===

def load_and_clean(path_us, path_jp):
    result = {}
    for path, region in [(path_us, 'us'), (path_jp, 'jp')]:
        df = pd.read_csv(path)
        cleaned = pd.DataFrame({
            'age_group': clean_age(df['age']),
            'gender': normalize_gender(df['gender']),
            'ethnicity': normalize_ethnicity(df['ethnicity'], region),
            'residence_country': normalize_country(df['residence-country']),
            'longest_residence_country': normalize_country(df['longest-residence-country']),
            'income': map_income(df['income'], region),
            'education': map_education(df['education'], region),
        })
        result[region] = cleaned
    return result


# === Visualization Functions ===

def plot_single_field(df, field, ax, fontsize=10):
    """Render one demographic field as a horizontal bar chart on the given axes."""
    counts = df[field].value_counts()
    total = counts.sum()
    pcts = counts / total * 100

    if field == 'age_group':
        # Reverse age order so youngest appears at top of barh plot
        pcts = pcts.reindex([l for l in AGE_LABELS if l in pcts.index][::-1])
    else:
        # Pin "Prefer not to answer" to bottom of barh (index 0 = bottom)
        prefer = None
        if 'Prefer not to answer' in pcts.index:
            prefer = pcts['Prefer not to answer']
            pcts = pcts.drop('Prefer not to answer')
        pcts = pcts.sort_values(ascending=True)
        if prefer is not None:
            pcts = pd.concat([pd.Series([prefer], index=['Prefer not to answer']), pcts])

    ax.barh(range(len(pcts)), pcts.values, color='#2c7fb8', height=0.6)
    ax.set_yticks(range(len(pcts)))
    ax.set_yticklabels([LABEL_DISPLAY.get(cat, cat) for cat in pcts.index],
                       fontsize=fontsize)
    ax.set_xlim(0, max(pcts.max() * 1.15, 110))
    ax.grid(axis='x', alpha=0.3)

    for i, (cat, pct) in enumerate(pcts.items()):
        ax.text(pct + max(pcts.max() * 0.02, 1), i, f'{pct:.1f}%',
                va='center', fontsize=max(6, fontsize - 2), color='#2c7fb8')

    ax.set_title(f'{FIELD_TITLES.get(field, field)}  (n={total})', fontsize=fontsize + 2)


def plot_demographics(df, output_path, label, fields=None, no_title=False, fontsize=10):
    """Generate a grid figure for a single country."""
    if fields is None:
        fields = DEMOGRAPHIC_FIELDS
    n = len(fields)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, field in enumerate(fields):
        plot_single_field(df, field, axes[i], fontsize=fontsize)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    if not no_title:
        plt.suptitle(f'{label.upper()} Participant Demographics', fontsize=fontsize + 4, y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    svg_path = output_path.rsplit('.', 1)[0] + '.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')
    print(f'Saved {svg_path}')


def plot_comparison(us_df, jp_df, output_path, fields=None, no_title=False, fontsize=10):
    """Generate a side-by-side comparison figure."""
    if fields is None:
        fields = DEMOGRAPHIC_FIELDS
    fig, axes = plt.subplots(len(fields), 2, figsize=(10, 3.5 * len(fields)))
    if len(fields) == 1:
        axes = axes.reshape(1, -1)

    for i, field in enumerate(fields):
        plot_single_field(us_df, field, axes[i, 0], fontsize=fontsize)
        plot_single_field(jp_df, field, axes[i, 1], fontsize=fontsize)

        axes[i, 0].set_ylabel('')
        axes[i, 1].set_ylabel('')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    if not no_title:
        plt.suptitle('US vs JP Demographics Comparison', fontsize=fontsize + 4, y=0.995)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    svg_path = output_path.rsplit('.', 1)[0] + '.svg'
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')
    print(f'Saved {svg_path}')


# === Summary Output ===

def print_summary(df, label, fields=None):
    print(f'\n=== {label.upper()} Demographics Composition ===')
    if fields is None:
        fields = DEMOGRAPHIC_FIELDS
    for field in fields:
        counts = df[field].value_counts()
        if field == 'age_group':
            counts = counts.reindex([l for l in AGE_LABELS if l in counts.index])
        elif 'Prefer not to answer' in counts.index:
            prefer = counts['Prefer not to answer']
            counts = counts.drop('Prefer not to answer').sort_values(ascending=False)
            counts = pd.concat([counts, pd.Series([prefer], index=['Prefer not to answer'])])
        total = counts.sum()
        print(f'\n  {FIELD_TITLES.get(field, field)}  (n={total})')
        print(f'  {"Category":<30s} {"Count":>6s} {"Pct":>6s}')
        for cat, cnt in counts.items():
            print(f'  {str(cat):<30s} {cnt:>6d} {cnt/total*100:>5.1f}%')
    print()


# === CLI Entry Point ===

def main():
    parser = argparse.ArgumentParser(
        description='Visualize demographic composition for US and JP participants')
    parser.add_argument('--participants-us', required=True,
                        help='Path to participants-us.csv')
    parser.add_argument('--participants-jp', required=True,
                        help='Path to participants-jp.csv')
    parser.add_argument('--output', required=True,
                        help='Output directory for PNG figures')
    parser.add_argument('--fields', nargs='+', choices=DEMOGRAPHIC_FIELDS,
                        default=DEMOGRAPHIC_FIELDS,
                        help='Fields to visualize (default: all). '
                             f'Choices: {", ".join(DEMOGRAPHIC_FIELDS)}')
    parser.add_argument('--no-title', action='store_true',
                        help='Suppress the figure-wide title')
    parser.add_argument('--font-size', type=int, default=10,
                        help='Font size for labels (default: 10)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cleaned = load_and_clean(args.participants_us, args.participants_jp)

    for region in ('us', 'jp'):
        print_summary(cleaned[region], region, args.fields)
        out_path = os.path.join(args.output, f'demographics_{region}.png')
        plot_demographics(cleaned[region], out_path, region, args.fields, args.no_title,
                          fontsize=args.font_size)

    plot_comparison(cleaned['us'], cleaned['jp'],
                    os.path.join(args.output, 'demographics_comparison.png'),
                    args.fields, args.no_title, fontsize=args.font_size)


if __name__ == '__main__':
    main()
