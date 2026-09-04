"""Dataset classes for VIPAC pairwise comparison training and scoring."""

import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PairedComparisonDataset(Dataset):
    """Load training pairs from VIPAC pairwise-responses CSV.

    Reads pairwise-responses-{us|jp}.csv, filters by attribute,
    optionally filters participants by dummy-question correctness,
    removes dummy trials and duplicate image pairs.

    Column conventions (chosen-side, dummy-flag):
        chosen-side: 0 = left image chosen, 1 = right image chosen
        dummy-flag:  0 = real trial, 1 = dummy (quality-check) trial
    """

    def __init__(self, csv_path, image_dir, attribute=None,
                 transform=None, ncorrect_thre=0, participants_path=None,
                 age_min=0, age_max=130):
        df = pd.read_csv(csv_path)
        if attribute is not None:
            df = df[df['attribute'] == attribute]

        # Age filtering (matches original rsscnn_20250116.py age filter)
        if participants_path is not None:
            df = PairedComparisonDataset._apply_age_filter(
                df, participants_path, age_min, age_max)

        # Compute per-participant dummy correctness
        dummy_rows = df[df['dummy-flag'] == 1]
        participant_ncorrect = {}
        for sid in df['participant-id'].unique():
            subj_dummies = dummy_rows[dummy_rows['participant-id'] == sid].sort_values('trial-index')
            ncorrect = 0
            for _, row in subj_dummies.iterrows():
                i = (int(row['trial-index']) - 9) // 10
                expected_label = i % 2  # alternating 0, 1, 0, 1, ...
                if int(row['chosen-side']) == expected_label:
                    ncorrect += 1
            participant_ncorrect[sid] = ncorrect

        if ncorrect_thre > 0:
            qualified_sids = {sid for sid, nc in participant_ncorrect.items()
                              if nc >= ncorrect_thre}
            df = df[df['participant-id'].isin(qualified_sids)]
            n_qualified = len(qualified_sids)
            n_total = len(participant_ncorrect)
            print(f'Participant filtering: {n_qualified}/{n_total} ({n_qualified/n_total:.1%}) qualified '
                  f'(ncorrect >= {ncorrect_thre})')

        # Remove dummy trials
        df = df[df['dummy-flag'] == 0]

        # Deduplicate image pairs (keep first occurrence)
        # Canonical pair (smaller ID first) is used ONLY for dedup key.
        # chosen-side and image IDs are stored in original CSV order (display order).
        seen = set()
        left_ids, right_ids, labels = [], [], []
        for _, row in df.iterrows():
            l = int(row['left-image-id'])
            r = int(row['right-image-id'])
            lbl = int(row['chosen-side'])
            # Canonical pair for dedup (order-independent)
            canon = (min(l, r), max(l, r))
            if canon not in seen:
                seen.add(canon)
                left_ids.append(l)
                right_ids.append(r)
                labels.append(lbl)

        self.left_ids = left_ids
        self.right_ids = right_ids
        self.labels = labels
        self.image_dir = image_dir.rstrip('/')
        self.transform = transform

    @staticmethod
    def _apply_age_filter(df, participants_path, age_min, age_max):
        """Filter out participants with age outside [age_min, age_max].

        Participants with non-numeric age (e.g., empty/unanswered) are kept.
        """
        p = pd.read_csv(participants_path)
        ages = pd.to_numeric(p['age'], errors='coerce')
        in_range = (ages >= age_min) & (ages <= age_max)
        valid_sids = set(p[in_range | ages.isna()]['participant-id'])
        n_before = len(df['participant-id'].unique())
        df = df[df['participant-id'].isin(valid_sids)]
        n_after = len(df['participant-id'].unique())
        if n_before != n_after:
            print(f'Age filtering ({age_min}-{age_max}): '
                  f'{n_after}/{n_before} participants ({n_before - n_after} excluded)')
        return df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        l_id, r_id = self.left_ids[idx], self.right_ids[idx]
        label = self.labels[idx]

        left_img = Image.open(f"{self.image_dir}/{l_id:06d}.png").convert('RGB')
        right_img = Image.open(f"{self.image_dir}/{r_id:06d}.png").convert('RGB')

        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img, label


class ImageDataset(Dataset):
    """Load all 120,000 images for scoring.

    Reads image IDs from metadata-images.csv or generates them
    from the config formula.
    """

    def __init__(self, image_dir, metadata_path=None, transform=None):
        if metadata_path:
            df = pd.read_csv(metadata_path)
            self.img_ids = df['image-id'].tolist()
        else:
            from vipac_analysis.config import all_image_ids
            self.img_ids = list(all_image_ids())

        self.image_dir = image_dir.rstrip('/')
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = Image.open(f"{self.image_dir}/{img_id:06d}.png").convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_id
