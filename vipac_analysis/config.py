"""Shared constants for VIPAC dataset."""

KWORDS = [
    'modern', 'sporty', 'sleek', 'elegant', 'stylish',
    'dynamic', 'aerodynamic', 'sophisticated', 'luxurious', 'aggressive',
]

COLORS = ['white', 'black', 'silver', 'blue', 'red', 'yellow']
VIDS = list(range(20))
SEEDS = range(100)


def make_image_id(color_idx, kword_idx, vid_idx, seed):
    """Compute image ID from component indices.

    Formula: icolor*20000 + ikword*2000 + ivid*100 + iseed
    """
    return color_idx * 20000 + kword_idx * 2000 + vid_idx * 100 + seed


def image_id_str(img_id):
    """Format image ID as 6-digit zero-padded string."""
    return f"{img_id:06d}"


US_INCOME_MAP = {
    1: '0USD',
    2: '1-9999USD',
    3: '10000-24999USD',
    4: '25000-49999USD',
    5: '50000-74999USD',
    6: '75000-99999USD',
    7: '100000-149999USD',
    8: '150000USDover',
    9: 'Prefer not to answer',
}

US_EDUCATION_MAP = {
    1: 'No schooling completed',
    2: 'Nursery school',
    3: 'Grades 1 through 11',
    4: '12th grade - no diploma',
    5: 'GED or alternative credential',
    6: 'Some college credit but less than 1 year of college',
    7: '1 or more years of college credit no degree',
    8: 'Associates degree',
    9: 'Regular high school diploma',
    10: 'Bachelor degree',
    11: 'Master degree',
    12: 'Doctorate degree',
    13: 'Professional degree beyond bachelor degree',
    14: 'Prefer not to answer',
}


def all_image_ids():
    """Generate all 120,000 image IDs."""
    for ic in range(len(COLORS)):
        for ik in range(len(KWORDS)):
            for iv in range(len(VIDS)):
                for is_ in SEEDS:
                    yield make_image_id(ic, ik, iv, is_)
