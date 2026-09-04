# VIPAC Dataset -- Vehicle Impressions from Pairwise Assessments across Cultures

## Overview

A large-scale dataset of visual impressions of vehicle design, containing
120,000 AI-synthesized vehicle images and pairwise comparison judgments
collected from human participants in two countries.

- 120,000 images (256px resolution)
- 10,000 US participants (100 pairwise judgments each)
- 10,000 JP participants (100 pairwise judgments each)
- 10 impression attributes

Images were generated using Stable Diffusion XL + ControlNet, conditioned
on vehicle geometry to ensure structural consistency.

## Attributes

modern, sporty, sleek, elegant, stylish, dynamic, aerodynamic,
sophisticated, luxurious, aggressive

## Image Generation

Each image is generated from a combination of:

- 6 colors (white, black, silver, blue, red, yellow)
- 10 conditioning attributes (listed above)
- 20 vehicle IDs (0-19)
- 100 images per combination (90 generated at conditioning-scale 0.2
  and 10 at conditioning-scale 1.0)

Total: 6 x 10 x 20 x 100 = 120,000 images

Image IDs run from 0 to 119,999, stored as 6-digit zero-padded PNG files.

## File Structure

| File | Description |
|---|---|
| `images.zip` | 120,000 images at 256px resolution; extract to create `images/` |
| `metadata-images.csv` | Image metadata (120,001 lines with header) |
| `pairwise-responses-us.csv` | US pairwise judgments (1,000,001 lines with header) |
| `pairwise-responses-jp.csv` | JP pairwise judgments (1,000,001 lines with header) |
| `participants-us.csv` | US participant demographics (10,001 lines with header) |
| `participants-jp.csv` | JP participant demographics (10,001 lines with header) |
| `dummy-pairs.csv` | Dummy trial image pairs (101 lines with header) |

`images.zip` contains the 120,000 PNG files under an `images/` prefix.
Extract it inside `VIPAC/` to create the `images/` directory:

    cd VIPAC
    unzip images.zip

After extraction, images are available at `VIPAC/images/` as 6-digit
zero-padded PNG file names (e.g. `VIPAC/images/000000.png`).

## Column Definitions

### metadata-images.csv (120,000 rows)

- **image-id** — Unique image identifier (0-119,999)
- **vehicle-id** — Vehicle model ID (0-19)
- **color** — Vehicle color (white, black, silver, blue, red, yellow)
- **attribute** — Conditioning attribute (10 attributes above)
- **seed** — Generator seed used for the image (observed range 1000-2912). The base seed is set by the attribute (modern = 1000, sporty = 1100, sleek = 1200, elegant = 1300, stylish = 1400, dynamic = 1500, aerodynamic = 1600, sophisticated = 1700, luxurious = 1800, aggressive = 1900) and images within a (color, attribute, vehicle) combination use sequential offsets from the base. A small number of images regenerated in a supplementary batch use the corresponding 2000-series base seeds (e.g. 2000 for modern).
- **conditioning-scale** — SDXL conditioning scale (0.2 or 1.0; 108,000 images at 0.2, 12,000 at 1.0)

### pairwise-responses-us.csv / pairwise-responses-jp.csv (1,000,000 rows each)

- **participant-id** — Participant identifier (0-9,999)
- **trial-index** — Trial number within the session (0-99)
- **left-image-id** — Image ID shown on the left side
- **right-image-id** — Image ID shown on the right side
- **attribute** — The impression attribute being judged
- **chosen-side** — Which image was chosen
  - 0 = left image chosen
  - 1 = right image chosen
- **dummy-flag** — Whether this trial is a quality-check dummy trial
  - 0 = real trial
  - 1 = dummy trial

Each participant completes 100 trials:

- 90 real comparison trials
- 10 dummy trials (inserted at trial indices 9, 19, 29, ..., 99)
  with predetermined correct answers for reliability screening.

### participants-us.csv (10,000 rows)

- **participant-id** — Participant identifier (0-9,999)
- **age** — Age in years (free-text field; a small number of entries are non-numeric or implausible and should be cleaned before analysis)
- **gender** — Self-reported gender (Female, Male, Non-binary, Other, Prefer not to answer)
- **ethnicity** — Self-reported ethnicity (fixed list of 14 categories: American Indian or Alaska native, Asian Indian, Black, Chinese, Filipino, Japanese, Korean, Native Hawaiian, Other Asian, Other Pacific Islander, Other race, Vietnamese, White, Prefer not to answer)
- **residence-country** — Current country of residence (free-form; US responses often give state or city names; 5 entries missing)
- **longest-residence-country** — Country of longest residence (free-form; 5 entries missing)
- **income** — Income level (ordinal code 1-9)
- **education** — Education level (ordinal code 1-14)
- **attribute** — Assigned impression attribute
- **work-time-seconds** — Time in seconds to complete the task

### participants-jp.csv (10,000 rows)

- **participant-id** — Participant identifier (0-9,999)
- **age** — Age in years (missing for 100 participants)
- **gender** — Self-reported gender: male, female, other, prefer not to answer
- **ethnicity** — Self-reported ethnicity: asian, hispanic/latino, black/african american, white, other, prefer not to answer
- **residence-country** — Current country of residence: JP, other, prefer not to answer
- **longest-residence-country** — Country of longest residence
- **income** — Household income level (ordinal code 1-12)
- **education** — Education level (ordinal code 1-9)
- **attribute** — Assigned impression attribute
- **work-time-seconds** — (Empty -- not measured for JP participants)
- **interest-in-automobiles** — Interest in automobiles (ordinal code 1-5)
- **owns-car** — Whether the participant personally selected and purchased a car (ordinal code 1-3)

### dummy-pairs.csv (100 rows)

- **attribute** — Impression attribute
- **image-id-high** — Image with a high predicted impression score (selected near the top of the preliminary model ranking)
- **image-id-low** — Image with a low predicted impression score (selected near the bottom of the preliminary model ranking)

Contains 10 dummy image pairs per attribute (100 total), derived from
preliminary RSS-CNN model rankings. Used to construct quality-check trials.

## Code Value Definitions

### US Income (participants-us.csv)

| Code | Value |
|---|---|
| 1 | 0 USD |
| 2 | 1-9,999 USD |
| 3 | 10,000-24,999 USD |
| 4 | 25,000-49,999 USD |
| 5 | 50,000-74,999 USD |
| 6 | 75,000-99,999 USD |
| 7 | 100,000-149,999 USD |
| 8 | 150,000 USD or more |
| 9 | Prefer not to answer |

### US Education (participants-us.csv)

| Code | Value |
|---|---|
| 1 | No schooling completed |
| 2 | Nursery school |
| 3 | Grades 1 through 11 |
| 4 | 12th grade, no diploma |
| 5 | GED or alternative credential |
| 6 | Some college credit, less than 1 year |
| 7 | 1 or more years of college credit, no degree |
| 8 | Associate degree |
| 9 | Regular high school diploma |
| 10 | Bachelor degree |
| 11 | Master degree |
| 12 | Doctorate degree |
| 13 | Professional degree beyond bachelor |
| 14 | Prefer not to answer |

### JP Income (participants-jp.csv)

| Code | Value |
|---|---|
| 1 | Under 2 million JPY |
| 2 | 2-3 million JPY |
| 3 | 3-4 million JPY |
| 4 | 4-5 million JPY |
| 5 | 5-6 million JPY |
| 6 | 6-7 million JPY |
| 7 | 7-8 million JPY |
| 8 | 8-10 million JPY |
| 9 | 10-15 million JPY |
| 10 | 15-20 million JPY |
| 11 | 20 million JPY or more |
| 12 | Don't know / Prefer not to answer |

### JP Education (participants-jp.csv)

| Code | Value |
|---|---|
| 1 | Junior high school graduate |
| 2 | High school graduate |
| 3 | Vocational school graduate |
| 4 | Junior college graduate |
| 5 | University graduate |
| 6 | Master's degree |
| 7 | Doctoral degree |
| 8 | None of the above |
| 9 | Prefer not to answer |

### JP Interest in Automobiles (participants-jp.csv)

| Code | Value |
|---|---|
| 1 | Very interested |
| 2 | Somewhat interested |
| 3 | Neither interested nor uninterested |
| 4 | Not very interested |
| 5 | Not interested at all |

### JP Car Ownership (participants-jp.csv)

Question: "Do you have a car that you personally selected and purchased?"

| Code | Value |
|---|---|
| 1 | Yes |
| 2 | No |
| 3 | Prefer not to answer |

## Trial Structure

Each participant is assigned one of 10 impression attributes and completes
100 pairwise comparison trials for that attribute.

- 90 real trials comparing two vehicle images
- 10 dummy trials at fixed positions (trial indices 9, 19, 29, 39, 49,
  59, 69, 79, 89, 99) where the correct answer is predetermined

Dummy trial responses are used to screen participant quality. Participants
who fail too many dummy checks can be excluded using the `--ncorrect-thre`
option in the training pipeline.

Dummy trial answer pattern (trial index 9 -> expect chosen-side 0,
trial index 19 -> expect chosen-side 1, alternating).

## License

This dataset is released under CC BY 4.0.
