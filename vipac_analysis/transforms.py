"""Image transforms for training and evaluation.

Matches the original rsscnn_20250116.py transform:
    Resize((224, 224)) + Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
No data augmentation is applied — the original code does not use any.
"""

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

eval_transform = train_transform
