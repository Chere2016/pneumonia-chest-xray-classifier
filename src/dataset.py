import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def get_data_loaders(data_dir, batch_size=32, config=None):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Build augmentation list dynamically from config
    aug_list = [transforms.RandomHorizontalFlip()]

    if config and config.get('use_color_jitter', False):
        brightness = config.get('color_jitter_brightness', 0.2)
        contrast = config.get('color_jitter_contrast', 0.2)
        aug_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
        print(f"ColorJitter enabled (brightness={brightness}, contrast={contrast})")

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Data augmentation for training
    train_transform = transforms.Compose(aug_list)

    # Validation and testing transforms
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    # Imbalance handling: WeightedRandomSampler
    # Calculate weights for each class
    class_counts = np.bincount(train_dataset.targets)
    # Weight for each class is 1 / count
    class_weights = 1. / class_counts
    # Weight for each sample in the dataset
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Note: when using sampler, shuffle must be False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader, train_dataset.classes, class_counts
