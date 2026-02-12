"""
Dataset utilities for spice image classification
Handles data loading, preprocessing, and splitting
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import Config


class SpiceDataset(Dataset):
    """Custom Dataset for Spice Images"""
    
    def __init__(self, image_paths, labels, transform=None, class_names=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform (callable, optional): Optional transform to be applied on images
            class_names (list, optional): List of class names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(mode='train'):
    """
    Get image transforms for different modes
    
    Args:
        mode (str): 'train', 'val', or 'test'
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(
                Config.IMAGE_SIZE, 
                scale=Config.AUGMENTATION['random_crop_scale'],
                ratio=Config.AUGMENTATION['random_crop_ratio']
            ),
            transforms.RandomHorizontalFlip(p=Config.AUGMENTATION['random_horizontal_flip']),
            transforms.RandomVerticalFlip(p=Config.AUGMENTATION['random_vertical_flip']),
            transforms.RandomRotation(Config.AUGMENTATION['random_rotation']),
            transforms.ColorJitter(
                brightness=Config.AUGMENTATION['color_jitter']['brightness'],
                contrast=Config.AUGMENTATION['color_jitter']['contrast'],
                saturation=Config.AUGMENTATION['color_jitter']['saturation'],
                hue=Config.AUGMENTATION['color_jitter']['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])


def load_dataset(dataset_path):
    """
    Load dataset from folder-per-class structure
    
    Args:
        dataset_path (str): Path to dataset root directory
    
    Returns:
        tuple: (image_paths, labels, class_names, class_to_idx)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Get all class folders
    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {dataset_path}")
    
    class_names = [folder.name for folder in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Collect all images
    image_paths = []
    labels = []
    
    for class_folder in class_folders:
        class_idx = class_to_idx[class_folder.name]
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(class_folder.glob(ext))
            image_files.extend(class_folder.glob(ext.upper()))
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"Total images found: {len(image_paths)}")
    
    # Print class distribution
    class_counts = Counter(labels)
    print("\nClass distribution:")
    for class_name, class_idx in class_to_idx.items():
        count = class_counts[class_idx]
        print(f"  {class_name:25s}: {count:5d} images")
    
    return image_paths, labels, class_names, class_to_idx


def create_data_splits(image_paths, labels, stratify=True):
    """
    Split data into train, validation, and test sets
    
    Args:
        image_paths (list): List of image paths
        labels (list): List of labels
        stratify (bool): Whether to use stratified split
    
    Returns:
        dict: Dictionary containing splits
    """
    # First split: train+val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, 
        labels,
        test_size=Config.TEST_RATIO,
        random_state=Config.RANDOM_SEED,
        stratify=labels if stratify else None
    )
    
    # Second split: train and val
    val_ratio_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_ratio_adjusted,
        random_state=Config.RANDOM_SEED,
        stratify=train_val_labels if stratify else None
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    return {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels}
    }


def create_dataloaders(splits, class_names):
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        splits (dict): Dictionary containing data splits
        class_names (list): List of class names
    
    Returns:
        dict: Dictionary containing DataLoaders
    """
    # Create datasets
    train_dataset = SpiceDataset(
        splits['train']['paths'],
        splits['train']['labels'],
        transform=get_transforms('train'),
        class_names=class_names
    )
    
    val_dataset = SpiceDataset(
        splits['val']['paths'],
        splits['val']['labels'],
        transform=get_transforms('val'),
        class_names=class_names
    )
    
    test_dataset = SpiceDataset(
        splits['test']['paths'],
        splits['test']['labels'],
        transform=get_transforms('test'),
        class_names=class_names
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def calculate_class_weights(labels):
    """
    Calculate class weights for handling class imbalance
    
    Args:
        labels (list): List of labels
    
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Calculate weights: inverse of class frequency
    weights = []
    for i in range(num_classes):
        count = class_counts[i]
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def visualize_class_distribution(labels, class_names, output_path):
    """
    Visualize class distribution
    
    Args:
        labels (list): List of labels
        class_names (list): List of class names
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    # Count samples per class
    class_counts = Counter(labels)
    counts = [class_counts[i] for i in range(len(class_names))]
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(range(len(class_names)), counts, color=colors, alpha=0.8, edgecolor='black')
    
    plt.xlabel('Spice Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {output_path}")


def visualize_augmented_samples(dataloader, class_names, output_path, num_samples=16):
    """
    Visualize augmented training samples
    
    Args:
        dataloader (DataLoader): Training dataloader
        class_names (list): List of class names
        output_path (str): Path to save the plot
        num_samples (int): Number of samples to visualize
    """
    # Get a batch
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Denormalize images
    mean = torch.tensor(Config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.NORMALIZE_STD).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx in range(min(num_samples, 16)):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        label = labels[idx].item()
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'{class_names[label]}', fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle('Sample Augmented Training Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Augmented samples visualization saved to {output_path}")


def get_sample_images_per_class(image_paths, labels, class_names, num_per_class=1):
    """
    Get sample images for each class
    
    Args:
        image_paths (list): List of image paths
        labels (list): List of labels
        class_names (list): List of class names
        num_per_class (int): Number of samples per class
    
    Returns:
        dict: Dictionary mapping class names to image paths
    """
    samples = {name: [] for name in class_names}
    
    for path, label in zip(image_paths, labels):
        class_name = class_names[label]
        if len(samples[class_name]) < num_per_class:
            samples[class_name].append(path)
    
    return samples
