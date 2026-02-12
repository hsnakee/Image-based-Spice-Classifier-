"""
Configuration file for Spice Image Classification Pipeline
Contains all hyperparameters and settings
"""

import torch
import random
import numpy as np

class Config:
    """Configuration class containing all hyperparameters"""
    
    # ==================== PATHS ====================
    # These will be set dynamically
    DATASET_PATH = None  # To be set by user
    OUTPUT_PATH = None   # To be set by user
    
    # ==================== REPRODUCIBILITY ====================
    RANDOM_SEED = 42
    
    # ==================== DATA SPLITS ====================
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    
    # ==================== MODEL CONFIGURATION ====================
    MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'convnext_tiny'
    PRETRAINED = True
    NUM_CLASSES = None  # Will be set automatically based on dataset
    
    # ==================== IMAGE SETTINGS ====================
    IMAGE_SIZE = 224  # Standard size for most pre-trained models
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # ==================== TRAINING HYPERPARAMETERS ====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # ==================== OPTIMIZER & SCHEDULER ====================
    OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
    SCHEDULER = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'cosine', 'step'
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # ==================== EARLY STOPPING ====================
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # ==================== TRAINING SETTINGS ====================
    USE_MIXED_PRECISION = True  # Automatic Mixed Precision (AMP)
    FREEZE_BACKBONE_EPOCHS = 5  # Epochs to freeze backbone before fine-tuning
    GRADIENT_CLIP_VALUE = 1.0
    
    # ==================== DATA LOADING ====================
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # ==================== AUGMENTATION SETTINGS ====================
    AUGMENTATION = {
        'random_horizontal_flip': 0.5,
        'random_vertical_flip': 0.3,
        'random_rotation': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'random_crop_scale': (0.8, 1.0),
        'random_crop_ratio': (0.9, 1.1)
    }
    
    # ==================== DEVICE ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== VISUALIZATION ====================
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    DPI = 150
    FIGSIZE = (12, 8)
    
    # ==================== INFERENCE ====================
    TOP_K = 5  # Top-K predictions to show
    CONFIDENCE_THRESHOLD = 0.5
    
    # ==================== CLASS IMBALANCE ====================
    USE_CLASS_WEIGHTS = True  # Automatically handle class imbalance
    
    # ==================== BONUS FEATURES ====================
    ENABLE_GRADCAM = True
    ENABLE_TENSORBOARD = True
    SAVE_MISCLASSIFIED = True
    
    @staticmethod
    def set_seed(seed=None):
        """Set random seeds for reproducibility"""
        if seed is None:
            seed = Config.RANDOM_SEED
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    @staticmethod
    def print_config():
        """Print current configuration"""
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        for key, value in vars(Config).items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30s}: {value}")
        print("=" * 70)
        
    @staticmethod
    def save_config(filepath):
        """Save configuration to file"""
        import json
        config_dict = {
            key: str(value) for key, value in vars(Config).items()
            if not key.startswith('_') and not callable(value)
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)


# Set seed when module is imported
Config.set_seed()
