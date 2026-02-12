"""
Model architectures and utilities for spice image classification
Supports multiple pre-trained models with transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights, EfficientNet_B0_Weights, 
    EfficientNet_B3_Weights, ConvNeXt_Tiny_Weights
)
from config import Config


class SpiceClassifier(nn.Module):
    """Transfer learning model for spice classification"""
    
    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): Name of the model architecture
            num_classes (int): Number of output classes
            pretrained (bool): Use pretrained weights
        """
        super(SpiceClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone
        if model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        print(f"Initialized {model_name} with {num_classes} classes")
        
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        if self.model_name == 'resnet50':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name in ['efficientnet_b0', 'efficientnet_b3']:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.model_name == 'convnext_tiny':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        print("Backbone frozen, classifier unfrozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning")
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes):
    """
    Create model based on config
    
    Args:
        num_classes (int): Number of output classes
    
    Returns:
        SpiceClassifier: Model instance
    """
    model = SpiceClassifier(
        model_name=Config.MODEL_NAME,
        num_classes=num_classes,
        pretrained=Config.PRETRAINED
    )
    
    model = model.to(Config.DEVICE)
    
    print(f"Model loaded on {Config.DEVICE}")
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   val_accuracy, filepath, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'model_name': Config.MODEL_NAME,
        'num_classes': Config.NUM_CLASSES,
        'is_best': is_best
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        print(f"✓ Best model saved to {filepath}")
    else:
        print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance (optional)
        scheduler: Scheduler instance (optional)
    
    Returns:
        dict: Checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return checkpoint


def load_model_for_inference(model_path, num_classes):
    """
    Load model for inference
    
    Args:
        model_path: Path to saved model
        num_classes: Number of classes
    
    Returns:
        model: Loaded model in eval mode
    """
    # Create model
    model = SpiceClassifier(
        model_name=Config.MODEL_NAME,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How long to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(f"Architecture: {Config.MODEL_NAME}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    print(f"Device: {Config.DEVICE}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*70 + "\n")
