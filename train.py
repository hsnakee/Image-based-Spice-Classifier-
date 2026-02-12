"""
Main training script for spice image classification
Handles complete training pipeline with evaluation and visualization
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from config import Config
from dataset_utils import (
    load_dataset, create_data_splits, create_dataloaders,
    calculate_class_weights, visualize_class_distribution,
    visualize_augmented_samples
)
from model import (
    create_model, save_checkpoint, EarlyStopping, print_model_summary
)
from evaluation import (
    evaluate_model, plot_training_history, plot_confusion_matrix,
    plot_roc_curves, save_metrics, plot_per_class_metrics,
    find_misclassified_samples, visualize_misclassified
)
from tensorboard_logger import create_tensorboard_logger, log_training_epoch


class Trainer:
    """Training manager for spice classification"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 class_names, output_dir, class_weights=None):
        """
        Args:
            model: Model instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            class_names: List of class names
            output_dir: Directory to save outputs
            class_weights: Class weights for imbalanced data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logger
        if Config.ENABLE_TENSORBOARD:
            self.tb_logger = create_tensorboard_logger(
                self.output_dir, 
                f'{Config.MODEL_NAME}_spice_classification'
            )
            print("✓ TensorBoard logging enabled")
        else:
            self.tb_logger = None
        
        # Loss function
        if class_weights is not None and Config.USE_CLASS_WEIGHTS:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE))
            print("Using weighted CrossEntropyLoss for class imbalance")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if Config.USE_MIXED_PRECISION else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        ) if Config.EARLY_STOPPING else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def _create_optimizer(self):
        """Create optimizer based on config"""
        if Config.OPTIMIZER == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        elif Config.OPTIMIZER == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        elif Config.OPTIMIZER == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=Config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        if Config.SCHEDULER == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=Config.SCHEDULER_FACTOR,
                patience=Config.SCHEDULER_PATIENCE
            )
        elif Config.SCHEDULER == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=Config.NUM_EPOCHS,
                eta_min=1e-6
            )
        elif Config.SCHEDULER == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if Config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if Config.GRADIENT_CLIP_VALUE > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        Config.GRADIENT_CLIP_VALUE
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if Config.GRADIENT_CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        Config.GRADIENT_CLIP_VALUE
                    )
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]  ')
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Complete training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        # Freeze backbone for initial epochs
        if Config.FREEZE_BACKBONE_EPOCHS > 0:
            print(f"\nFreezing backbone for first {Config.FREEZE_BACKBONE_EPOCHS} epochs")
            self.model.freeze_backbone()
        
        for epoch in range(Config.NUM_EPOCHS):
            # Unfreeze backbone after specified epochs
            if epoch == Config.FREEZE_BACKBONE_EPOCHS and Config.FREEZE_BACKBONE_EPOCHS > 0:
                print("\nUnfreezing backbone for fine-tuning")
                self.model.unfreeze_backbone()
                # Optionally reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = Config.LEARNING_RATE * 0.1
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # TensorBoard logging
            if self.tb_logger:
                log_training_epoch(self.tb_logger, epoch + 1, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'lr': current_lr
                })
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, train_loss, val_loss, val_acc,
                    self.output_dir / 'best_model.pt',
                    is_best=True
                )
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, train_loss, val_loss, val_acc,
                    self.output_dir / f'checkpoint_epoch_{epoch+1}.pt'
                )
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_loss, epoch + 1)
                if self.early_stopping.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best epoch was {self.best_epoch} with val_acc: {self.best_val_acc:.2f}%")
                    break
            
            print("-" * 70)
        
        # Training completed
        training_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total training time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        # Log hyperparameters to TensorBoard
        if self.tb_logger:
            hparams = {
                'model': Config.MODEL_NAME,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'epochs': Config.NUM_EPOCHS,
                'optimizer': Config.OPTIMIZER
            }
            final_metrics = {
                'best_val_acc': self.best_val_acc,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss
            }
            self.tb_logger.log_hyperparameters(hparams, final_metrics)
            self.tb_logger.close()
            print("✓ TensorBoard logs saved")
        
        # Save final model
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            Config.NUM_EPOCHS, train_loss, val_loss, val_acc,
            self.output_dir / 'final_model.pt'
        )
        
        # Save training history
        self._save_training_history()
        
        return self.history
    
    def _save_training_history(self):
        """Save training history to CSV"""
        df = pd.DataFrame(self.history)
        df['epoch'] = range(1, len(df) + 1)
        df = df[['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']]
        
        csv_path = self.output_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        print(f"Training history saved to {csv_path}")


def main(dataset_path, output_path):
    """
    Main training function
    
    Args:
        dataset_path (str): Path to dataset directory
        output_path (str): Path to output directory
    """
    # Set paths in config
    Config.DATASET_PATH = dataset_path
    Config.OUTPUT_PATH = output_path
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    Config.print_config()
    
    # Save configuration
    Config.save_config(output_dir / 'config.json')
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    # Load dataset
    image_paths, labels, class_names, class_to_idx = load_dataset(dataset_path)
    
    # Set number of classes in config
    Config.NUM_CLASSES = len(class_names)
    
    # Save class names
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump({
            'class_names': class_names,
            'class_to_idx': class_to_idx
        }, f, indent=4)
    
    # Visualize class distribution
    visualize_class_distribution(
        labels, class_names, 
        output_dir / 'class_distribution.png'
    )
    
    # Create data splits
    splits = create_data_splits(image_paths, labels, stratify=True)
    
    # Create dataloaders
    dataloaders = create_dataloaders(splits, class_names)
    
    # Visualize augmented samples
    visualize_augmented_samples(
        dataloaders['train'], class_names,
        output_dir / 'augmented_samples.png'
    )
    
    # Calculate class weights
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(splits['train']['labels'])
        print(f"\nClass weights: {class_weights.numpy()}")
    
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    # Create model
    model = create_model(Config.NUM_CLASSES)
    print_model_summary(model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        class_names=class_names,
        output_dir=output_dir,
        class_weights=class_weights
    )
    
    # Train model
    history = trainer.train()
    
    # Plot training history
    plot_training_history(history, output_dir / 'training_curves.png')
    
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Load best model for evaluation
    from model import load_checkpoint
    load_checkpoint(output_dir / 'best_model.pt', model)
    
    # Evaluate on test set
    metrics = evaluate_model(
        model=model,
        test_loader=dataloaders['test'],
        class_names=class_names,
        output_dir=output_dir
    )
    
    # Save metrics
    save_metrics(metrics, output_dir / 'test_metrics.json')
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix.png'
    )
    
    # Plot ROC curves
    plot_roc_curves(
        metrics['all_labels'],
        metrics['all_probs'],
        class_names,
        output_dir / 'roc_curves.png'
    )
    
    # Plot per-class metrics
    plot_per_class_metrics(
        metrics['classification_report'],
        class_names,
        output_dir / 'per_class_metrics.png'
    )
    
    # Analyze misclassified samples
    if Config.SAVE_MISCLASSIFIED:
        print("\n" + "="*70)
        print("ANALYZING MISCLASSIFIED SAMPLES")
        print("="*70)
        
        misclassified = find_misclassified_samples(
            model,
            dataloaders['test'],
            class_names,
            num_samples=20
        )
        
        if len(misclassified) > 0:
            visualize_misclassified(
                misclassified,
                output_dir / 'misclassified_samples.png'
            )
            print(f"Found and visualized {len(misclassified)} misclassified samples")
        else:
            print("No misclassified samples found (perfect accuracy!)")
    
    # Generate Grad-CAM visualizations
    if Config.ENABLE_GRADCAM:
        print("\n" + "="*70)
        print("GENERATING GRAD-CAM VISUALIZATIONS")
        print("="*70)
        
        try:
            from gradcam import create_gradcam_comparison
            create_gradcam_comparison(
                model,
                dataloaders['test'],
                class_names,
                output_dir,
                num_samples=8
            )
            print("✓ Grad-CAM visualizations generated")
        except Exception as e:
            print(f"Note: Grad-CAM generation skipped: {e}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1_macro']:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train spice image classifier')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    main(args.dataset_path, args.output_path)
