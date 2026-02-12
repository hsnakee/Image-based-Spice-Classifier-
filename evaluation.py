"""
Evaluation metrics and visualization utilities
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from config import Config


def evaluate_model(model, test_loader, class_names, output_dir):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: List of class names
        output_dir: Directory to save outputs
    
    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(Config.DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision (macro):  {precision_macro:.4f}")
    print(f"Recall (macro):     {recall_macro:.4f}")
    print(f"F1 Score (macro):   {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1 Score (weighted):  {f1_weighted:.4f}")
    print("="*70)
    
    print("\nPer-Class Metrics:")
    print("-"*70)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-"*70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {int(support[i])}")
    print("-"*70)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'all_preds': all_preds.tolist(),
        'all_labels': all_labels.tolist(),
        'all_probs': all_probs.tolist()
    }
    
    return metrics


def plot_training_history(history, output_path):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history['lr'], 'g-^', label='Learning Rate', linewidth=2, markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_confusion_matrix(cm, class_names, output_path, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save plot
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        linewidths=0.5, linecolor='gray'
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curves(all_labels, all_probs, class_names, output_path):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        all_labels: True labels
        all_probs: Predicted probabilities
        class_names: List of class names
        output_path: Path to save plot
    """
    n_classes = len(class_names)
    
    # Binarize labels
    all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_bin.ravel(), np.array(all_probs).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=3
    )
    
    # Plot per-class ROC curves
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, linewidth=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {output_path}")


def plot_per_class_metrics(classification_report, class_names, output_path):
    """
    Plot per-class precision, recall, and F1-score
    
    Args:
        classification_report: Classification report dictionary
        class_names: List of class names
        output_path: Path to save plot
    """
    # Extract metrics
    precision = [classification_report[name]['precision'] for name in class_names]
    recall = [classification_report[name]['recall'] for name in class_names]
    f1 = [classification_report[name]['f1-score'] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Spice Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {output_path}")


def save_metrics(metrics, output_path):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    # Create a copy without large arrays
    metrics_to_save = {
        'accuracy': metrics['accuracy'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'precision_weighted': metrics['precision_weighted'],
        'recall_weighted': metrics['recall_weighted'],
        'f1_weighted': metrics['f1_weighted'],
        'per_class_precision': metrics['per_class_precision'],
        'per_class_recall': metrics['per_class_recall'],
        'per_class_f1': metrics['per_class_f1'],
        'per_class_support': metrics['per_class_support'],
        'per_class_accuracy': metrics['per_class_accuracy'],
        'classification_report': metrics['classification_report']
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"Metrics saved to {output_path}")


def find_misclassified_samples(model, test_loader, class_names, num_samples=20):
    """
    Find misclassified samples
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: List of class names
        num_samples: Number of misclassified samples to return
    
    Returns:
        list: List of misclassified sample information
    """
    model.eval()
    
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(Config.DEVICE)
            
            outputs = model(images_gpu)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Find misclassified samples in this batch
            mask = predicted.cpu() != labels
            if mask.sum() > 0:
                for i in range(len(mask)):
                    if mask[i]:
                        misclassified.append({
                            'image': images[i],
                            'true_label': labels[i].item(),
                            'pred_label': predicted[i].item(),
                            'true_class': class_names[labels[i]],
                            'pred_class': class_names[predicted[i]],
                            'confidence': probs[i][predicted[i]].item()
                        })
                        
                        if len(misclassified) >= num_samples:
                            return misclassified
    
    return misclassified


def visualize_misclassified(misclassified, output_path):
    """
    Visualize misclassified samples
    
    Args:
        misclassified: List of misclassified sample information
        output_path: Path to save plot
    """
    if len(misclassified) == 0:
        print("No misclassified samples found!")
        return
    
    n_samples = min(len(misclassified), 16)
    rows = 4
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.ravel()
    
    # Denormalize function
    mean = torch.tensor(Config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.NORMALIZE_STD).view(3, 1, 1)
    
    for idx in range(n_samples):
        sample = misclassified[idx]
        img = sample['image']
        
        # Denormalize
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"True: {sample['true_class']}\n"
            f"Pred: {sample['pred_class']}\n"
            f"Conf: {sample['confidence']:.2f}",
            fontsize=9, color='red'
        )
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Misclassified samples visualization saved to {output_path}")
