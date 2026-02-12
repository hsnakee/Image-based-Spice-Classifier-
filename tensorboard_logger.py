"""
TensorBoard logging utilities for training visualization
Provides real-time monitoring of training metrics
"""

import os
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class TensorBoardLogger:
    """TensorBoard logger for training visualization"""
    
    def __init__(self, log_dir='runs', experiment_name='spice_classification'):
        """
        Args:
            log_dir (str): Directory for TensorBoard logs
            experiment_name (str): Name of the experiment
        """
        self.enabled = TENSORBOARD_AVAILABLE
        
        if self.enabled:
            self.log_dir = Path(log_dir) / experiment_name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            print(f"TensorBoard logging enabled at: {self.log_dir}")
            print(f"Start TensorBoard with: tensorboard --logdir {log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled (tensorboard not installed)")
    
    def log_scalar(self, tag, value, step):
        """
        Log a scalar value
        
        Args:
            tag (str): Name of the scalar (e.g., 'Loss/train')
            value (float): Value to log
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """
        Log multiple scalars in one plot
        
        Args:
            main_tag (str): Parent name (e.g., 'Loss')
            tag_scalar_dict (dict): Dictionary of name->value pairs
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag, img_tensor, step):
        """
        Log an image
        
        Args:
            tag (str): Name of the image
            img_tensor (torch.Tensor): Image tensor [C, H, W]
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_images(self, tag, img_tensors, step):
        """
        Log multiple images in a grid
        
        Args:
            tag (str): Name of the image grid
            img_tensors (torch.Tensor): Image tensors [N, C, H, W]
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            from torchvision.utils import make_grid
            grid = make_grid(img_tensors)
            self.writer.add_image(tag, grid, step)
    
    def log_histogram(self, tag, values, step):
        """
        Log a histogram
        
        Args:
            tag (str): Name of the histogram
            values (torch.Tensor or numpy.ndarray): Values to plot
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag, text, step):
        """
        Log text
        
        Args:
            tag (str): Name of the text
            text (str): Text content
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            self.writer.add_text(tag, text, step)
    
    def log_model_graph(self, model, input_tensor):
        """
        Log model architecture graph
        
        Args:
            model: PyTorch model
            input_tensor: Sample input tensor
        """
        if self.enabled and self.writer:
            try:
                self.writer.add_graph(model, input_tensor)
                print("Model graph added to TensorBoard")
            except Exception as e:
                print(f"Could not add model graph: {e}")
    
    def log_hyperparameters(self, hparam_dict, metric_dict):
        """
        Log hyperparameters and metrics
        
        Args:
            hparam_dict (dict): Dictionary of hyperparameters
            metric_dict (dict): Dictionary of final metrics
        """
        if self.enabled and self.writer:
            self.writer.add_hparams(hparam_dict, metric_dict)
    
    def log_confusion_matrix(self, confusion_matrix, class_names, step):
        """
        Log confusion matrix as an image
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            import torch
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_title('Confusion Matrix', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Convert to tensor
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            self.writer.add_image('Confusion_Matrix', img, step)
            plt.close(fig)
    
    def log_pr_curve(self, tag, labels, predictions, step):
        """
        Log precision-recall curve
        
        Args:
            tag (str): Name of the curve
            labels (numpy.ndarray): True labels
            predictions (numpy.ndarray): Predicted probabilities
            step (int): Step/epoch number
        """
        if self.enabled and self.writer:
            self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def log_embedding(self, features, labels, images=None, tag='embedding'):
        """
        Log embeddings for visualization
        
        Args:
            features (torch.Tensor): Feature vectors [N, D]
            labels (list): Labels for each feature
            images (torch.Tensor, optional): Images [N, C, H, W]
            tag (str): Name of the embedding
        """
        if self.enabled and self.writer:
            self.writer.add_embedding(
                features,
                metadata=labels,
                label_img=images,
                tag=tag
            )
    
    def close(self):
        """Close the TensorBoard writer"""
        if self.enabled and self.writer:
            self.writer.close()
            print("TensorBoard logger closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def log_training_epoch(logger, epoch, metrics_dict):
    """
    Helper function to log all training metrics for an epoch
    
    Args:
        logger (TensorBoardLogger): Logger instance
        epoch (int): Current epoch
        metrics_dict (dict): Dictionary of metrics to log
    """
    if logger.enabled:
        # Log losses
        if 'train_loss' in metrics_dict and 'val_loss' in metrics_dict:
            logger.log_scalars(
                'Loss',
                {
                    'train': metrics_dict['train_loss'],
                    'val': metrics_dict['val_loss']
                },
                epoch
            )
        
        # Log accuracies
        if 'train_acc' in metrics_dict and 'val_acc' in metrics_dict:
            logger.log_scalars(
                'Accuracy',
                {
                    'train': metrics_dict['train_acc'],
                    'val': metrics_dict['val_acc']
                },
                epoch
            )
        
        # Log learning rate
        if 'lr' in metrics_dict:
            logger.log_scalar('Learning_Rate', metrics_dict['lr'], epoch)


def create_tensorboard_logger(output_dir, experiment_name='spice_classification'):
    """
    Create a TensorBoard logger
    
    Args:
        output_dir (str): Output directory path
        experiment_name (str): Name of the experiment
    
    Returns:
        TensorBoardLogger: Logger instance
    """
    log_dir = Path(output_dir) / 'tensorboard_logs'
    return TensorBoardLogger(str(log_dir), experiment_name)


# Example usage
if __name__ == '__main__':
    # Create logger
    logger = TensorBoardLogger('runs', 'example_experiment')
    
    # Log scalars
    for step in range(100):
        loss = 1.0 / (step + 1)
        accuracy = step / 100
        
        logger.log_scalar('Loss/train', loss, step)
        logger.log_scalar('Accuracy/train', accuracy, step)
        
        # Log multiple scalars together
        logger.log_scalars(
            'Metrics',
            {'loss': loss, 'accuracy': accuracy},
            step
        )
    
    # Log hyperparameters
    hparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'model': 'resnet50'
    }
    metrics = {
        'accuracy': 0.95,
        'loss': 0.05
    }
    logger.log_hyperparameters(hparams, metrics)
    
    # Close logger
    logger.close()
    
    print("\nRun TensorBoard with:")
    print("tensorboard --logdir runs")
