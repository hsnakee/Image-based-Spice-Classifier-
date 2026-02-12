"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
Visualizes which regions of the image the model focuses on for predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained model
            target_layer: Target layer for Grad-CAM (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate class activation map
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Class activation map
            predicted_class: Predicted class index
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU on CAM
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), target_class
    
    def visualize(self, image_path, transform, output_path=None, target_class=None):
        """
        Visualize Grad-CAM for an image
        
        Args:
            image_path: Path to input image
            transform: Image transform
            output_path: Path to save visualization
            target_class: Target class for CAM (if None, use prediction)
        
        Returns:
            tuple: (cam, predicted_class, predicted_prob)
        """
        # Load image
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        
        # Transform image
        input_tensor = transform(original_image).unsqueeze(0).to(Config.DEVICE)
        
        # Generate CAM
        cam, predicted_class = self.generate_cam(input_tensor, target_class)
        
        # Get prediction probability
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            predicted_prob = probs[0, predicted_class].item()
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_array.shape[1], original_array.shape[0]))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay CAM on original image
        alpha = 0.5
        overlayed = cv2.addWeighted(original_array, 1-alpha, cam_colored, alpha, 0)
        
        if output_path:
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_array)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # CAM heatmap
            im = axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            
            # Overlayed
            axes[2].imshow(overlayed)
            axes[2].set_title(f'Overlay\nPredicted Class: {predicted_class}\n'
                            f'Confidence: {predicted_prob*100:.2f}%',
                            fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Grad-CAM visualization saved to {output_path}")
        
        return cam_resized, predicted_class, predicted_prob


def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM based on model architecture
    
    Args:
        model: Model instance
        model_name: Name of model architecture
    
    Returns:
        target_layer: Layer to use for Grad-CAM
    """
    if model_name == 'resnet50':
        # Last conv layer in ResNet50
        return model.backbone.layer4[-1].conv3
    
    elif model_name in ['efficientnet_b0', 'efficientnet_b3']:
        # Last conv layer in EfficientNet
        return model.backbone.features[-1][0]
    
    elif model_name == 'convnext_tiny':
        # Last conv layer in ConvNeXt
        return model.backbone.features[-1][-1].block[5]
    
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def visualize_multiple_images(model, image_paths, class_names, transform, output_dir):
    """
    Generate Grad-CAM visualizations for multiple images
    
    Args:
        model: Trained model
        image_paths: List of image paths
        class_names: List of class names
        transform: Image transform
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target layer
    target_layer = get_target_layer(model, Config.MODEL_NAME)
    
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)
    
    print(f"Generating Grad-CAM visualizations for {len(image_paths)} images...")
    
    for i, img_path in enumerate(image_paths):
        output_path = output_dir / f'gradcam_{Path(img_path).stem}.png'
        
        try:
            cam, pred_class, pred_prob = gradcam.visualize(
                img_path, transform, output_path
            )
            print(f"✓ {i+1}/{len(image_paths)}: {Path(img_path).name} -> "
                  f"{class_names[pred_class]} ({pred_prob*100:.2f}%)")
        except Exception as e:
            print(f"✗ Error processing {img_path}: {e}")


def create_gradcam_comparison(model, test_loader, class_names, output_dir, num_samples=8):
    """
    Create a comparison grid of Grad-CAM visualizations
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: List of class names
        output_dir: Directory to save output
        num_samples: Number of samples to visualize
    """
    from dataset_utils import get_transforms
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target layer
    target_layer = get_target_layer(model, Config.MODEL_NAME)
    gradcam = GradCAM(model, target_layer)
    
    # Get sample images
    model.eval()
    images_to_viz = []
    labels_to_viz = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            for img, label in zip(images, labels):
                if len(images_to_viz) < num_samples:
                    images_to_viz.append(img)
                    labels_to_viz.append(label.item())
                else:
                    break
            if len(images_to_viz) >= num_samples:
                break
    
    # Create visualization grid
    rows = 2
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.ravel() if num_samples > 1 else [axes]
    
    transform = get_transforms('test')
    mean = torch.tensor(Config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.NORMALIZE_STD).view(3, 1, 1)
    
    for idx in range(min(num_samples, len(images_to_viz))):
        img_tensor = images_to_viz[idx].unsqueeze(0).to(Config.DEVICE)
        true_label = labels_to_viz[idx]
        
        # Generate CAM
        cam, pred_class = gradcam.generate_cam(img_tensor)
        
        # Denormalize image for display
        img_display = images_to_viz[idx] * std + mean
        img_display = torch.clamp(img_display, 0, 1)
        img_array = img_display.permute(1, 2, 0).cpu().numpy()
        
        # Resize CAM
        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        
        # Create overlay
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        overlayed = 0.6 * img_array + 0.4 * cam_colored
        overlayed = np.clip(overlayed, 0, 1)
        
        # Plot
        axes[idx].imshow(overlayed)
        title_color = 'green' if pred_class == true_label else 'red'
        axes[idx].set_title(
            f'True: {class_names[true_label]}\n'
            f'Pred: {class_names[pred_class]}',
            fontsize=10, color=title_color, fontweight='bold'
        )
        axes[idx].axis('off')
    
    plt.suptitle('Grad-CAM Visualizations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'gradcam_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM comparison saved to {output_dir / 'gradcam_comparison.png'}")
