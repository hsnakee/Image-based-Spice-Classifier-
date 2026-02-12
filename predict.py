"""
Inference script for spice image classification
Predict classes for single images or batches
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

from config import Config
from model import SpiceClassifier


class SpicePredictor:
    """Predictor class for spice classification"""
    
    def __init__(self, model_path, class_names_path):
        """
        Args:
            model_path: Path to trained model checkpoint
            class_names_path: Path to class names JSON file
        """
        # Load class names
        with open(class_names_path, 'r') as f:
            data = json.load(f)
            self.class_names = data['class_names']
            self.class_to_idx = data['class_to_idx']
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.num_classes = len(self.class_names)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
        
        print(f"Predictor initialized with {self.num_classes} classes")
    
    def _load_model(self, model_path):
        """Load trained model"""
        # Create model architecture
        model = SpiceClassifier(
            model_name=Config.MODEL_NAME,
            num_classes=self.num_classes,
            pretrained=False
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.DEVICE)
        
        return model
    
    def predict_image(self, image_path, top_k=5):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            dict: Prediction results
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
        
        image_tensor = self.transform(image).unsqueeze(0).to(Config.DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx],
                'probability': float(prob),
                'confidence_percent': float(prob * 100)
            })
        
        result = {
            'image_path': str(image_path),
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['probability'],
            'top_k_predictions': predictions
        }
        
        return result
    
    def predict_folder(self, folder_path, top_k=5):
        """
        Predict classes for all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            top_k: Number of top predictions to return per image
        
        Returns:
            list: List of prediction results
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Found {len(image_files)} images")
        
        # Predict for each image
        results = []
        for img_path in image_files:
            try:
                result = self.predict_image(img_path, top_k)
                results.append(result)
                print(f"✓ {img_path.name}: {result['predicted_class']} "
                      f"({result['confidence']*100:.2f}%)")
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
        
        return results
    
    def visualize_prediction(self, image_path, output_path=None, top_k=5):
        """
        Visualize prediction with image and top-k probabilities
        
        Args:
            image_path: Path to image file
            output_path: Path to save visualization (optional)
            top_k: Number of top predictions to show
        """
        # Get prediction
        result = self.predict_image(image_path, top_k)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f'Input Image\n{Path(image_path).name}', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Show predictions
        predictions = result['top_k_predictions']
        classes = [p['class'] for p in predictions]
        probs = [p['probability'] for p in predictions]
        
        colors = ['green' if i == 0 else 'gray' for i in range(len(classes))]
        bars = ax2.barh(classes, probs, color=colors, alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_title(f'Top-{top_k} Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result
    
    def predict_and_print(self, image_path, top_k=5):
        """
        Predict and print results in a formatted way
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to show
        """
        result = self.predict_image(image_path, top_k)
        
        print("\n" + "="*60)
        print(f"IMAGE: {Path(image_path).name}")
        print("="*60)
        print(f"\n{'Rank':<6} {'Class':<25} {'Confidence':<12} {'Bar'}")
        print("-"*60)
        
        for i, pred in enumerate(result['top_k_predictions'], 1):
            conf = pred['confidence_percent']
            bar_length = int(conf / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            
            print(f"{i:<6} {pred['class']:<25} {conf:>6.2f}%      {bar}")
        
        print("-"*60)
        print(f"\nPREDICTED CLASS: {result['predicted_class']}")
        print(f"CONFIDENCE: {result['confidence']*100:.2f}%")
        print("="*60 + "\n")


def main():
    """Main function for command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict spice classes for images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image
  python predict.py --model best_model.pt --classes class_names.json --image spice.jpg
  
  # Predict with visualization
  python predict.py --model best_model.pt --classes class_names.json --image spice.jpg --visualize
  
  # Predict all images in folder
  python predict.py --model best_model.pt --classes class_names.json --folder test_images/
  
  # Show top-10 predictions
  python predict.py --model best_model.pt --classes class_names.json --image spice.jpg --top_k 10
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--classes', type=str, required=True,
                       help='Path to class names JSON file')
    parser.add_argument('--image', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing images')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--output', type=str,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Either --image or --folder must be specified")
    
    # Create predictor
    predictor = SpicePredictor(args.model, args.classes)
    
    # Single image prediction
    if args.image:
        if args.visualize:
            predictor.visualize_prediction(args.image, args.output, args.top_k)
        else:
            predictor.predict_and_print(args.image, args.top_k)
    
    # Folder prediction
    elif args.folder:
        results = predictor.predict_folder(args.folder, args.top_k)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: Predicted {len(results)} images")
        print(f"{'='*60}")
        
        # Save results to JSON
        output_file = Path(args.folder) / 'predictions.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
