# 🌶️ Spice Image Classification Pipeline (Dataset from Indian biological image dataset(https://ibdc.dbtindia.gov.in/ibia/))

Ai Assitant used : Claude Sonet
IDE : Pycharm

A production-ready, deep learning pipeline for multi-class spice image classification using PyTorch and transfer learning.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Jupyter Notebook](#jupyter-notebook)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Output Artifacts](#output-artifacts)
- [Results](#results)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## ✨ Features

### Core Features
- ✅ **Transfer Learning**: Support for ResNet50, EfficientNet-B0/B3, ConvNeXt-Tiny
- ✅ **Automatic Mixed Precision (AMP)**: Faster training with lower memory usage
- ✅ **Data Augmentation**: Comprehensive augmentation pipeline
- ✅ **Class Imbalance Handling**: Weighted loss for imbalanced datasets
- ✅ **Early Stopping**: Prevents overfitting with configurable patience
- ✅ **Learning Rate Scheduling**: Multiple scheduler options
- ✅ **Reproducible Results**: Fixed seeds for deterministic training

### Evaluation & Visualization
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- 📈 **Training Curves**: Loss and accuracy plots
- 🔥 **Confusion Matrix**: Detailed per-class performance
- 📉 **ROC Curves**: Multi-class ROC analysis
- 🎨 **Class Distribution**: Visual analysis of dataset balance

### Bonus Features
- 🔍 **Grad-CAM Visualization**: See what the model focuses on
- 🎯 **Misclassified Samples**: Identify and visualize errors
- 📝 **TensorBoard Logging**: Real-time training monitoring
- 🚀 **Batch Inference**: Predict on multiple images efficiently

## 📁 Dataset Structure

The pipeline expects a folder-per-class structure:

```
dataset_root/
├── Asafoetida/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── BayLeaf/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── BlackCardamom/
│   └── ...
└── ... (other spice classes)
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, but not required)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/spice-classification.git
cd spice-classification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n spice-classifier python=3.9
conda activate spice-classifier

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🎯 Quick Start

### Training (5 minutes setup)
```bash
python train.py \
    --dataset_path /path/to/spice/dataset \
    --output_path ./outputs
```

### Inference (Single Image)
```bash
python predict.py \
    --model ./outputs/best_model.pt \
    --classes ./outputs/class_names.json \
    --image test_image.jpg \
    --visualize
```

## 📖 Usage

### Training

#### Basic Training
```bash
python train.py \
    --dataset_path /path/to/dataset \
    --output_path ./results
```

#### Jupyter Notebook Training
See `train_notebook.ipynb` for interactive training:

```python
# In Jupyter notebook
from train import main

# Set your paths
dataset_path = "/path/to/spice/dataset"
output_path = "./outputs"

# Run training
main(dataset_path, output_path)
```

#### Configuration
Edit `config.py` to customize training:

```python
# Model selection
Config.MODEL_NAME = 'efficientnet_b0'  # Options: resnet50, efficientnet_b0, efficientnet_b3, convnext_tiny

# Training hyperparameters
Config.BATCH_SIZE = 32
Config.NUM_EPOCHS = 50
Config.LEARNING_RATE = 0.001

# Data splits
Config.TRAIN_RATIO = 0.70
Config.VAL_RATIO = 0.15
Config.TEST_RATIO = 0.15

# Enable/disable features
Config.USE_MIXED_PRECISION = True
Config.EARLY_STOPPING = True
Config.USE_CLASS_WEIGHTS = True
```

### Inference

#### Single Image Prediction
```bash
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --image spice_sample.jpg \
    --top_k 5
```

#### Visualized Prediction
```bash
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --image spice_sample.jpg \
    --visualize \
    --output prediction_viz.png
```

#### Batch Prediction (Folder)
```bash
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --folder test_images/ \
    --top_k 3
```

#### Programmatic Usage
```python
from predict import SpicePredictor

# Initialize predictor
predictor = SpicePredictor(
    model_path='outputs/best_model.pt',
    class_names_path='outputs/class_names.json'
)

# Single prediction
result = predictor.predict_image('spice.jpg', top_k=5)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Batch prediction
results = predictor.predict_folder('test_images/', top_k=5)

# Visualize
predictor.visualize_prediction('spice.jpg', 'output.png')
```

## 🏗️ Model Architecture

### Supported Models

| Model | Parameters | Image Size | Best For |
|-------|-----------|------------|----------|
| **ResNet50** | 25.6M | 224×224 | Balanced performance |
| **EfficientNet-B0** | 5.3M | 224×224 | Lightweight, fast |
| **EfficientNet-B3** | 12M | 300×300 | Higher accuracy |
| **ConvNeXt-Tiny** | 28.6M | 224×224 | State-of-the-art |

### Transfer Learning Strategy
1. **Phase 1**: Freeze backbone, train classifier (5 epochs)
2. **Phase 2**: Unfreeze backbone, fine-tune all layers (remaining epochs)
3. **Learning Rate**: Reduced by 10× when unfreezing backbone

## ⚙️ Configuration

### Key Configuration Options

```python
# config.py

# Model Configuration
MODEL_NAME = 'efficientnet_b0'
IMAGE_SIZE = 224
NUM_CLASSES = 19  # Auto-detected

# Training Settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Optimization
OPTIMIZER = 'adam'  # adam, sgd, adamw
SCHEDULER = 'reduce_on_plateau'  # reduce_on_plateau, cosine, step
USE_MIXED_PRECISION = True

# Early Stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

# Augmentation
AUGMENTATION = {
    'random_horizontal_flip': 0.5,
    'random_vertical_flip': 0.3,
    'random_rotation': 15,
    'color_jitter': {...}
}
```

## 📤 Output Artifacts

After training, the following files are generated in the output directory:

```
outputs/
├── best_model.pt                  # Best model checkpoint
├── final_model.pt                 # Final epoch model
├── training_history.csv           # Training logs
├── config.json                    # Training configuration
├── class_names.json              # Class mapping
├── test_metrics.json             # Test set metrics
│
├── Visualizations:
├── training_curves.png           # Loss & accuracy curves
├── confusion_matrix.png          # Confusion matrix heatmap
├── roc_curves.png               # Multi-class ROC curves
├── per_class_metrics.png        # Per-class performance
├── class_distribution.png       # Dataset statistics
├── augmented_samples.png        # Sample augmentations
└── gradcam_comparison.png       # Grad-CAM visualizations
```

## 📊 Results

### Expected Performance
On the 19-class spice dataset with proper training:

| Metric | Expected Range |
|--------|---------------|
| **Test Accuracy** | 85-95% |
| **F1 Score (Macro)** | 0.83-0.93 |
| **Training Time** | 15-30 min (GPU) |

### Sample Output

```
=================================================================
TEST SET RESULTS
=================================================================
Accuracy:           0.9245
Precision (macro):  0.9187
Recall (macro):     0.9156
F1 Score (macro):   0.9168
=================================================================
```

## 🗂️ Project Structure

```
spice-classification/
│
├── train.py                 # Main training script
├── predict.py              # Inference script
├── config.py               # Configuration settings
├── model.py                # Model architectures
├── dataset_utils.py        # Data loading & preprocessing
├── evaluation.py           # Metrics & visualization
├── gradcam.py             # Grad-CAM implementation
│
├── notebooks/
│   ├── train_notebook.ipynb      # Training in Jupyter
│   └── inference_notebook.ipynb  # Inference examples
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── LICENSE                # License file
```

## 🎨 Advanced Features

### Grad-CAM Visualization

Visualize model attention:

```python
from gradcam import GradCAM, get_target_layer, visualize_multiple_images
from model import load_model_for_inference

# Load model
model = load_model_for_inference('outputs/best_model.pt', num_classes=19)

# Get target layer
target_layer = get_target_layer(model, 'efficientnet_b0')

# Create Grad-CAM
gradcam = GradCAM(model, target_layer)

# Visualize
gradcam.visualize('spice.jpg', transform, 'gradcam_output.png')
```

### Misclassified Samples Analysis

```python
from evaluation import find_misclassified_samples, visualize_misclassified

# Find misclassified samples
misclassified = find_misclassified_samples(
    model, test_loader, class_names, num_samples=20
)

# Visualize
visualize_misclassified(misclassified, 'misclassified.png')
```

### TensorBoard Logging

```bash
# Start TensorBoard
tensorboard --logdir outputs/tensorboard_logs

# Open browser to http://localhost:6006
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in config.py
Config.BATCH_SIZE = 16  # or 8
```

**2. Low Accuracy**
- Ensure sufficient training data (>50 images per class recommended)
- Increase number of epochs
- Try different model architectures
- Check for data quality issues

**3. Import Errors**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

**4. Slow Training**
```python
# Enable mixed precision
Config.USE_MIXED_PRECISION = True

# Increase num_workers
Config.NUM_WORKERS = 4  # adjust based on CPU cores
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{spice_classification_2024,
  author = {Ekansh Jadon},
  title = {Spice Image Classification Pipeline},
  year = {2025},
  url = {https://github.com/hsnakee/spice-classification}
}
@Dataset {Indian biological image dataset,
  source = {https://ibdc.dbtindia.gov.in/ibia/},
  year = {2025},
  url = {https://ibdc.dbtindia.gov.in/ibia/}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for the Computer Vision Community**
