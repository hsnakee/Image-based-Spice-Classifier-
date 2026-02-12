# 💡 Usage Examples - Spice Image Classification

Real-world examples for common tasks.

---

## 📋 Table of Contents
1. [Basic Training](#basic-training)
2. [Custom Configuration](#custom-configuration)
3. [Inference Examples](#inference-examples)
4. [Jupyter Notebook Usage](#jupyter-notebook-usage)
5. [Advanced Features](#advanced-features)
6. [Batch Processing](#batch-processing)
7. [Production Deployment](#production-deployment)

---

## 1️⃣ Basic Training

### Minimal Example
```bash
python train.py \
    --dataset_path /data/spices \
    --output_path ./outputs
```

### With GPU Selection
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_path /data/spices \
    --output_path ./outputs
```

### Python Script
```python
from train import main

# Train with default settings
main(
    dataset_path="/data/spices",
    output_path="./outputs"
)
```

---

## 2️⃣ Custom Configuration

### Use Different Model
```python
# Edit config.py before training
from config import Config

Config.MODEL_NAME = 'resnet50'  # Options: resnet50, efficientnet_b0, efficientnet_b3, convnext_tiny
Config.BATCH_SIZE = 64
Config.NUM_EPOCHS = 100
Config.LEARNING_RATE = 0.0005

# Then train
from train import main
main("/data/spices", "./outputs")
```

### Fast Training (Small Dataset)
```python
from config import Config

# Quick experiment settings
Config.MODEL_NAME = 'efficientnet_b0'
Config.BATCH_SIZE = 32
Config.NUM_EPOCHS = 30
Config.EARLY_STOPPING_PATIENCE = 5
Config.FREEZE_BACKBONE_EPOCHS = 3
```

### High Accuracy (Large Dataset)
```python
from config import Config

# Maximum performance settings
Config.MODEL_NAME = 'convnext_tiny'
Config.BATCH_SIZE = 16  # Larger model needs smaller batch
Config.NUM_EPOCHS = 100
Config.LEARNING_RATE = 0.0001
Config.EARLY_STOPPING_PATIENCE = 15
Config.FREEZE_BACKBONE_EPOCHS = 10
```

---

## 3️⃣ Inference Examples

### Single Image Prediction
```bash
# Basic prediction
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --image test_spice.jpg

# With visualization
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --image test_spice.jpg \
    --visualize \
    --output prediction.png

# Top-10 predictions
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --image test_spice.jpg \
    --top_k 10
```

### Python API Usage
```python
from predict import SpicePredictor

# Initialize
predictor = SpicePredictor(
    model_path='outputs/best_model.pt',
    class_names_path='outputs/class_names.json'
)

# Simple prediction
result = predictor.predict_image('spice.jpg', top_k=5)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Access top-k predictions
for i, pred in enumerate(result['top_k_predictions'], 1):
    print(f"{i}. {pred['class']}: {pred['confidence_percent']:.2f}%")

# Visualize
predictor.visualize_prediction('spice.jpg', 'output.png', top_k=5)

# Print formatted results
predictor.predict_and_print('spice.jpg', top_k=5)
```

### Folder Batch Prediction
```bash
# Predict all images in folder
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --folder test_images/ \
    --top_k 3
```

```python
# Python API
from predict import SpicePredictor

predictor = SpicePredictor('outputs/best_model.pt', 'outputs/class_names.json')

# Batch predict
results = predictor.predict_folder('test_images/', top_k=5)

# Process results
for result in results:
    print(f"{result['image_path']}: {result['predicted_class']} "
          f"({result['confidence']*100:.2f}%)")
```

---

## 4️⃣ Jupyter Notebook Usage

### Training in Notebook
```python
# Cell 1: Setup
from train import main
from config import Config

# Cell 2: Configure
DATASET_PATH = "/data/spices"
OUTPUT_PATH = "./outputs"

Config.MODEL_NAME = 'efficientnet_b0'
Config.BATCH_SIZE = 32
Config.NUM_EPOCHS = 50

# Cell 3: Train
main(DATASET_PATH, OUTPUT_PATH)

# Cell 4: View results
from IPython.display import Image
Image(filename='outputs/training_curves.png')
```

### Inference in Notebook
```python
# Cell 1: Load predictor
from predict import SpicePredictor

predictor = SpicePredictor(
    'outputs/best_model.pt',
    'outputs/class_names.json'
)

# Cell 2: Predict
result = predictor.predict_image('test.jpg', top_k=5)
predictor.visualize_prediction('test.jpg', 'pred.png')

# Cell 3: Display
from IPython.display import Image
Image(filename='pred.png')
```

---

## 5️⃣ Advanced Features

### Grad-CAM Visualization
```python
from gradcam import GradCAM, get_target_layer
from model import load_model_for_inference
from dataset_utils import get_transforms
from config import Config
import json

# Load model and classes
with open('outputs/class_names.json', 'r') as f:
    data = json.load(f)
    num_classes = len(data['class_names'])

model = load_model_for_inference('outputs/best_model.pt', num_classes)

# Get target layer
target_layer = get_target_layer(model, Config.MODEL_NAME)

# Create Grad-CAM
gradcam = GradCAM(model, target_layer)

# Generate visualization
transform = get_transforms('test')
cam, pred_class, pred_prob = gradcam.visualize(
    'test_image.jpg',
    transform,
    'gradcam_output.png'
)

print(f"Predicted class: {pred_class}")
print(f"Confidence: {pred_prob*100:.2f}%")
```

### Find Misclassified Samples
```python
from evaluation import find_misclassified_samples, visualize_misclassified
from model import load_model_for_inference
from dataset_utils import create_dataloaders, create_data_splits, load_dataset
import json

# Load model and data
with open('outputs/class_names.json', 'r') as f:
    data = json.load(f)
    class_names = data['class_names']

model = load_model_for_inference('outputs/best_model.pt', len(class_names))

# Load test data
image_paths, labels, _, _ = load_dataset('/data/spices')
splits = create_data_splits(image_paths, labels)
dataloaders = create_dataloaders(splits, class_names)

# Find misclassified
misclassified = find_misclassified_samples(
    model, 
    dataloaders['test'], 
    class_names,
    num_samples=20
)

# Visualize
visualize_misclassified(misclassified, 'misclassified.png')
print(f"Found {len(misclassified)} misclassified samples")
```

### Custom Augmentation
```python
from config import Config

# Configure custom augmentation
Config.AUGMENTATION = {
    'random_horizontal_flip': 0.7,
    'random_vertical_flip': 0.5,
    'random_rotation': 30,
    'color_jitter': {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.2
    },
    'random_crop_scale': (0.7, 1.0),
    'random_crop_ratio': (0.8, 1.2)
}

# Then train as normal
from train import main
main('/data/spices', './outputs')
```

---

## 6️⃣ Batch Processing

### Process Multiple Images
```python
from predict import SpicePredictor
import pandas as pd
from pathlib import Path

# Initialize predictor
predictor = SpicePredictor('outputs/best_model.pt', 'outputs/class_names.json')

# Get all images
image_folder = Path('test_images')
image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))

# Process batch
results = []
for img_path in image_files:
    result = predictor.predict_image(str(img_path), top_k=1)
    results.append({
        'filename': img_path.name,
        'predicted_class': result['predicted_class'],
        'confidence': result['confidence']
    })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values('confidence', ascending=False)

# Save results
df.to_csv('batch_predictions.csv', index=False)
print(df)
```

### Parallel Batch Processing
```python
from predict import SpicePredictor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

predictor = SpicePredictor('outputs/best_model.pt', 'outputs/class_names.json')

def predict_single(img_path):
    try:
        result = predictor.predict_image(str(img_path), top_k=1)
        return {
            'path': str(img_path),
            'class': result['predicted_class'],
            'confidence': result['confidence']
        }
    except Exception as e:
        return {'path': str(img_path), 'error': str(e)}

# Get images
image_files = list(Path('test_images').glob('*.jpg'))

# Parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(predict_single, image_files))

# Save
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 7️⃣ Production Deployment

### REST API Example (Flask)
```python
from flask import Flask, request, jsonify
from predict import SpicePredictor
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Initialize predictor once
predictor = SpicePredictor(
    'outputs/best_model.pt',
    'outputs/class_names.json'
)

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(BytesIO(image_data))
    
    # Save temporarily
    temp_path = 'temp.jpg'
    image.save(temp_path)
    
    # Predict
    result = predictor.predict_image(temp_path, top_k=5)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Load Model Once (Efficiency)
```python
# singleton_predictor.py
class SingletonPredictor:
    _instance = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_predictor(self):
        if self._predictor is None:
            from predict import SpicePredictor
            self._predictor = SpicePredictor(
                'outputs/best_model.pt',
                'outputs/class_names.json'
            )
        return self._predictor

# Usage
predictor = SingletonPredictor().get_predictor()
result = predictor.predict_image('image.jpg')
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY *.py .
COPY outputs/ outputs/

# Expose port
EXPOSE 5000

# Run
CMD ["python", "api.py"]
```

---

## 🔥 Performance Tips

### Speed Up Training
```python
from config import Config

# Enable mixed precision
Config.USE_MIXED_PRECISION = True

# Increase workers
Config.NUM_WORKERS = 8

# Larger batch size (if GPU memory allows)
Config.BATCH_SIZE = 64

# Reduce validation frequency
# (modify train.py to validate every N epochs)
```

### Speed Up Inference
```python
# Use smaller model
Config.MODEL_NAME = 'efficientnet_b0'

# Batch predictions instead of one-by-one
results = predictor.predict_folder('images/')  # Faster than loop
```

### Reduce Memory Usage
```python
# Smaller batch size
Config.BATCH_SIZE = 16

# Smaller model
Config.MODEL_NAME = 'efficientnet_b0'

# Fewer workers
Config.NUM_WORKERS = 2
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Experiment
```python
# 1. Configure for speed
from config import Config
Config.MODEL_NAME = 'efficientnet_b0'
Config.NUM_EPOCHS = 20
Config.BATCH_SIZE = 32

# 2. Train
from train import main
main('/data/spices', './exp1')

# 3. Quick evaluate
from predict import SpicePredictor
predictor = SpicePredictor('exp1/best_model.pt', 'exp1/class_names.json')
predictor.predict_and_print('test.jpg')
```

### Workflow 2: Production Model
```python
# 1. Configure for accuracy
from config import Config
Config.MODEL_NAME = 'efficientnet_b3'
Config.NUM_EPOCHS = 100
Config.BATCH_SIZE = 32
Config.EARLY_STOPPING_PATIENCE = 15

# 2. Train with monitoring
from train import main
main('/data/spices', './production_model')

# 3. Thorough evaluation
# Check outputs/test_metrics.json
# Review confusion_matrix.png
# Analyze misclassified samples

# 4. Deploy
# Use outputs/best_model.pt in production
```

### Workflow 3: Continuous Improvement
```python
# 1. Train baseline
from config import Config
Config.MODEL_NAME = 'efficientnet_b0'
# ... train ...

# 2. Analyze errors
from evaluation import find_misclassified_samples
# ... find problematic classes ...

# 3. Augment data for weak classes
# Add more images or stronger augmentation

# 4. Retrain with adjustments
Config.USE_CLASS_WEIGHTS = True
# ... retrain ...

# 5. Compare metrics
# Compare test_metrics.json between versions
```

---

## 📊 Monitoring & Logging

### View Training Progress
```python
# Load training history
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/training_history.csv')

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_acc'], label='Train')
plt.plot(df['epoch'], df['val_acc'], label='Val')
plt.legend()
plt.title('Accuracy')
plt.show()
```

### Compare Models
```python
import json

# Load metrics from different experiments
with open('exp1/test_metrics.json') as f:
    metrics1 = json.load(f)

with open('exp2/test_metrics.json') as f:
    metrics2 = json.load(f)

# Compare
print(f"Experiment 1 Accuracy: {metrics1['accuracy']:.4f}")
print(f"Experiment 2 Accuracy: {metrics2['accuracy']:.4f}")
```

---

**Happy Classifying! 🌶️**
