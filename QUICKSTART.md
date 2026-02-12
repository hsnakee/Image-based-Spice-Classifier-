# 🚀 Quick Start Guide - Spice Image Classification

Get up and running in 5 minutes!

## 📦 Step 1: Install Dependencies (2 minutes)

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow opencv-python jupyter
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 📁 Step 2: Organize Your Dataset

Make sure your dataset follows this structure:
```
my_spice_dataset/
├── Asafoetida/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── BayLeaf/
│   ├── img1.jpg
│   └── ...
└── ... (other spice folders)
```

## 🎓 Step 3: Train the Model (15-30 minutes)

### Option A: Command Line
```bash
python train.py \
    --dataset_path /path/to/my_spice_dataset \
    --output_path ./outputs
```

### Option B: Jupyter Notebook
```bash
jupyter notebook train_notebook.ipynb
```
Then follow the notebook cells!

### Option C: Python Script
```python
from train import main

main(
    dataset_path="/path/to/my_spice_dataset",
    output_path="./outputs"
)
```

## 🔮 Step 4: Make Predictions

### Single Image
```bash
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --image test_spice.jpg \
    --visualize
```

### Multiple Images
```bash
python predict.py \
    --model outputs/best_model.pt \
    --classes outputs/class_names.json \
    --folder test_images/
```

### In Python
```python
from predict import SpicePredictor

predictor = SpicePredictor(
    model_path='outputs/best_model.pt',
    class_names_path='outputs/class_names.json'
)

# Predict
result = predictor.predict_image('spice.jpg', top_k=5)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## ⚙️ Step 5: Customize (Optional)

Edit `config.py` to change settings:

```python
# Change model
Config.MODEL_NAME = 'efficientnet_b0'  # or 'resnet50', 'efficientnet_b3', 'convnext_tiny'

# Adjust batch size
Config.BATCH_SIZE = 32  # reduce if GPU memory error

# Change epochs
Config.NUM_EPOCHS = 50

# Adjust learning rate
Config.LEARNING_RATE = 0.001
```

## 📊 What You'll Get

After training, check the `outputs/` folder:

```
outputs/
├── best_model.pt              ← Use this for inference!
├── training_curves.png        ← Loss & accuracy plots
├── confusion_matrix.png       ← See which classes confuse the model
├── roc_curves.png            ← ROC analysis
├── class_distribution.png    ← Dataset statistics
├── test_metrics.json         ← All metrics
└── ... (more files)
```

## 🎯 Expected Results

With good data (50+ images per class):
- **Accuracy**: 85-95%
- **Training Time**: 15-30 minutes (GPU) / 1-2 hours (CPU)

## 🐛 Troubleshooting

**"CUDA out of memory"**
```python
Config.BATCH_SIZE = 16  # Reduce batch size
```

**"No module named 'torch'"**
```bash
pip install torch torchvision
```

**"Dataset path does not exist"**
- Check your dataset path is correct
- Make sure folders contain images

**Low accuracy (<70%)**
- Ensure sufficient data (50+ images per class)
- Increase epochs (Config.NUM_EPOCHS = 100)
- Try different model (Config.MODEL_NAME = 'efficientnet_b3')

## 📚 Next Steps

1. ✅ Train model → Done!
2. 📈 Check metrics in `outputs/test_metrics.json`
3. 🔍 Try Grad-CAM: See `gradcam.py`
4. 📓 Explore `inference_notebook.ipynb` for more examples
5. 🎨 Fine-tune hyperparameters in `config.py`

## 💡 Tips

- **GPU recommended** but not required
- **More data = better results**
- Start with `efficientnet_b0` (fast & accurate)
- Use early stopping to prevent overfitting
- Check confusion matrix to identify problem classes

## 📞 Need Help?

Check the full [README.md](README.md) for detailed documentation!

---

Happy Classifying! 🌶️
