# 📊 TensorBoard Logging Guide

Complete guide for using TensorBoard with the Spice Classification Pipeline.

---

## 🚀 Quick Start

### 1. Start Training (TensorBoard automatically enabled)
```python
from train import main

main(
    dataset_path="/path/to/spice/dataset",
    output_path="./outputs"
)
```

### 2. Launch TensorBoard
```bash
tensorboard --logdir outputs/tensorboard_logs
```

### 3. Open Browser
Navigate to: **http://localhost:6006**

---

## 📈 What You'll See in TensorBoard

### 1. **SCALARS Tab**
Real-time training metrics:

#### Loss Curves
- **Loss/train** - Training loss per epoch
- **Loss/val** - Validation loss per epoch
- Combined view showing both curves

#### Accuracy Curves
- **Accuracy/train** - Training accuracy per epoch
- **Accuracy/val** - Validation accuracy per epoch
- Track model improvement over time

#### Learning Rate
- **Learning_Rate** - Learning rate schedule
- See when learning rate changes (scheduler)

### 2. **IMAGES Tab**
Visual monitoring:
- **Confusion_Matrix** - Updated after evaluation
- **Sample_Augmentations** - Training data augmentations
- **Grad-CAM** - Attention maps (if enabled)

### 3. **GRAPHS Tab**
- Model architecture visualization
- Layer connections and data flow

### 4. **HPARAMS Tab**
Experiment comparison:
- Compare different hyperparameter configurations
- Sort by accuracy, loss, etc.
- Parallel coordinates plot
- Scatter plot matrix

### 5. **DISTRIBUTIONS Tab**
- Weight distributions per layer
- Activation distributions
- Gradient distributions (if logged)

---

## 🔧 Configuration

TensorBoard is **enabled by default**. To disable:

```python
# In config.py
Config.ENABLE_TENSORBOARD = False
```

### Custom Log Directory
```python
# Default: outputs/tensorboard_logs
# Customize in tensorboard_logger.py
logger = create_tensorboard_logger(
    output_dir='./my_logs',
    experiment_name='my_experiment'
)
```

---

## 💡 Advanced Usage

### Compare Multiple Experiments

Run multiple trainings with different settings:

```bash
# Experiment 1: ResNet50
python train.py --dataset_path /data --output_path ./exp1_resnet50

# Experiment 2: EfficientNet
# (Change Config.MODEL_NAME to 'efficientnet_b0')
python train.py --dataset_path /data --output_path ./exp2_efficientnet

# Launch TensorBoard with both
tensorboard --logdir ./
```

TensorBoard will show all experiments in one view!

### Filter Experiments
```bash
# Show only specific experiments
tensorboard --logdir ./exp1_resnet50/tensorboard_logs,./exp2_efficientnet/tensorboard_logs
```

### Custom Port
```bash
# If port 6006 is busy
tensorboard --logdir outputs/tensorboard_logs --port 6007
```

### Remote Access
```bash
# Allow external access
tensorboard --logdir outputs/tensorboard_logs --host 0.0.0.0 --port 6006
```

---

## 📊 What Gets Logged

### Every Epoch:
- ✅ Training loss
- ✅ Validation loss
- ✅ Training accuracy
- ✅ Validation accuracy
- ✅ Current learning rate

### After Training:
- ✅ Hyperparameters (model, batch size, LR, etc.)
- ✅ Final metrics (best accuracy, final loss)
- ✅ Model architecture graph

### Optional (if enabled):
- ✅ Confusion matrix visualization
- ✅ Sample augmented images
- ✅ Weight histograms
- ✅ Gradient histograms

---

## 🎯 Interpreting the Plots

### Loss Curves
**Good signs:**
- Training and validation loss both decreasing
- Validation loss following training loss closely
- Smooth curves without wild fluctuations

**Warning signs:**
- **Overfitting**: Training loss keeps decreasing but validation loss increases
- **Underfitting**: Both losses high and not improving
- **Unstable**: Wild fluctuations in loss

**Solutions:**
- Overfitting → Add regularization, dropout, data augmentation
- Underfitting → Increase model capacity, train longer
- Unstable → Reduce learning rate, use gradient clipping

### Accuracy Curves
**Good signs:**
- Validation accuracy increasing steadily
- Training accuracy slightly higher than validation
- Curves converging to high values

**Warning signs:**
- Large gap between train/val accuracy (overfitting)
- Validation accuracy plateauing early
- Validation accuracy decreasing

### Learning Rate
**What to look for:**
- Learning rate reductions correspond to loss plateaus
- If using ReduceLROnPlateau, see step-wise decreases
- If using Cosine Annealing, see smooth decay

---

## 🔍 Finding the Best Model

Use TensorBoard's **HPARAMS** tab:

1. Click **HPARAMS** tab
2. Select metrics to compare (e.g., `best_val_acc`)
3. Sort experiments by performance
4. See which hyperparameters work best

### Parallel Coordinates View
- Shows relationship between hyperparameters and metrics
- Helps identify patterns (e.g., larger batch size → better accuracy)

### Scatter Plot Matrix
- Visualize correlations
- Find optimal hyperparameter combinations

---

## 📸 Screenshots Guide

### Training Progress
![Loss Curves](tensorboard_loss.png)
- Blue line: Training
- Orange line: Validation
- Look for convergence

### Accuracy Over Time
![Accuracy Curves](tensorboard_accuracy.png)
- Monitor improvement
- Identify when to stop training

### Learning Rate Schedule
![Learning Rate](tensorboard_lr.png)
- Visualize scheduler behavior
- Verify learning rate changes

---

## 🚀 Pro Tips

### 1. Real-Time Monitoring
Leave TensorBoard open during training:
```bash
# Start TensorBoard before training
tensorboard --logdir outputs/tensorboard_logs

# In another terminal, start training
python train.py --dataset_path /data --output_path ./outputs
```

Refresh the browser to see updates!

### 2. Smooth Curves
Adjust smoothing slider (top-left):
- Low smoothing (0.0): See raw data
- High smoothing (0.9): See trends

### 3. Zoom & Pan
- Click and drag to zoom
- Double-click to reset view
- Use mouse wheel to zoom

### 4. Download Data
- Hover over chart
- Click download icon
- Export as CSV or SVG

### 5. Compare Runs
- Select multiple runs in the left sidebar
- Toggle visibility with checkboxes
- Use different colors for clarity

---

## 🐛 Troubleshooting

### TensorBoard Not Starting
```bash
# Check if tensorboard is installed
pip list | grep tensorboard

# Install if missing
pip install tensorboard

# Try with full path
python -m tensorboard.main --logdir outputs/tensorboard_logs
```

### No Data Showing
1. Check if log directory exists:
   ```bash
   ls outputs/tensorboard_logs
   ```

2. Verify files are being created:
   ```bash
   ls outputs/tensorboard_logs/*/
   ```

3. Refresh browser (Ctrl+R or Cmd+R)

### Port Already in Use
```bash
# Use different port
tensorboard --logdir outputs/tensorboard_logs --port 6007

# Or kill existing TensorBoard
pkill -f tensorboard
```

### Logs Not Updating
1. **During training**: Wait for epoch to complete
2. **After training**: Click refresh button in browser
3. **Multiple runs**: Check if correct experiment is selected

---

## 📁 Log Structure

```
outputs/
└── tensorboard_logs/
    └── resnet50_spice_classification/
        ├── events.out.tfevents...  # Scalar logs
        ├── events.out.tfevents...  # More logs
        └── ...
```

### Understanding Event Files
- Each run creates event files
- Files are in Protocol Buffer format
- TensorBoard reads these automatically
- **Don't delete** while TensorBoard is running

---

## 🎨 Customization

### Log Custom Metrics

```python
from tensorboard_logger import TensorBoardLogger

# Create logger
logger = TensorBoardLogger('runs', 'my_experiment')

# Log custom scalar
logger.log_scalar('Custom_Metric/f1_score', f1_score, epoch)

# Log images
logger.log_image('Predictions/sample', image_tensor, epoch)

# Log histogram
logger.log_histogram('Weights/layer1', weights, epoch)

# Close
logger.close()
```

### Log Confusion Matrix
```python
# Automatically logged after evaluation
# See tensorboard_logger.py for implementation
logger.log_confusion_matrix(cm, class_names, epoch)
```

---

## 📊 Best Practices

### 1. Experiment Naming
Use descriptive names:
```python
# Good
'resnet50_lr0.001_batch32_aug_heavy'

# Bad
'experiment1'
```

### 2. Log Regularly
Don't log every batch (too much data):
```python
# Good: Log per epoch
if batch_idx % len(dataloader) == 0:
    logger.log_scalar('Loss', loss, epoch)

# Bad: Log every batch (100s of times per epoch)
```

### 3. Clean Up Old Logs
```bash
# Delete old experiments
rm -rf outputs/tensorboard_logs/old_experiment_*/
```

### 4. Save Important Experiments
```bash
# Archive successful experiments
cp -r outputs/tensorboard_logs/best_model/ ./archived_logs/
```

---

## 📈 Example Workflow

```bash
# 1. Start TensorBoard
tensorboard --logdir ./outputs &

# 2. Run training
python train.py --dataset_path /data/spices --output_path ./outputs

# 3. Monitor in browser
# Open http://localhost:6006

# 4. Wait for training to complete
# Watch loss curves decrease
# Check if overfitting occurs

# 5. Analyze results
# Compare with previous experiments
# Check confusion matrix
# Review hyperparameters

# 6. Stop TensorBoard when done
pkill -f tensorboard
```

---

## 🎓 Learning Resources

### Official Documentation
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)

### Video Tutorials
- Search YouTube: "TensorBoard PyTorch tutorial"
- Look for: Real-time training monitoring

### Common Patterns
- Monitor training: Use SCALARS tab
- Debug model: Use GRAPHS tab
- Compare experiments: Use HPARAMS tab
- Visualize data: Use IMAGES tab

---

## ✅ Checklist

Before training:
- [ ] TensorBoard installed (`pip install tensorboard`)
- [ ] `Config.ENABLE_TENSORBOARD = True`
- [ ] Output directory set correctly

During training:
- [ ] TensorBoard running (`tensorboard --logdir outputs/tensorboard_logs`)
- [ ] Browser open to http://localhost:6006
- [ ] Monitoring loss curves
- [ ] Checking for overfitting

After training:
- [ ] Review final metrics in HPARAMS
- [ ] Compare with previous experiments
- [ ] Archive important logs
- [ ] Stop TensorBoard

---

## 🎯 Summary

TensorBoard provides:
- ✅ **Real-time monitoring** of training
- ✅ **Visual comparison** of experiments
- ✅ **Hyperparameter tuning** insights
- ✅ **Model debugging** tools
- ✅ **Publication-ready** plots

Start with:
```bash
tensorboard --logdir outputs/tensorboard_logs
```

Then navigate to **http://localhost:6006** and enjoy! 🚀

---

**Happy Monitoring! 📊**
