# CIFAR-100 Image Classification with ResNet-50

A state-of-the-art deep learning project for CIFAR-100 image classification using transfer learning with ResNet-50, achieving **84.35% test accuracy** through progressive fine-tuning and advanced training techniques.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project implements a robust image classification pipeline for the CIFAR-100 dataset, featuring:

- **Transfer Learning**: Leveraging ResNet-50 pretrained on ImageNet
- **Progressive Fine-tuning**: Three-stage unfreezing strategy for optimal convergence
- **Advanced Augmentation**: Comprehensive data augmentation with Mixup/CutMix
- **Interactive Web Interface**: Streamlit-based application for real-time predictions
- **Production-Ready**: Complete training pipeline with early stopping and model checkpointing

## 🚀 Features

- ✨ **High Accuracy**: 84.35% test accuracy on CIFAR-100
- 🎨 **Rich Augmentation**: ColorJitter, RandomRotation, GaussianBlur, RandomErasing, Mixup/CutMix
- 📊 **Interactive UI**: Upload images and get instant predictions with confidence scores
- 🔄 **Progressive Training**: Three-stage fine-tuning strategy
- 📈 **Learning Rate Scheduling**: OneCycleLR for optimal convergence
- 🛡️ **Early Stopping**: Prevents overfitting with patience-based monitoring
- 📉 **Comprehensive Logging**: Track training metrics across all epochs

## 📋 Table of Contents

- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

## 🏗️ Model Architecture

### Base Model
- **Architecture**: ResNet-50
- **Pretrained Weights**: ImageNet1K-V2
- **Total Parameters**: 23,712,932
- **Final Layer**: Modified FC layer (2048 → 100 classes)

### Training Configuration
```python
Optimizer: SGD (Nesterov momentum=0.9, weight_decay=5e-4)
Loss Function: SoftTargetCrossEntropy (training) / CrossEntropyLoss (validation)
Scheduler: OneCycleLR with cosine annealing
Batch Size: 32
Input Size: 224×224
```

## 🎓 Training Strategy

### Three-Stage Progressive Fine-tuning

#### **Stage 1: FC Layer Only (Epochs 1-10)**
- Only the final fully connected layer is trainable
- Learning Rate: max_lr=0.01
- Warmup: 30% of cycle
- **Purpose**: Adapt the classifier to CIFAR-100 without disrupting pretrained features

#### **Stage 2: Deep Layers (Epochs 11-25)**
- Unfreeze: Layer 3, Layer 4, and FC
- Differential Learning Rates:
  - Layer 3: 0.0005
  - Layer 4: 0.001
  - FC: 0.005
- **Purpose**: Fine-tune deeper layers for CIFAR-100 specific features

#### **Stage 3: Full Model (Epochs 26-100)**
- All layers trainable
- Layer-wise Learning Rates:
  - Layer 1: 0.00005
  - Layer 2: 0.0001
  - Layer 3: 0.0003
  - Layer 4: 0.0008
  - FC: 0.002
- **Purpose**: End-to-end fine-tuning with discriminative learning rates

### Data Augmentation Pipeline

#### Training Augmentation
```python
- Resize to 224×224
- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
- RandomGrayscale (p=0.2)
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±15°)
- RandomAffine (degrees=15, translate=(0.1, 0.1))
- RandomPerspective (distortion=0.5, p=0.2)
- GaussianBlur (kernel=3, sigma=(0.1, 0.2))
- Normalization (ImageNet stats)
- RandomErasing (p=0.3, scale=(0.05, 0.2))
```

#### Mixup & CutMix
```python
- Mixup Alpha: 0.8
- CutMix Alpha: 0.8
- Switch Probability: 0.5
- Label Smoothing: 0.1
```

## 💻 Installation

### Prerequisites
- Python 3.10
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Amirali-SoltaniRad/cifar100-classification.git
cd cifar100-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
torch==2.1.2
torchvision==0.16.2
streamlit==1.51.0
numpy<2
pandas==2.3.3
Pillow==11.3.0
timm==1.0.22
tqdm==4.67.1
jupyter>=1.1.1
notebook>=7.4.7
```

4. **Download CIFAR-100 dataset**

Download dataset from release and extract

e.g., download cifar-100-python.tar.gz from [releases page](https://github.com/Amirali-SoltaniRad/cifar100-classification/releases) and extract to ./data

## 🎮 Usage

### Training

Run the training notebook to train your own model:

```bash
jupyter notebook train.ipynb
```

Or convert to script and run:
```bash
jupyter nbconvert --to script train.ipynb
python train.py
```

**Training will automatically**:
- Split data into train/val/test (90%/10%/test)
- Apply progressive fine-tuning
- Save best model based on validation loss
- Implement early stopping (patience=10)

### Inference with Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features**:
- Upload images (JPG, JPEG, PNG)
- Real-time predictions
- Confidence scores for all 100 classes
- Probability distribution visualization

### Programmatic Inference

```python
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(2048, 100)
model.load_state_dict(torch.load("best_model.pth")["model_state_dict"])
model.to(device)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), 
                        (0.2675, 0.2565, 0.2761))
])

image = Image.open("path/to/image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    
print(f"Predicted class: {predicted.item()}")
print(f"Confidence: {confidence.item():.2%}")
```

## 📊 Results

### Final Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **84.35%** |
| **Test Loss** | 0.7622 |
| **Best Validation Loss** | 0.9729 |
| **Training Epochs** | 66/100 (Early Stopping) |
| **Total Parameters** | 23,712,932 |
| **Training Time** | ~15.5 hours (GPU - NVIDIA GeForce GTX 1650) |

### Training Progression

| Stage | Epochs | Best Val Acc | Description |
|-------|--------|-------------|-------------|
| Stage 1 | 1-10 | ~40% | FC layer training |
| Stage 2 | 11-25 | ~73% | Deep layers fine-tuning |
| Stage 3 | 26-66 | ~79% | Full model fine-tuning |

### Loss Curves

The model achieved steady convergence with:
- Consistent training loss reduction
- Validation loss stabilization around epoch 56
- Early stopping triggered at epoch 66

## 📁 Dataset

### CIFAR-100

- **Total Images**: 60,000 (32×32 RGB)
- **Training Set**: 45,000 images (90% split)
- **Validation Set**: 5,000 images (10% split)
- **Test Set**: 10,000 images
- **Classes**: 100 fine-grained categories
- **Superclasses**: 20 coarse categories

#### Class Distribution
All classes are balanced with 500 training images and 100 test images per class.

#### Sample Classes
```
Aquatic mammals: beaver, dolphin, otter, seal, whale
Fish: aquarium fish, flatfish, ray, shark, trout
Flowers: orchids, poppies, roses, sunflowers, tulips
Food: apples, mushrooms, oranges, pears, peppers
Household: bottles, bowls, cans, cups, plates
... (95 more classes)
```

## 📂 Project Structure

```
cifar100-classification/
│
├── app.py                      # Streamlit web application
├── train.ipynb                 # Training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── model_checkpoint.pth        # Pretrained model checkpoint
|__ cifar-100-python.tar.gz     # compressed dataset

```

> **Note:** You can download `model_checkpoint.pth` and `cifar-100-python.tar.gz` from the [release page](https://github.com/Amirali-SoltaniRad/cifar100-classification/releases).

## 🔧 Customization

### Modify Training Parameters

Edit `train.ipynb` to adjust:

```python
# Hyperparameters
batch_size = 32
max_epochs = 100
patience = 10

# Learning rates per stage
stage1_lr = 0.01
stage2_lrs = [0.0005, 0.001, 0.005]
stage3_lrs = [0.00005, 0.0001, 0.0003, 0.0008, 0.002]

# Augmentation
mixup_alpha = 0.8
cutmix_alpha = 0.8
label_smoothing = 0.1
```

### Change Model Architecture

```python
from torchvision import models

# Try different ResNet variants
model = models.resnet34(weights="IMAGENET1K_V1")  # Lighter model
model = models.resnet101(weights="IMAGENET1K_V2")  # Deeper model

# Or use different architectures
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model = models.vit_b_16(weights="IMAGENET1K_V1")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/MIT) file for details.

## 🙏 Acknowledgments

- **CIFAR-100 Dataset**: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009
- **ResNet Architecture**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He et al., 2015
- **PyTorch Team**: For the excellent [deep learning framework](https://pytorch.org/)
- **Streamlit Team**: For the intuitive [web app framework](https://streamlit.io/)
- **TIMM Library**: [Ross Wightman](https://github.com/rwightman/pytorch-image-models) for data augmentation utilities
  
## 📚 References

1. Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR.
3. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR.
4. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899). ICCV.
5. Smith, L. N., & Topin, N. (2019). [Super-convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120).

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

**© 2025 Amirali Soltani Rad - [GitHub](https://github.com/Amirali-SoltaniRad)**

---

⭐ If you find this project helpful, please consider giving it a star!

**Happy Classifying! 🎨🤖**
