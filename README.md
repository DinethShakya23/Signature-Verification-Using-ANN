# Signature Verification Using Siamese Neural Networks

<p align="center">
  <img src="docs/images/coverpage.png" width="600" alt="Signature Verification Cover">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Accuracy-97.92%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Dataset-CEDAR-blue" alt="Dataset">
  <img src="https://img.shields.io/badge/Tracked-W%26B-yellow?logo=weightsandbiases&logoColor=black" alt="W&B">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

> A Siamese Neural Network that achieves **97.92% test accuracy** on the CEDAR signature dataset by learning a shared embedding space and classifying signature pairs as genuine or forged using L1 distance comparison.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Tracking](#experiment-tracking)
- [Reproducing Experiments](#reproducing-experiments)
- [License](#license)

---

## Overview

Handwritten signature verification is a core biometric authentication challenge. This project implements a **Siamese Neural Network** that learns a discriminative embedding space for signature images. Two signatures are independently passed through a shared CNN, and the resulting embeddings are compared to determine whether the pair is genuine or forged.

Two comparison strategies were explored:

| Approach | Strategy | Notebook |
|---|---|---|
| Approach 1 | Concatenated embeddings → classification head | `approach_01_concatenation.ipynb` |
| Approach 2 | L1 absolute difference → classification head | `approach_02_abs_difference.ipynb` ✓ best |

Approach 2 (L1 difference) produced the best results and is the recommended notebook to run.

---

## Results

Evaluated on a held-out 20% test split of the [CEDAR Signature dataset](#dataset):

| Metric | Forged | Genuine |
|---|---|---|
| Precision | 0.97 | 0.99 |
| Recall | 0.99 | 0.97 |
| F1-Score | 0.98 | 0.98 |
| **Test Accuracy** | | **97.92%** |

Training converged in ~14 epochs with early stopping (patience=5). Full training curves, confusion matrix, and sample pair visualizations are available on the [W&B dashboard](https://wandb.ai/e20055-university-of-peradeniya/Signature_Verification02?nw=nwusere20055).

---

## Architecture

```
Input A ──┐
          ├── Shared CNN Embedding Model ──┐
Input B ──┘                                ├── |emb_A − emb_B| → Dense(64, ReLU) → Dense(1, Sigmoid)
```

**Shared embedding CNN:**

```
Conv2D(64,  3×3, ReLU) → MaxPool(2×2)
Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
Conv2D(256, 3×3, ReLU) → MaxPool(4×4)
Conv2D(512, 3×3, ReLU) → MaxPool(4×4)
Flatten
```

Input images are 128×128 grayscale. The embedding model weights are shared between both branches, enforcing that the same feature extractor processes both signatures. The L1 absolute difference between embeddings is passed to a small dense classifier with binary cross-entropy loss.

---

## Dataset

This project uses the **CEDAR (Center of Excellence for Document Analysis and Recognition) Signature Dataset**:

- **55 signers** × 24 genuine signatures = 1,320 genuine samples
- **55 signers** × 24 forged signatures = 1,320 forged samples
- Images: grayscale PNG, resized to 128×128 for training

**Download:** [CEDAR Signature Dataset on Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)

> The dataset is not included in this repository. Download it from Kaggle and place it under `signatures/` as described in [Usage](#usage).

---

## Repository Structure

```
Signature-Verification-Using-ANN/
├── codes/
│   ├── approach_01_concatenation.ipynb    # Siamese net with concatenated embeddings (local)
│   └── approach_02_abs_difference.ipynb   # Siamese net with L1 difference + W&B logging (Kaggle/GPU)
├── trained model/
│   └── siamese_signature_model.keras      # Trained Keras model
├── docs/
│   └── images/
│       └── coverpage.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/DinethShakya23/Signature-Verification-Using-ANN.git
cd Signature-Verification-Using-ANN

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

For GPU training, install a TensorFlow build that matches your CUDA/cuDNN versions.

---

## Usage

### 1. Dataset layout

After downloading the CEDAR dataset, structure it as:

```
signatures/
├── full_org/     ← genuine signature images (.png)
└── full_forg/    ← forged signature images (.png)
```

### 2. Training

Open `codes/approach_02_abs_difference.ipynb` in Kaggle or Colab (GPU recommended) and run all cells. For a local CPU run, use `codes/approach_01_concatenation.ipynb` and update `data_dir` to your local path.

### 3. Inference

```python
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('trained model/siamese_signature_model.keras')

def preprocess(path, size=128):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size)).astype('float32') / 255.0
    return img.reshape(1, size, size, 1)

sig1 = preprocess('path/to/reference.png')
sig2 = preprocess('path/to/test.png')

score = model.predict([sig1, sig2])[0][0]
print("Genuine" if score > 0.8 else "Forged", f"— score: {score:.4f}")
```

---

## Experiment Tracking

Training metrics, confusion matrices, and model artifacts are tracked with **Weights & Biases**:

**[W&B Project Dashboard →](https://wandb.ai/e20055-university-of-peradeniya/Signature_Verification02?nw=nwusere20055)**

To log your own runs:

```bash
wandb login
```

Then set `wandb.init(project="your-project-name")` at the top of the training notebook.

---

## Reproducing Experiments

1. Download the CEDAR dataset and place it at `signatures/`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `codes/approach_02_abs_difference.ipynb` on a GPU runtime
4. Run all cells in order — seeds are fixed for reproducibility:
   ```python
   import random, numpy as np, tensorflow as tf
   random.seed(42); np.random.seed(42); tf.random.set_seed(42)
   ```

**Hyperparameters used:**

| Parameter | Value |
|---|---|
| Image size | 128×128 |
| Batch size | 32 |
| Learning rate | 0.001 (with ReduceLROnPlateau) |
| Max epochs | 20 (early stopping, patience=5) |
| Optimizer | Adam |
| Loss | Binary cross-entropy |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
