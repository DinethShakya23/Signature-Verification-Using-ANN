<!-- <p align="center">
  <img src="docs/images/coverpage.png" width="500">
</p>

WanDB tacking :- https://wandb.ai/e20055-university-of-peradeniya/Signature_Verification02?nw=nwusere20055 -->

# Signature Verification Using ANN

<p align="center">
  <img src="docs/images/coverpage.png" width="500" alt="Cover">
</p>

W&B tracking: [Signature_Verification02](https://wandb.ai/e20055-university-of-peradeniya/Signature_Verification02?nw=nwusere20055)

---

Table of contents
- Project overview
- Key features
- Repository structure
- Installation
- Dataset & expected layout
- How it works (model & pipeline)
- Usage
  - Training
  - Evaluation
  - Signature verification (inference)
- Logging & artifacts (Weights & Biases)
- Results & visualizations
- Reproducing experiments
- Troubleshooting & FAQ
- Contributing
- License
- Contact

---

## Project overview

This repository demonstrates signature verification using artificial neural networks (ANN). The primary approach implemented is a Siamese-style architecture where an embedding (feature extraction) model is used to convert signature images to embeddings and a comparison head determines whether two signatures match (genuine) or not (forgery).

The main experimental code is provided as Jupyter Notebooks under `codes/`, with training, evaluation, visualization, and a simple signature verification interface.

---

## Key features

- Siamese network built with TensorFlow/Keras to compare signature image pairs.
- Shared embedding model (convolutional feature extractor).
- Option to use absolute difference or concatenation of embeddings as the pair-comparison input.
- Training, validation, and evaluation pipelines with plotting of loss/accuracy and confusion matrix.
- Logging of training/metrics/artifacts to Weights & Biases (wandb).
- Tools to save and load full models and separate embedding models for reuse.
- Example verification utility that computes similarity and threshold-based matching.

---

## Repository structure

- `README.md` (this file)
- `docs/`
  - `images/coverpage.png` (project cover image)
- `codes/`
  - `Approach 01.ipynb` — Primary notebook: defines embedding and siamese models, training, evaluation, visualization, wandb logging, model saving, verification utilities.
  - `Approach 02.ipynb` — Alternate notebook / experiments (contains experimental code; check for differences from Approach 01).
- `signatures/` (datasets expected locally — not included)
  - `full_org/` — genuine signature images
  - `full_forg/` — forged signature images
- (Other files may exist in the repository; please review.)

---

## Installation

Recommended: create a new conda or venv environment.

Requirements (representative; versions used in the notebooks):
- Python 3.8+ (notebooks show Python 3.10 environment)
- tensorflow (e.g., `pip install tensorflow`) — GPU build optional
- opencv-python (`pip install opencv-python`)
- numpy
- matplotlib
- scikit-learn
- wandb (`pip install wandb`)
- jupyter / jupyterlab (to run notebooks)

Example pip install:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn wandb jupyter
```

If you use GPU, install a TensorFlow GPU build that matches your CUDA/cuDNN versions.

---

## Dataset & expected layout

The notebooks expect a local dataset folder (the code uses `data_dir = 'signatures'` or `data_dir = '/kaggle/input/signatures'` depending on environment). The expected structure is:

signatures/
- full_org/        (genuine signature images, e.g., PNG files)
- full_forg/       (forgery signature images)

Image format: grayscale images are loaded using OpenCV (`cv2.imread(..., cv2.IMREAD_GRAYSCALE)`) and resized to a target size (default in notebooks: 128x128). Make sure filenames are standard image extensions (.png/.jpg).

Notes:
- The notebooks also have example absolute paths (Windows). If running locally, update these paths accordingly.

---

## How it works (model & pipeline)

1. Preprocessing
   - Read images in grayscale
   - Resize to a common input size (default size = 128)
   - Normalize pixel values to [0, 1]

2. Embedding model (shared feature extractor)
   - Several Conv2D + MaxPooling layers, flattening into a feature vector
   - The embedding model is defined as a small CNN (see `build_embedding_model` in `Approach 01.ipynb`)

3. Siamese/Pair model
   - Two input branches (input A, input B) share the embedding model
   - The embeddings can be concatenated or combined via absolute difference
   - Dense layers after embedding difference/concat to produce a binary classification (sigmoid output)
   - Loss: binary crossentropy. Metric: accuracy

4. Training
   - Generate pairs (genuine vs forged) and labels
   - Train with callbacks: EarlyStopping, ReduceLROnPlateau
   - Typical configurations used: epochs=20, batch_size=32, learning_rate=0.001

5. Evaluation
   - Evaluate on a held-out test set
   - Compute classification report, confusion matrix, and plot example predictions
   - Save the best model: `siamese_signature_model.keras`
   - Save embedding model separately for reuse: `signature_embedding_model.keras`

6. Verification/inference
   - Use the embedding model to compute embeddings for two images, compute similarity (cosine or custom formula), and compare to a threshold to decide genuine/forged.
   - Example verification function provided in notebooks: `verify_signature(signature1_path, signature2_path)` or `create_signature_verifier(embedding_model, threshold=0.5)`.

---

## Usage

Below are concise examples extracted and adapted from the notebooks.

1) Preprocess & load a single image (inference helper)
```python
import cv2
import numpy as np

def preprocess_single_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)  # shape: (H, W, 1)
```

2) Load models and verify two signatures (example)
```python
import tensorflow as tf
import numpy as np

# load embedding model or full siamese model as appropriate
embedding_model = tf.keras.models.load_model('signature_embedding_model.keras')
# or load full siamese classifier if you prefer direct pair prediction:
siamese_model = tf.keras.models.load_model('siamese_signature_model.keras')

def verify_signature_with_embedding(sig1_path, sig2_path, embedding_model, threshold=0.5, size=128):
    def proc(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size)) / 255.0
        return img.reshape(1, size, size, 1)

    a = proc(sig1_path)
    b = proc(sig2_path)
    em_a = embedding_model.predict(a)
    em_b = embedding_model.predict(b)
    # Compute similarity (example from notebook)
    similarity = 1 - np.sum(np.abs(em_a - em_b)) / np.sum(np.abs(em_a) + np.abs(em_b))
    return (similarity > threshold), float(similarity)
```

3) Predict using full siamese model (pair input)
```python
# given two preprocessed images img1 and img2 with shape (size, size, 1)
prediction = siamese_model.predict([np.expand_dims(img1,0), np.expand_dims(img2,0)])
# prediction is a value in [0,1] — notebook used >0.8 as genuine threshold for that particular experiment
```

---

## Logging & artifacts (Weights & Biases)

The notebooks log experiments to Weights & Biases (wandb). Typical logs include:
- Training/validation loss & accuracy per epoch
- Confusion matrix and sample predictions as images
- Model artifacts: `siamese_signature_model.keras` and `signature_embedding_model.keras`

If you want to use wandb:
1. `pip install wandb`
2. `wandb login` (in a terminal or via notebook)
3. Configure `wandb.init(project="signature_verification02")` or modify to your project name.

---

## Results & visualizations

The notebooks generate:
- Training and validation loss/accuracy plots
- Confusion matrix
- Sample visualizations of signature pairs with predicted labels and match status
- W&B dashboard contains run artifacts and visual reports

Note: exact metrics (accuracy, thresholds) depend on dataset splits and training hyperparameters. The notebook prints test accuracy and classification report after evaluation.

---

## Reproducing experiments

1. Prepare the dataset (images in the `signatures/full_org` and `signatures/full_forg` directories).
2. Install dependencies (see Installation).
3. Launch `codes/Approach 01.ipynb` in Jupyter or Colab/Kaggle. If using Kaggle or Colab, adjust `data_dir` to the notebook environment.
4. Ensure wandb is configured if you want logging.
5. Run cells in order:
   - Data loading & preprocessing
   - Model definition (embedding and siamese)
   - Training (with callbacks)
   - Evaluation & visualization
   - Save models & artifacts

If using Colab/Kaggle GPU, prefer GPU runtime for faster training.

---

## Troubleshooting & FAQ

- Q: Model not training / loss is NaN
  - A: Check input normalization (divide by 255.0). Ensure labels are 0/1 and batches include positive and negative pairs.

- Q: Images can't be read
  - A: Verify the dataset path and that files are readable (`cv2.imread` returns None if path is wrong or file is invalid).

- Q: Different results between runs
  - A: Results may vary due to randomness. Set seeds for reproducibility:
    ```python
    import numpy as np, tensorflow as tf, random
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    ```

- Q: I want to change architecture/hyperparameters
  - A: Edit the model-defining cells (embedding model, classifier head), or change `learning_rate`, `batch_size`, `epochs`, and callbacks in the training cell.

Acknowledgements
- The notebooks use common libraries (TensorFlow, OpenCV, scikit-learn, wandb).
- Example project structure and training pipeline follow standard siamese-learning approaches for image verification.

