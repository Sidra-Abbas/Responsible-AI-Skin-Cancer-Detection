# Responsible AI for Skin Cancer Classification

## Overview

This project classifies skin lesions using deep learning with responsible AI practices. It uses the HAM10000 dataset containing 10,015 dermatoscopic images across 7 skin lesion types.

## Dataset

**HAM10000 Dataset**

7 Classes:
- nv: Melanocytic nevi (6,705 images)
- mel: Melanoma (1,113 images)
- bkl: Benign keratosis (1,099 images)
- bcc: Basal cell carcinoma (514 images)
- akiec: Actinic keratoses (327 images)
- vasc: Vascular lesions (142 images)
- df: Dermatofibroma (115 images)

## Model Architecture

Convolutional Neural Network:
- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization after each Conv layer
- Max Pooling (2x2)
- Global Average Pooling
- 2 Dropout layers (0.5)
- Dense layer (256 units)
- Output layer (7 classes)

## Training Details

- Image Size: 224x224
- Batch Size: 16
- Epochs: 5
- Optimizer: Adam (lr=0.0001)
- Loss: Sparse Categorical Crossentropy
- Data Split: 80% train, 20% test

## Responsible AI Features

1. **Class Weighting**: Uses balanced class weights instead of oversampling to handle imbalance
2. **Fairness Analysis**: Evaluates performance across age groups
3. **Uncertainty Detection**: Flags predictions with confidence below 70%
4. **Medical Safety**: Monitors melanoma detection recall separately

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.3.0
```

## Installation

```bash
git clone https://github.com/yourusername/Responsible-AI-Skin-Cancer-Classification.git
cd Responsible-AI-Skin-Cancer-Classification
pip install -r requirements.txt
```

## Usage

### Training

Run the notebook cells in order:
1. Load and preprocess data
2. Calculate class weights
3. Train model with class weights
4. Evaluate on test set

### Key Code Snippets

**Calculate Class Weights:**
```python
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
```

**Train with Weights:**
```python
model.fit(train_gen, 
          validation_data=val_gen,
          class_weight=class_weights,
          epochs=5)
```

**Identify Uncertain Predictions:**
```python
predictions = model.predict(X_test)
confidence = predictions.max(axis=1)
low_confidence = confidence < 0.7
```

## Results

The model provides:
- Overall test accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Melanoma detection performance
- Age-stratified performance
- Uncertainty analysis

## File Structure

```
├── README.md
├── requirements.txt
├── skin_cancer_classification.ipynb
├── models/
│   └── skin_cancer_cnn_responsible.h5
└── results/
    ├── confusion_matrix.png
    └── training_history.png
```

## Important Notes

- This is a decision support tool, not for autonomous diagnosis
- Model requires clinical validation before deployment
- Low-confidence predictions should be reviewed by experts
- Performance may vary on different populations

## License

For academic and research purposes.
