# Face Detection System using HOG and SVM

This system detects faces in images using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier, with advanced post-processing techniques for improved accuracy.

## Features

- **HOG Feature Extraction**: Captures gradient information in 8x8 cells
- **Linear SVM Classifier**: Trained on positive/negative samples
- **Adaptive Parameter Tuning**: Automatically adjusts parameters based on image size
- **Soft Non-Maximum Suppression**: Handles overlapping detections
- **DBSCAN Clustering**: Groups duplicate detections
- **Multi-stage Validation**: Verifies faces using geometric and appearance features

## Key Algorithms

### 1. HOG Feature Extraction
- Divides image into 8x8 pixel cells
- Computes gradient histograms (9 orientation bins)
- Normalizes blocks of 2x2 cells using L2-Hys norm
- G = √(I_x² + I_y²), θ = arctan(I_y/I_x)

### 2. SVM Classification
- Linear kernel with balanced class weights
- Regularization parameter C=0.5
- Decision function thresholding with adaptive confidence

### 3. Adaptive Scaling
| Image Type       | Min Face Size | Scales               | Step Size |
|------------------|---------------|----------------------|-----------|
| Small (<250px)   | 15% of min-dim | [0.95, 1.0, 1.05]   | 10       |
| Medium           | 5% of min-dim  | [0.6, 0.8, 1.0, 1.2]| 8        |
| Crowded          | 3% of min-dim  | [0.1-1.4]           | 6        |

### 4. Post-processing
- **Soft-NMS**: Decays confidence of overlapping boxes
- **DBSCAN**: Clusters detections using adaptive epsilon
- **Validation**:
  - Aspect ratio (0.6-1.5)
  - Contrast (std > 15)
  - Skin color (>25% pixels)
  - Eye detection (for large faces)

## Performance Optimizations
- Dynamic confidence thresholding using KDE
- Early termination for high-confidence faces
- CLAHE contrast enhancement
- Image downscaling for large inputs

## Usage
```python
from detector import detect_faces

# Initialize model
model, scaler, window_size = train_detector()

# Detect faces in image
boxes, scores = detect_faces(image, model, scaler, window_size)