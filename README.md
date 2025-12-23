## Overview
This repository contains a **pure from-scratch implementation of the GoogLeNet (Inception-v1) architecture**, developed explicitly to demonstrate architectural understanding rather than reliance on pretrained models or high-level abstractions. Every major component of the network—including Inception modules, auxiliary classifiers, and training logic—is manually constructed using the Keras Functional API.

The objective of this work is **architectural replication, experimental validation, and educational rigor**, aligning with expectations for advanced academic and doctoral-level research portfolios.

---

## Research Motivation
GoogLeNet introduced the concept of **multi-scale feature extraction within a single layer** through Inception modules, significantly improving representational efficiency while controlling parameter growth. Re-implementing this architecture from first principles serves multiple research purposes:
- Deep understanding of convolutional design trade-offs
- Analysis of parallel receptive fields
- Study of auxiliary classifiers for gradient stabilization
- Controlled experimentation without pretrained bias

This implementation prioritizes **conceptual fidelity to the original architecture** while remaining fully reproducible.

---

## Dataset and Preprocessing
The MNIST dataset is used as a controlled benchmark to validate architectural correctness and training behavior.

Preprocessing steps include:
- Reshaping grayscale images into channel-aware tensors
- Normalization to the \[0,1\] range for numerical stability
- Spatial resizing to 224×224 to match GoogLeNet input constraints
- Channel expansion from single-channel to three-channel inputs

These steps allow a canonical architecture to be evaluated on a lightweight dataset.

---

## Architectural Design

### Inception Block (Core Building Unit)
Each Inception block is manually constructed using four parallel paths:
1. 1×1 convolution for dimensionality reduction
2. 1×1 → 3×3 convolution path for medium-scale features
3. 1×1 → 5×5 convolution path for large-scale features
4. 3×3 max pooling → 1×1 convolution for contextual aggregation

Outputs from all paths are concatenated along the channel axis, enabling multi-scale feature fusion.

---

### Full GoogLeNet Architecture
The complete network closely follows the original Inception-v1 design:
- Initial convolution and pooling layers for early feature extraction
- Stacked Inception blocks with increasing depth and width
- Two auxiliary classifiers placed at intermediate depths to:
  - Improve gradient flow
  - Act as implicit regularizers
- Global Average Pooling to reduce parameters
- Dropout for regularization
- Final softmax classification layer

The model produces **three outputs**: one main classifier and two auxiliary classifiers, all trained jointly.

---

## Training Strategy
- Optimizer: Adam
- Loss Function: Sparse categorical cross-entropy
- Multi-output loss optimization
- Generator-based data pipeline for efficient batch processing
- Early stopping based on auxiliary classifier accuracy
- Limited epochs to emphasize architectural validation over dataset saturation

This setup ensures stable convergence while preserving the instructional focus of the implementation.

---

## Evaluation and Analysis
Model performance is analyzed using:
- Accuracy metrics across all outputs
- Prediction probability inspection
- Confusion matrix analysis on sample predictions
- Visualization of training dynamics

The evaluation focuses on **structural correctness and learning behavior**, rather than absolute benchmark performance.

---

## Key Contributions
- Complete GoogLeNet (Inception-v1) implemented from scratch
- Manual construction of Inception blocks and auxiliary classifiers
- Multi-output deep supervision strategy
- Generator-based training pipeline
- Faithful architectural replication without pretrained weights

---

## Technologies Used
Python, TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Seaborn

---

## Research Significance
This repository demonstrates **architectural literacy in deep convolutional networks**, a critical competency for advanced research in computer vision and deep learning. By avoiding pretrained components and high-level shortcuts, the implementation emphasizes foundational understanding, reproducibility, and extensibility—making it suitable as a core artifact in a PhD-level research portfolio.

Future extensions may include architectural ablation studies, scaling experiments, or application to higher-resolution and domain-specific datasets.
