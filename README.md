# 🌿 Papaya Leaf Disease Detection and Classification Using ResNet-18 with CBAM

## 📌 Project Overview

This project focuses on detecting and classifying papaya leaf diseases using **Deep Learning, ResNet-18, and the Convolutional Block Attention Module (CBAM)**.

The system analyzes papaya leaf images to identify eight categories, including seven disease-related categories and healthy leaves. CBAM enhances feature extraction by helping the model focus on important visual regions of the leaf.

The goal is to support early disease identification, improve classification efficiency, and contribute to smart agriculture.

## 🎯 Objectives

- Develop a deep learning model for papaya leaf disease classification.
- Implement ResNet-18 for image feature extraction.
- Integrate CBAM to emphasize relevant channel and spatial features.
- Apply image preprocessing and data augmentation.
- Evaluate model performance using classification metrics.
- Explore future applications in automated agricultural disease detection.

## 🦠 Disease Categories

The project covers the following eight image classes:

| S. No. | Class Name |
|---|---|
| 1 | Anthracnose |
| 2 | Bacterial Spot |
| 3 | Leaf Curl |
| 4 | Healthy |
| 5 | Mealy Bug |
| 6 | Mite Disease |
| 7 | Mosaic |
| 8 | Ring Spot |

## 🛠️ Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch
- **Model Architecture:** ResNet-18
- **Attention Mechanism:** CBAM (Channel and Spatial Attention)
- **Image Processing:** Torchvision, PIL
- **Numerical Processing:** NumPy
- **Evaluation:** Scikit-learn
- **Development Environment:** Google Colab / VS Code

## 🧠 Proposed Model Architecture

The proposed system combines ResNet-18 with CBAM.

1. **Input:** Papaya leaf images resized to 224 × 224 pixels.
2. **Preprocessing:** Image normalization and data augmentation.
3. **Feature Extraction:** ResNet-18 extracts visual patterns, textures, and disease characteristics.
4. **Attention Mechanism:** CBAM applies channel attention and spatial attention to emphasize informative features.
5. **Classification:** A fully connected layer predicts one of the eight classes.
6. **Evaluation:** Predictions are assessed using accuracy, precision, recall, F1-score, and a confusion matrix.

## 📂 Dataset Structure

Organize the dataset into training and validation folders:

```text
papaya_dataset/
├── train/
│   ├── Anthracnose/
│   ├── Bacterial_Spot/
│   ├── Leaf_Curl/
│   ├── Healthy/
│   ├── Mealy_Bug/
│   ├── Mite_Disease/
│   ├── Mosaic/
│   └── Ring_Spot/
└── val/
    ├── Anthracnose/
    ├── Bacterial_Spot/
    ├── Leaf_Curl/
    ├── Healthy/
    ├── Mealy_Bug/
    ├── Mite_Disease/
    ├── Mosaic/
    └── Ring_Spot/
```

Place each image in its corresponding class folder. Use consistent folder names in both training and validation directories.

**Dataset note:** The PPT describes a dataset of 3,626 images split into 80% training and 20% testing for the existing methods. Confirm the size and split of the actual dataset used in your implementation before reporting these figures as your experimental setup.

## ⚙️ Training Configuration

The PPT describes the following configuration for the proposed method:

| Parameter | Configuration |
|---|---|
| Input image size | 224 × 224 |
| Batch size | 32 |
| Number of classes | 8 |
| Epochs | 25 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Loss function | Cross-Entropy Loss |
| Learning rate scheduler | StepLR |
| Pretrained weights | ImageNet |
| Attention module | CBAM |

These settings describe the proposed model in the presentation. Verify that the final training code uses these settings before treating them as the actual implementation configuration.

## 🚀 Implementation Workflow

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

### 2. Configure the Dataset Path

For Google Colab, update the path to your uploaded dataset:

```python
DATA_DIR = "/content/papaya_dataset"
```

### 3. Preprocess Images

- Resize images to 224 × 224.
- Apply random horizontal flipping and rotation.
- Use color augmentation during training.
- Convert images to tensors.
- Normalize using ImageNet mean and standard deviation.

### 4. Train the Model

Initialize the pretrained ResNet-18 architecture, integrate CBAM, and configure the eight-class output layer. Train using the selected optimizer and loss function.

### 5. Evaluate and Save the Model

Evaluate the trained model using validation images and generate classification metrics. Save the trained weights for later inference.

## 📊 Results and Evaluation

The PPT presents a comparison of VGG16, ResNet-50, and the proposed ResNet-18 with CBAM model across the eight categories.

The evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The presentation reports promising class-wise results for the proposed model. These values should be treated as presentation-reported results until they have been verified against the actual model evaluation output.

**Actual test or validation accuracy:** Add the measured result from your final implementation here.

## 🌱 Applications

- Papaya leaf disease identification.
- Automated agricultural image classification.
- Early disease screening and crop monitoring.
- Research in computer vision and smart agriculture.

## 🔮 Future Scope

- Develop a web or mobile application for leaf image prediction.
- Expand the dataset with more diverse field images.
- Explore real-time disease identification using cameras or drones.
- Investigate integration with IoT-based agricultural monitoring systems.

## 👩‍💻 Project Team

**Project Title:** A Deep Learning Approach for Papaya Leaf Disease Detection and Classification Using ResNet-18 CBAM

**Department:** Computer Science and Engineering  
**Institution:** YSR Engineering College of Yogi Vemana University

## 📄 License
This project is intended for educational and research purposes.

---

⭐ If you find this project useful, consider starring the repository!
