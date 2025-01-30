# Deep Multimodal Wound Assessment Tool

## Overview
This repository contains sample code for the **Deep Multimodal Wound Assessment Tool (DM-WAT)**, a framework designed for wound assessment and referral decision-making using machine learning techniques.

**Note:** This repository does not include the dataset due to privacy restrictions regarding medical images.

DM-WAT integrates **image and text data** to provide a comprehensive wound analysis, utilizing **CNN, Vision Transformer (ViT), and BERT-based models**. The code is structured into different modules corresponding to various functionalities.

---

## Repository Structure

### 1. `image_feature_extraction/`
This folder contains code for extracting features from wound images using deep learning models.
- Implements **CNN-based models** (e.g., VGG16, ResNet, EfficientNet) for wound classification.
- Includes **Vision Transformer (ViT) models**, such as **DeiT-Base-Distilled**, which capture long-range dependencies in images.
- Provides utilities for image preprocessing and augmentation to improve model generalization.

### 2. `text_feature_extraction/`
This folder focuses on extracting features from clinical notes using NLP models.
- Utilizes **BERT-based models** such as **DeBERTa** for contextual representation of medical notes.
- Includes utilities for preprocessing text and generating embeddings suitable for classification.
- Supports **GPT-based text augmentation** to mitigate small dataset challenges.

### 3. `multimodal/`
This folder contains code for **combining image and text features** into a unified model.
- Implements **intermediate fusion**, where embeddings from both modalities are concatenated.
- Uses classifiers like **Support Vector Machines (SVM) and Multi-Layer Perceptrons (MLP)** to predict wound referral decisions.
- Enables performance evaluation of multimodal models against unimodal (image-only, text-only) baselines.

### 4. `interpretation/`
This folder provides **interpretability methods** to enhance model transparency.
- Implements **Score-CAM** for visualizing which image regions influence model decisions in CNN and ViT models.
- Uses **Captum (Integrated Gradients)** for analyzing important words in clinical notes for BERT-based models.
- Includes utilities for generating heatmaps and text attribution scores to aid clinical validation.

---

## Requirements
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- OpenCV
- NumPy, Pandas
- Scikit-learn
- Matplotlib (for visualization)

To install dependencies, run:
```bash
pip install -r requirements.txt
```

---


---

## Citation
If you use this code, please cite our related paper:

```
@article{fard2025multimodal,
  title={Multimodal AI on Wound Images and Clinical Notes for Home Patient Referral},
  author={Fard, Reza Saadati and Agu, Emmanuel and Busaranuvong, Palawat and Kumar, Deepak and Gautam, Shefalika and Tulu, Bengisu and Strong, Diane},
  journal={arXiv preprint arXiv:2501.13247},
  year={2025}
}
```

For further questions, please contact the repository maintainer.

