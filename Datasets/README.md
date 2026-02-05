# 📁 Wang Dataset — Preview & Download

<img width="1090" height="579" alt="image" src="https://github.com/user-attachments/assets/bad1933b-aaca-4fe6-804f-a854113860f9" />


> Professional dataset README for the image dataset used in this project. This file documents where to download the dataset, its structure, usage examples, acknowledgement and citation information.

---

## 🔎 Overview
- Name: Image Dataset (Wang-derived / Kaggle mirror)
- Size: 1,000 images
- Categories: 10 classes (100 images per class)
- Source (project copy): Google Drive (see Download section)
- Original dataset: Wang dataset (referenced below)

Badges:
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Download-blue?logo=google-drive)](https://drive.google.com/drive/folders/10TDRhxsY0HgypXyWMBpheL-K2l-xOh7D?usp=sharing) [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Original-orange?logo=kaggle)](https://www.kaggle.com/datasets/ambarish/wangdataset)

---

## 🔗 Download
- Project-hosted copy (Google Drive): https://drive.google.com/drive/folders/10TDRhxsY0HgypXyWMBpheL-K2l-xOh7D?usp=sharing
- Original / reference dataset (Kaggle): https://www.kaggle.com/datasets/ambarish/wangdataset
- Original author site (source images): http://wang.ist.psu.edu/docs/related/

Recommended ways to download:
- From Google Drive: use browser download or Google Drive sync.
- From Kaggle: use the Kaggle CLI
  - kaggle datasets download -d ambarish/wangdataset
- If you prefer a scripted download from Drive, use `gdown` or `rclone` (make sure you have access rights).

---

## 📦 Dataset Contents & Structure
Top-level layout (place these folders under `Datasets/`):

- Datasets/
  - README.md
  - image.png                ← (sample preview image, add here)
  - train/
    - class_0/
      - 0.jpg
      - ...
    - class_1/
    - ...
  - val/
    - class_0/
    - ...
  - test/
    - class_0/
    - ...

Notes:
- There are 10 classes. Each class contains 100 images (1000 total).
- Class folder names here are placeholders (`class_0` … `class_9`). Replace with the actual category names if available.

---

## ℹ️ About the Data
Context
- The dataset contains 1000 natural images across 10 categories. Each category contains 100 images. This dataset is commonly used for image classification and retrieval experiments.

Content
- RGB images in JPG format.
- Mixed resolutions — transformations/resizing are commonly applied before model training.

Inspiration
- Designed for image analysis and experimentation with deep learning models.

Acknowledgements
- Front image (preview) credit: Photo by Khachik Simonian on Unsplash
- Original images and dataset source: http://wang.ist.psu.edu/docs/related/
- This dataset is a copy/mirror used for the course/project; please consult the original authors and Kaggle page for full copyright and terms.

---

## ⚙️ Usage Examples

PyTorch (using ImageFolder)
```python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder('Datasets/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
