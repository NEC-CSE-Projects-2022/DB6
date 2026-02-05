# 🖼️ DB6 – Dual Function Image System for Multimodal Interface



## 👥 Team Info

- 22471A05O7 — **Y. Krupa Chaitanya** ( [LinkedIn](www.linkedin.com/in/krupa1030) )  
  _Work Done: Designed the overall system architecture and implemented the CLIP-based semantic embedding module. Also integrated image retrieval and text-to-image generation into a unified pipeline._

- 22471A05N8 — **Sk. A. Abdul Kareem** ( [LinkedIn]() )  
  _Work Done: Performed dataset collection, cleaning, and preprocessing for WANG and ImageCLEF datasets. Conducted model evaluation using accuracy, precision, recall, and F1-score metrics._

- 23475A0507 — **G. L. Vara Prasad** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )  
  _Work Done: Implemented YOLOv8 for object detection and region extraction from images. Optimized FAISS indexing for fast similarity search and reduced query response time._

---

## 📌 Abstract
This project presents a dual-function image system that performs both **semantic image retrieval** and **text-to-image generation** in a unified framework. It uses **CLIP (ViT-B/32)** to extract semantic embeddings, **YOLOv8** for object detection, **FAISS** for fast similarity search, and **Stable Diffusion v1.5** for generating images from text prompts. The system is evaluated on **WANG** and **ImageCLEF** datasets and achieves high accuracy, recall, and low query time.

---

## 📄 Paper Reference (Inspiration)
👉 **[Improving the Efficiency of Semantic Image Retrieval Using a Combined Graph and SOM Model – Nguyen Minh Hai et al.](https://ieeexplore.ieee.org/document/10289012)**  

---

## 🚀 Our Improvement Over Existing Paper
- ❌ Removes static ontology dependency  
- ⚡ Faster retrieval using FAISS  
- 🧠 Uses deep semantic embeddings (CLIP)  
- 🖼️ Adds **Text-to-Image Generation** (not in original paper)  
- 🔄 Supports zero-shot learning  

---

## 🧩 About the Project
✔ Retrieves semantically similar images  
✔ Generates images from text  
✔ Useful for education, content search, and design  

### 🔁 Workflow
**Input Image / Text → Preprocessing → YOLOv8 (Object Detection) → CLIP (Feature Extraction) → FAISS (Search) → Output Image / Generated Image**

---

## 📊 Dataset Used
👉 **[WANG Dataset](https://www.kaggle.com/datasets/elkamel/corel-image-dataset)**  
👉 **[ImageCLEF Dataset](https://www.imageclef.org/)**  

### 🗂 Dataset Details
- 🟢 WANG: 10,000 images, 80 classes  
- 🔵 ImageCLEF: 20,000+ images, 276 categories  

---

## 🧰 Dependencies Used
- 🐍 Python  
- 👁️ OpenCV  
- 🔥 PyTorch  
- 📊 NumPy  
- 📈 scikit-learn  
- 🧠 CLIP  
- 📦 YOLOv8  
- ⚡ FAISS  
- 🎨 Stable Diffusion  
- 📉 Matplotlib  

---

## 🔍 EDA & Preprocessing
- 🖼️ RGB conversion  
- 📏 Resize to 512×512  
- 🧹 Remove corrupted images  
- 🏷️ Auto label extraction  
- ✂ Object cropping using YOLOv8  

---

## 🧪 Model Training Info
- 🧠 CLIP generates 512-D embeddings  
- 🎯 YOLOv8 detects objects  
- ⚡ FAISS indexes vectors  
- 🎨 Stable Diffusion generates images  

---

## 🧾 Model Testing / Evaluation
📏 Metrics Used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

🆚 Compared with:
- GP-Tree  
- Graph-GPTree  
- SgGP-Tree  

---

## 🏆 Results
### ✅ WANG Dataset
- 🎯 Top-1 Accuracy: **87.25%**  
- 🥇 Top-5 Accuracy: **94.38%**  
- ⏱ Query Time: **0.09 sec**

### ✅ ImageCLEF Dataset
- 🎯 Top-1 Accuracy: **90.38%**  
- 📊 F1-score: **91.45%**

✔ Outperforms traditional models in speed and accuracy

---

## ⚠️ Limitations & Future Work
- 💻 Needs high GPU power  
- 📉 ImageCLEF partially simulated  
- 🌐 Future:
  - Real-time feedback  
  - Larger datasets  
  - Web UI  
  - Domain fine-tuning  

---

## 🌍 Deployment Info
- 🖥 Python backend  
- ⚡ FAISS indexing  
- 🎨 Stable Diffusion on GPU  
- 🌐 Can use Flask / FastAPI  

---

✨ **Project By:**  
Krupa Chaitanya Yellamelli  
Dual Function Image System for Multimodal Interface  
