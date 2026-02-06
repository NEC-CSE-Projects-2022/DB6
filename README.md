# 🖼️ DB6 – Dual Function Image System for Multimodal Interface



## 👥 Team Info

- 22471A05O7 — **Y. Krupa Chaitanya** ( [@krupa1030](www.linkedin.com/in/krupa1030) )  
  _Work Done: Designed the overall system architecture and implemented the CLIP-based semantic embedding module. Also integrated image retrieval and text-to-image generation into a unified pipeline._

- 22471A05N8 — **Sk. A. Abdul Kareem** ( [@abdul](https://www.linkedin.com/in/shaik-anamtaram-abdul-kareem-301114288/) )  
  _Work Done: Performed dataset collection, cleaning, and preprocessing for WANG and ImageCLEF datasets. Conducted model evaluation using accuracy, precision, recall, and F1-score metrics._

- 23475A0507 — **G. L. Vara Prasad** ( [@prasad](https://www.linkedin.com/in/gogineniprasadchowdary/) )  
  _Work Done: Implemented YOLOv8 for object detection and region extraction from images. Optimized FAISS indexing for fast similarity search and reduced query response time._

---

## 📌 Abstract
This project presents a dual-function image system that performs both **semantic image retrieval** and **text-to-image generation** in a unified framework. It uses **CLIP (ViT-B/32)** to extract semantic embeddings, **YOLOv8** for object detection, **FAISS** for fast similarity search, and **Stable Diffusion v1.5** for generating images from text prompts. The system is evaluated on **WANG** and **ImageCLEF** datasets and achieves high accuracy, recall, and low query time.

---

## 📄 Paper Reference (Inspiration)
👉 **[Improving the Efficiency of Semantic Image Retrieval Using a Combined Graph and SOM Model – Nguyen Minh Hai et al.](https://ieeexplore.ieee.org/document/10289012)**  

---

## 🚀 Our Improvement Over Existing Paper
- ❌ Removes static ontology dependency: Unlike the existing paper which relies on manually built ontologies and static semantic structures, this project eliminates the need for predefined knowledge graphs and RDF-based representations, making the system more flexible and easier to scale.
- ⚡ Faster retrieval using FAISS: The system leverages Facebook AI Similarity Search (FAISS) for high-speed vector indexing and similarity matching, significantly reducing query response time compared to traditional graph traversal methods.
- 🧠 Uses deep semantic embeddings (CLIP): Semantic understanding is improved by using CLIP (ViT-B/32), which maps both images and text into a shared embedding space, enabling accurate cross-modal and semantic similarity search.
- 🖼️ Adds **Text-to-Image Generation**: In addition to image retrieval, the system supports text-to-image generation using Stable Diffusion v1.5, allowing users to synthesize realistic images from natural language prompts, which was not supported in the original model.
- 🔄 Supports zero-shot learning: The system can retrieve and classify unseen categories without retraining by using CLIP’s zero-shot capability, increasing generalization and real-world applicability.

---

## 🧩 About the Project
This project implements a dual-function multimodal system capable of performing semantic image retrieval and text-to-image generation within a single framework. Users can upload an image to retrieve visually and semantically similar images from the dataset or provide a text prompt to generate a new image.  
The system is useful for applications such as educational content discovery, digital media design, e-commerce product search, and visual surveillance.

### 🔁 Workflow
**Input Image / Text → Preprocessing → YOLOv8 (Object Detection) → CLIP (Feature Extraction) → FAISS (Search) → Output Image / Generated Image**  
- Input is taken either as an image or text.  
- Images are preprocessed and semantic regions are detected using YOLOv8.  
- CLIP extracts semantic embeddings from detected regions.  
- FAISS performs similarity search on indexed embeddings.  
- Output is returned as similar images or newly generated images.

---

## 📊 Dataset Used
👉 **[WANG Dataset](https://www.kaggle.com/datasets/elkamel/corel-image-dataset)**  
👉 **[ImageCLEF Dataset](https://www.imageclef.org/)**  

### 🗂 Dataset Details
- 🟢 **WANG Dataset**: Contains 10,000 natural images divided into 80 semantic classes with 100 images per class. It provides a clean and balanced benchmark for evaluating classification and retrieval accuracy.
- 🔵 **ImageCLEF Dataset**: Contains more than 20,000 images across 276 fine-grained categories, representing complex real-world scenes. It is used to evaluate system robustness and generalization.

---

## 🧰 Dependencies Used
- 🐍 **Python** – Core programming language used for system development  
- 👁️ **OpenCV** – Image loading, resizing, and preprocessing  
- 🔥 **PyTorch** – Deep learning framework for CLIP, YOLOv8, and Stable Diffusion  
- 📊 **NumPy** – Numerical computation and matrix operations  
- 📈 **scikit-learn** – Performance evaluation metrics  
- 🧠 **CLIP** – Semantic embedding generation for image and text  
- 📦 **YOLOv8** – Object detection and region extraction  
- ⚡ **FAISS** – Fast vector similarity search  
- 🎨 **Stable Diffusion** – Text-to-image generation  
- 📉 **Matplotlib** – Visualization of results

---

## 🔍 EDA & Preprocessing
- 🖼️ All images are converted to RGB format to maintain uniformity.  
- 📏 Images are resized to 512×512 pixels with aspect ratio preservation to ensure model compatibility.  
- 🧹 Corrupted and unsupported image files are removed during data cleaning.  
- 🏷️ Class labels are automatically extracted from directory names and file names to reduce manual annotation effort.  
- ✂ YOLOv8 is used to detect and crop semantic regions, enabling object-level feature extraction instead of full-image features.

---

## 🧪 Model Training Info
- 🧠 CLIP (ViT-B/32) generates 512-dimensional semantic embeddings for both images and text.  
- 🎯 YOLOv8 detects objects and extracts meaningful regions of interest.  
- ⚡ FAISS indexes the embeddings and performs nearest-neighbor similarity search efficiently.  
- 🎨 Stable Diffusion v1.5 generates high-resolution images from text prompts using a latent diffusion process.

---

## 🧾 Model Testing / Evaluation
📏 **Metrics Used:**  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

🆚 **Compared With:**  
- GP-Tree  
- Graph-GPTree  
- SgGP-Tree  

Performance is evaluated on WANG and ImageCLEF datasets and benchmarked against traditional tree-based semantic retrieval methods.

---

## 🏆 Results
### ✅ WANG Dataset
- 🎯 Top-1 Accuracy: **87.25%**  
- 🥇 Top-5 Accuracy: **94.38%**  
- ⏱ Average Query Time: **0.09 seconds**

### ✅ ImageCLEF Dataset
- 🎯 Top-1 Accuracy: **90.38%**  
- 📊 F1-score: **91.45%**

The proposed model outperforms existing graph-based retrieval systems in both accuracy and computational efficiency.

---

## ⚠️ Limitations & Future Work
- 💻 Requires high GPU resources for real-time inference and image generation.  
- 📉 Full ImageCLEF evaluation is partially simulated due to hardware constraints.  
- 🌐 Future enhancements include:
  - Real-time user feedback integration  
  - Experiments on larger-scale datasets  
  - Development of a web-based user interface  
  - Domain-specific fine-tuning of models

---

## 🌍 Deployment Info
- 🖥 Implemented using a Python-based backend  
- ⚡ FAISS is used for vector indexing and fast similarity search  
- 🎨 Stable Diffusion runs on CUDA-enabled GPU servers  
- 🌐 Can be deployed using Flask or FastAPI for web-based access


✨ **Project By:**  
Krupa Chaitanya Yellamelli  
Dual Function Image System for Multimodal Interface  
