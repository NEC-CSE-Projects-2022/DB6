DB6 – Project Title

# Team Number – Project Title

## Team Info
- 22471A05XX — **Name** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_

- 22471A05XX — **Name** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_

- 22471A05XX — **Name** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_

- 22471A05XX — **Name** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: xxxxxxxxxx_

---

Dual Function Image System for Multimodal Interface

Abstract

This project presents a dual-function image system that performs both semantic image retrieval and text-to-image generation in a unified framework. It uses CLIP (ViT-B/32) to extract semantic embeddings, YOLOv8 for object detection, FAISS for fast similarity search, and Stable Diffusion v1.5 for generating images from text prompts. The system is evaluated on WANG and ImageCLEF datasets and achieves high accuracy, recall, and low query time, enabling efficient multimodal interaction for image search and creative image synthesis.

Paper Reference (Inspiration)

👉 Improving the Efficiency of Semantic Image Retrieval Using a Combined Graph and SOM Model
– Nguyen Minh Hai et al.

Original conference/IEEE paper used as inspiration for the model.

Our Improvement Over Existing Paper

The existing paper uses Graph and SOM models with static ontology-based structures that require retraining and manual domain adaptation. Our project improves this by introducing a dynamic deep-learning-based framework using CLIP embeddings, YOLOv8 object detection, and FAISS indexing. It removes dependency on static ontologies, supports zero-shot semantic understanding, improves retrieval speed, and additionally adds text-to-image generation using Stable Diffusion, which is not present in the original model.

About the Project

The project allows users to retrieve semantically similar images and generate new images from text prompts.

It is useful for applications like content search, digital design, education, and surveillance.

Workflow:
Input image / text → Preprocessing → Object detection (YOLOv8) → Feature extraction (CLIP) → Vector indexing (FAISS) → Similar image output / Generated image (Stable Diffusion)

Dataset Used

👉 WANG (Corel) Dataset

👉 ImageCLEF Dataset

Dataset Details:
WANG dataset contains 10,000 images grouped into 80 classes with 100 images per class. ImageCLEF contains over 20,000 images across 276 categories with diverse real-world scenes. Both datasets are used to evaluate semantic retrieval performance.

Dependencies Used

Python, OpenCV, PyTorch, scikit-learn, CLIP, YOLOv8, FAISS, Stable Diffusion, NumPy, Matplotlib

EDA & Preprocessing

Images are converted to RGB format and resized to 512×512 pixels with aspect ratio preservation. Noise filtering and sharpening are applied to improve clarity. Corrupted images are removed and class labels are extracted automatically from file names. YOLOv8 is used to crop semantic regions for object-level feature extraction.

Model Training Info

CLIP (ViT-B/32) is used to generate 512-dimensional embeddings for both images and text. YOLOv8 is used for object detection and region extraction. FAISS indexes these embeddings for fast similarity search. Stable Diffusion v1.5 is used for text-to-image synthesis through latent denoising.

Model Testing / Evaluation

The system is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics. Experiments are conducted on WANG and ImageCLEF datasets and compared with GP-Tree, Graph-GPTree, and SgGP-Tree models.

Results

The proposed model achieves 87.25% Top-1 accuracy and 94.38% Top-5 accuracy on the WANG dataset with an average query time of 0.09 seconds. On ImageCLEF, it achieves 90.38% Top-1 accuracy and 91.45% F1-score, outperforming traditional graph-based retrieval models.

Limitations & Future Work

The system requires high GPU resources for training and generation. ImageCLEF evaluation is partially simulated due to hardware constraints. Future work includes real-time feedback integration, scaling to larger datasets, domain-specific fine-tuning, and developing a web-based user interface.

Deployment Info

The system can be deployed using a Python-based backend with GPU support. FAISS is used for vector indexing, and Stable Diffusion runs on a CUDA-enabled server. A web interface can be integrated using Flask or FastAPI for user interaction.
