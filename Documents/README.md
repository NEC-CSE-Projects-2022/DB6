# A Dual-Function Image System for Multimodal Interface

Repository for the dual-function image system that combines semantic image retrieval and text-to-image generation into a single unified framework.

Summary
- Two main functions:
  1. Semantic Image Retrieval — retrieve semantically related images given an input image (object-aware, content-based search).
  2. Text-to-Image Generation — synthesize high-quality 512×512 images from user-provided text prompts using Stable Diffusion v1.5.
- Key components: YOLOv8 (object detection & cropping), OpenAI CLIP (ViT-B/32) (semantic embeddings), FAISS (fast nearest-neighbor search), Stable Diffusion v1.5 (image synthesis).
- Datasets used for evaluation: WANG dataset (structured) and ImageCLEF (unstructured).
- Notable results: Top-1 accuracy: 90.38%; Top-5 accuracy (WANG): 94.38%; Macro-average ROC AUC: 0.9267; typical query times < 0.1s.

Repository contents
- CAMARA_READY_PAPER.pdf
  - Camera-ready version of the final paper describing the system, experiments, and results.
- DB6_ABSTRACT.pdf
  - Short abstract summarizing goals, methods, and main findings.
- DB6_COLLEGE_REVIEW.pptx
  - Internal review presentation (editable PPTX).
- DB6_CONF_PPT.ppt
  - Conference presentation (legacy PPT).
- DB6_PROJECT_DOCUMENTATION.pdf
  - Full project documentation: design, datasets, experimental setup, reproducibility instructions.
- images/
  - images/figure1.png (text-to-image flowchart — Image 1)
  - images/figure2.png (semantic retrieval flowchart — Image 2)

Figures

Image1: TEXT TO IMAGE 

<img width="420" height="337" alt="image" src="https://github.com/user-attachments/assets/69380410-8e0d-4f20-8133-a8cfe0dfc109" />

- Figure 1 — Text-to-Image Generation Pipeline (images/figure1.png)
  - Shows: Data load → Preprocessing → Stable Diffusion Model → Prompt → Image.
  - This diagram corresponds to the text-to-image generation part where prompts are converted into 512×512 outputs by Stable Diffusion v1.5.
 
Image2: IMAGE TO IMAGE 

<img width="372" height="367" alt="image" src="https://github.com/user-attachments/assets/5c58e48e-627a-41c8-9a6c-b06dd7392e5a" />


- Figure 2 — Semantic Image Retrieval Pipeline (images/figure2.png)
  - Shows: User uploads image → Preprocess → YOLOv8 detects & crops objects → CLIP (ViT-B/32) extracts features ��� FAISS search returns top-k similar images.
  - This diagram corresponds to the image-to-image retrieval flow and the role of object-centric cropping plus CLIP embeddings for semantic similarity.

Quick system description
- Input (retrieval): User uploads an image. The image is normalized and resized to 512×512. YOLOv8 detects objects and crops them. CLIP (ViT-B/32) computes embeddings for each crop. FAISS indexes embeddings and performs a k-NN search to return semantically similar images.
- Input (generation): User provides a text prompt. The system (Stable Diffusion v1.5) synthesizes a 512×512 image from the prompt, with optional preprocessing or prompt engineering.
- Scalability: The system avoids static ontologies by using zero-shot CLIP embeddings and automated label extraction. This enables adaptability to new datasets without retraining ontologies/SOM components.

Models & tools
- YOLOv8 — object detection and automated cropping of semantic regions.
- CLIP (ViT-B/32) — image and text encoders for high-dimensional semantic embeddings (used for retrieval and potential text-based search).
- FAISS — fast similarity search and indexing of CLIP image embeddings.
- Stable Diffusion v1.5 — text-to-image synthesis model used to generate 512×512 images from prompts.
- Python 3.10 with commonly used libraries: OpenCV, Matplotlib, scikit-learn, PyTorch (for models), transformers / diffusers (Stable Diffusion), ultralytics (YOLOv8), faiss (or faiss-cpu/faiss-gpu).

Environment & hardware (used in development)
- Programming language: Python 3.10
- Development: Local machines and Google Colab (for GPU-accelerated experiments)
- Example hardware used: Intel Core i5-12450H, 8 GB RAM (local), NVIDIA Tesla T4 GPU (16 GB VRAM) on Colab.
- OS: Windows 11 (x64) for local development; Colab Linux environment for training/inference.

Evaluation & performance
- Metrics reported: Top-1 accuracy, Top-5 accuracy, ROC AUC (macro-average), precision, recall.
- Example reported results:
  - Top-1 accuracy: 90.38%
  - Top-5 accuracy (WANG): 94.38%
  - Macro-average ROC AUC: 0.9267
  - Typical retrieval query time: < 0.1 seconds (FAISS + CLIP embeddings)
- Notes: Actual performance depends on dataset size, FAISS index type, and available hardware (GPU/CPU).

Installation (summary)
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo>
   ```
2. Create and activate a Python venv (recommended):
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .\.venv\Scripts\activate   # Windows
   ```
3. Install required packages (example):
   ```bash
   pip install -r requirements.txt
   ```
   Example packages (adjust to your environment):
   - torch, torchvision
   - diffusers, transformers
   - openai-clip (or CLIP implementation), faiss-cpu (or faiss-gpu)
   - opencv-python, matplotlib, scikit-learn, ultralytics
4. Download or place model weights (Stable Diffusion v1.5, CLIP, YOLOv8) as instructed in the documentation.

Usage (examples)
- Retrieve semantically similar images (pseudo-CLI):
  ```bash
  python retrieve.py --image path/to/query.jpg --k 5 --index_path data/faiss.index
  ```
- Generate an image from text (pseudo-CLI):
  ```bash
  python generate.py --prompt "A surreal painting of a city skyline at sunset" --output out.png --size 512
  ```

Reproducibility & scripts
- See DB6_PROJECT_DOCUMENTATION.pdf for:
  - Exact dataset splits used (WANG, ImageCLEF)
  - FAISS index construction parameters (index type, metric, nlist/nprobe)
  - CLIP embedding extraction settings (preprocessing, pooling)
  - YOLOv8 model config and thresholds used for cropping
  - Stable Diffusion prompt engineering and scheduler settings

Extensibility & future directions
- Add text-to-image retrieval: use CLIP text encoder to convert a textual query to the same embedding space and run FAISS search to find matching images.
- Integrate feedback loops: collect user relevance feedback to refine ranking or re-weight embeddings.
- Replace or augment CLIP with larger encoders for higher semantic fidelity.
- Support streaming / real-time indexing updates for dynamic datasets.

Security, privacy & licensing notes
- If hosting or serving user-uploaded images, follow privacy policies for storage and retention.
- Models like Stable Diffusion and CLIP are subject to their respective licenses — check and follow model license terms when redistributing or serving generated content.
- Choose a repository license in README (e.g., CC-BY for documents, MIT/Apache for code) and add a LICENSE file.



Notes & next steps
- Replace placeholders (authors, DOI) and ensure `images/figure1.png` and `images/figure2.png` are added to the `images/` folder in the repository.
- If you want, I can:
  - Generate a ready-to-use BibTeX entry once you provide author/title/venue/year.
  - Produce example notebooks for building the FAISS index and running retrieval/generation end-to-end.
  - Convert the legacy PPT to PPTX and check slide compatibility.

Contact & maintainers
- Maintainer: formystudiesbtech@gmail.com
- Last updated: 2026-02-05


