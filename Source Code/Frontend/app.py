import os
import io
import base64
from typing import List, Tuple

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
import torch

import numpy as np

# === Optional: FAISS for retrieval ===
# pip install faiss-cpu
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# === Optional: OpenCLIP for CLIP embeddings ===
# pip install open_clip_torch
try:
    import open_clip
    HAVE_OPENCLIP = True
except Exception:
    HAVE_OPENCLIP = False

# === Optional: Diffusers/SDXL hooks (replace placeholder later) ===
# pip install diffusers transformers accelerate
# from diffusers import StableDiffusionXLPipeline

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
MODELS_DIR = "models"
GALLERY_DIR = os.path.join("static", "gallery")
RETRIEVE_TOPK = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # or "stabilityai/stable-diffusion-2-1"

# ------------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------------------------
# Text-to-Image: placeholder generator (replace with SDXL)
# ------------------------------------------------------------------------------------
def generate_placeholder_image(prompt: str, size: Tuple[int, int] = (768, 512)) -> Image.Image:
    """
    Placeholder image showing the prompt text.
    Replace this function with a real SDXL pipeline using your UNet/VAE/TextEncoder .pth files.
    """
    img = Image.new("RGB", size, color=(24, 24, 32))
    draw = ImageDraw.Draw(img)

    # Try a common font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    margin = 30
    wrapped = []
    line = ""
    for word in prompt.split():
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) > (size[0] - 2 * margin):
            wrapped.append(line)
            line = word
        else:
            line = test
    if line:
        wrapped.append(line)

    y = margin
    title = "Generated (placeholder)"
    draw.text((margin, y), title, font=font, fill=(200, 200, 240))
    y += 40
    for ln in wrapped[:8]:
        draw.text((margin, y), ln, font=font, fill=(230, 230, 230))
        y += 30
    return img

# ------------------------------------------------------------------------------------
# CLIP + FAISS: image retrieval
# ------------------------------------------------------------------------------------
clip_model = None
clip_preprocess = None
faiss_index = None
gallery_paths: List[str] = []

def load_clip_and_index_gallery():
    """
    Loads CLIP (ViT-B/32) via open_clip, optionally applies your local weights,
    builds a FAISS index over images in static/gallery.
    """
    global clip_model, clip_preprocess, faiss_index, gallery_paths

    if not HAVE_OPENCLIP:
        print("[WARN] open_clip_torch not installed. Retrieval disabled.")
        return

    # Load OpenCLIP ViT-B/32
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"  # uses openai weights as baseline
    )
    model = model.to(DEVICE).eval()

    # If you have your own CLIP weights, load them
    custom_clip_path = os.path.join(MODELS_DIR, "clip_vitb32.pth")
    if os.path.isfile(custom_clip_path):
        try:
            state = torch.load(custom_clip_path, map_location=DEVICE)
            # Handles either full model or 'state_dict' style
            sd = state.get("state_dict", state)
            model.load_state_dict(sd, strict=False)
            print("[INFO] Loaded custom CLIP weights from models/clip_vitb32.pth")
        except Exception as e:
            print(f"[WARN] Could not load custom CLIP weights: {e}")

    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")  # kept for future text queries if needed
    clip_model = model
    clip_preprocess = preprocess

    # Build FAISS index
    if not os.path.isdir(GALLERY_DIR):
        os.makedirs(GALLERY_DIR, exist_ok=True)

    gallery_paths = []
    for root, _, files in os.walk(GALLERY_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                gallery_paths.append(os.path.join(root, f))
    gallery_paths.sort()

    if not gallery_paths:
        print(f"[WARN] No images found in {GALLERY_DIR}. Add images for retrieval.")

    if not HAVE_FAISS:
        print("[WARN] faiss-cpu not installed. Retrieval disabled.")
        return

    # Compute embeddings
    with torch.no_grad():
        embs = []
        for p in gallery_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
                feat = clip_model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embs.append(feat.squeeze(0).cpu().numpy().astype("float32"))
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

        if not embs:
            print("[WARN] No gallery embeddings computed.")
            return

        mat = np.vstack(embs)  # [N, D]
        d = mat.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine similarity because we normalized
        index.add(mat)
        faiss_index = index
        print(f"[INFO] FAISS index built: {len(embs)} images, dim={d}")

def clip_embed_image(pil_img: Image.Image) -> np.ndarray:
    """
    Produce a normalized CLIP embedding for an image.
    """
    tensor = clip_preprocess(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")  # [1, D]

def search_similar(pil_img: Image.Image, k=RETRIEVE_TOPK) -> List[str]:
    """
    Returns top-k similar image relative paths for display.
    """
    if clip_model is None or faiss_index is None or not gallery_paths:
        return []

    query = clip_embed_image(pil_img)  # [1, D]
    scores, idxs = faiss_index.search(query, k)
    idxs = idxs[0].tolist()
    results = [gallery_paths[i].replace("\\", "/") for i in idxs if 0 <= i < len(gallery_paths)]
    return results

# ------------------------------------------------------------------------------------
# SDXL hooks (replace placeholder with your real pipeline)
# ------------------------------------------------------------------------------------
# def load_sdxl_from_local_parts():
#     """
#     Example outline if you later want to wire SDXL properly from local .pth:
#     - Reconstruct the exact model classes used during training
#     - Instantiate UNet2DConditionModel / AutoencoderKL / text encoders
#     - load_state_dict for each from models/*.pth
#     - Build a pipeline-like function that takes a prompt and returns a PIL image
#     """
#     pass

# ------------------------------------------------------------------------------------
# Load Stable Diffusion
# ------------------------------------------------------------------------------------
def load_stable_diffusion():
    """
    Loads Stable Diffusion model from Hugging Face Hub
    """
    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None  # Optional: disable safety checker for speed
    )
    pipe = pipe.to(DEVICE)
    
    # Enable memory efficient attention if using CUDA
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        # Optionally enable more optimizations
        # pipe.enable_sequential_cpu_offload()
        # pipe.enable_vae_slicing()
    
    return pipe

def generate_image(prompt: str, pipe, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Generates an image using Stable Diffusion pipeline
    
    Args:
        prompt (str): Text prompt for generation
        pipe: Stable Diffusion pipeline instance
        size (tuple): Output image dimensions (width, height)
        
    Returns:
        PIL.Image: Generated image
    """
    try:
        # Add negative prompt for better quality
        negative_prompt = "ugly, blurry, poor quality, distorted, low resolution, bad anatomy, bad proportions, deformed"
        
        # Generate the image
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=size[0],
            height=size[1]
        )
        
        # Return the first generated image
        image = output.images[0]
        
        return image
        
    except Exception as e:
        print(f"Generation error: {e}")
        # Fallback to placeholder if generation fails
        return generate_placeholder_image(prompt, size)

# ------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/hero")
def hero():
    return render_template("hero.html")

@app.route("/generate", methods=["POST"])
def api_generate():
    """
    Text-to-Image generation using Stable Diffusion
    """
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        # Generate image using Stable Diffusion
        img = generate_image(prompt, sd_pipeline, size=(512, 512))
        
        # Convert to base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({"image_b64": encoded})
        
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/retrieve", methods=["POST"])
def api_retrieve():
    """
    Image-to-Image retrieval using CLIP + FAISS over static/gallery.
    """
    if clip_model is None or faiss_index is None or not gallery_paths:
        return jsonify({"results": [], "warn": "Retrieval not initialized. Ensure open_clip + faiss installed and images in static/gallery/."})

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded with field name 'image'."}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image."}), 400

    results = search_similar(img, k=RETRIEVE_TOPK)
    # Convert absolute/relative for client display via Flask static
    web_paths = [p.replace("static/", "/static/") for p in results]
    return jsonify({"results": web_paths})

# Optional: serve gallery listing
@app.route("/gallery")
def list_gallery():
    out = [p.replace("static/", "/static/") for p in gallery_paths]
    return jsonify({"gallery": out})

# ------------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] Device: {DEVICE}")
    
    # Load models
    sd_pipeline = load_stable_diffusion()
    print("[INFO] Stable Diffusion loaded")
    
    load_clip_and_index_gallery()
    
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
