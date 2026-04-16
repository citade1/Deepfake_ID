"""
Inference script for the Multi-Layer LID deepfake classifier.

Input:
    ./data/         — mixed directory of images (.jpg/.jpeg/.png)
                       and videos (.mp4/.avi), no sub-folders.
Output:
    submission.csv  — two columns: filename (str), label (int, 0=Real 1=Fake)

Usage:
    python inference.py [--data_dir DATA_DIR] [--model_dir MODEL_DIR]
                        [--batch_size BATCH_SIZE] [--output OUTPUT]

Requires the classifier weights produced by train.py:
    ./model/lid_classifier.pth
"""

import argparse
import csv
import multiprocessing
import os
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.model import MultiLayerLIDModel

# ── Constants ─────────────────────────────────────────────────────────────────
PRETRAINED       = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_MODEL_DIR = "./model"
CLASSIFIER_FILE  = "lid_classifier.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi"}

NUM_VIDEO_FRAMES = 30          # frames sampled per video
RESIZE_FOR_DETECT = 640        # downscale before dlib for speed
K = 20                         # must match train.py
# ─────────────────────────────────────────────────────────────────────────────


# ── Face detection helpers ────────────────────────────────────────────────────

def _get_bounding_box(face, width: int, height: int):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(cx - size_bb // 2, 0)
    y1 = max(cy - size_bb // 2, 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def detect_and_crop_face(image: Image.Image,
                         target_size=(224, 224)) -> Image.Image | None:
    """Return a face-cropped PIL image, or None if no face detected."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image)
    orig_h, orig_w = arr.shape[:2]

    # Downscale for faster detection
    if orig_w > RESIZE_FOR_DETECT:
        scale = RESIZE_FOR_DETECT / orig_w
        small = cv2.resize(arr,
                           (RESIZE_FOR_DETECT, int(orig_h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = arr

    detector = dlib.get_frontal_face_detector()
    faces = detector(small, 1)
    if not faces:
        return None

    face  = max(faces, key=lambda r: r.width() * r.height())
    # Scale rect back to original resolution
    face  = dlib.rectangle(
        left=int(face.left() / scale),
        top=int(face.top() / scale),
        right=int(face.right() / scale),
        bottom=int(face.bottom() / scale),
    )

    x, y, size = _get_bounding_box(face, orig_w, orig_h)
    cropped = arr[y : y + size, x : x + size]
    return Image.fromarray(cropped).resize(target_size, Image.BICUBIC)


# ── Per-file preprocessing (runs in worker process) ───────────────────────────

def process_file(file_path: Path):
    """
    Preprocess a single image or video file.

    Returns:
        (filename: str, faces: list[PIL.Image], error: str | None)
    """
    faces = []
    ext   = file_path.suffix.lower()
    error = None

    try:
        if ext in IMAGE_EXTS:
            img  = Image.open(file_path)
            face = detect_and_crop_face(img)
            if face:
                faces.append(face)

        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                indices = np.linspace(0, total - 1, NUM_VIDEO_FRAMES, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face = detect_and_crop_face(img)
                    if face:
                        faces.append(face)
            cap.release()

    except Exception as exc:
        error = str(exc)

    return file_path.name, faces, error


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    data_dir: str  = "./data",
    model_dir: str = DEFAULT_MODEL_DIR,
    batch_size: int = 64,
    output: str    = "submission.csv",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    classifier_path = os.path.join(model_dir, CLASSIFIER_FILE)
    print(f"Loading ViT backbone from '{PRETRAINED}' …")
    processor  = ViTImageProcessor.from_pretrained(PRETRAINED)
    base_model = ViTForImageClassification.from_pretrained(PRETRAINED)
    model      = MultiLayerLIDModel(base_model, k=K).to(device)

    print(f"Loading classifier weights from '{classifier_path}' …")
    state = torch.load(classifier_path, map_location=device)
    model.classifier.load_state_dict(state)
    model.eval()

    # ── Collect files ──────────────────────────────────────────────────────────
    data_path = Path(data_dir)
    files = sorted(
        p for p in data_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    )
    print(f"Files found: {len(files)}")

    # ── Parallel preprocessing ─────────────────────────────────────────────────
    n_workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
    print(f"Preprocessing with {n_workers} workers …")

    # filename → list of face PIL images (may be empty)
    faces_per_file: dict[str, list] = {}

    with multiprocessing.Pool(processes=n_workers) as pool:
        with tqdm(total=len(files), desc="Preprocessing") as pbar:
            for filename, faces, error in pool.imap_unordered(process_file, files):
                if error:
                    print(f"  [warn] {filename}: {error}")
                faces_per_file[filename] = faces
                pbar.update(1)

    # ── Batch inference ────────────────────────────────────────────────────────
    # Batch size must be > K for meaningful LID; pad small batches by repeating.
    print("Running inference …")
    results: dict[str, int] = {}

    def predict_faces(face_imgs: list) -> int:
        """Classify a list of face images (from one file) → 0 or 1."""
        if not face_imgs:
            return 0  # no face detected → default Real

        # Replicate faces if too few for LID (need > K points)
        while len(face_imgs) <= K:
            face_imgs = face_imgs * 2

        inputs = processor(images=face_imgs, return_tensors="pt")
        pv     = inputs["pixel_values"].to(device)

        # Process in sub-batches to avoid OOM on large video sets
        all_probs = []
        for start in range(0, pv.size(0), batch_size):
            chunk = pv[start : start + batch_size]
            # Ensure chunk > K; pad by repeating if necessary
            while chunk.size(0) <= K:
                chunk = torch.cat([chunk, chunk], dim=0)
            with torch.no_grad():
                logits = model(chunk)
                probs  = F.softmax(logits, dim=1)[:, 1].cpu()  # P(Fake)
            all_probs.append(probs)

        avg_prob = torch.cat(all_probs).mean().item()
        return int(avg_prob >= 0.5)

    for p in tqdm(files, desc="Classifying"):
        faces = faces_per_file.get(p.name, [])
        results[p.name] = predict_faces(faces)

    # ── Write submission.csv ───────────────────────────────────────────────────
    out_path = Path(output)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for p in files:
            writer.writerow([p.name, results.get(p.name, 0)])

    n_fake = sum(results.values())
    n_real = len(results) - n_fake
    print(f"\nDone. Real: {n_real}  Fake: {n_fake}")
    print(f"Saved → {out_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-layer LID deepfake inference"
    )
    parser.add_argument("--data_dir",   default="./data",
                        help="Directory containing test images/videos")
    parser.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR,
                        help="Directory containing lid_classifier.pth")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Sub-batch size for GPU inference")
    parser.add_argument("--output",     default="submission.csv",
                        help="Output CSV path")
    args, _ = parser.parse_known_args()   # _: ignore Jupyter extra args

    run_inference(
        data_dir   = args.data_dir,
        model_dir  = args.model_dir,
        batch_size = args.batch_size,
        output     = args.output,
    )
