"""Inference for the Multi-Layer LID deepfake classifier.

Reads a flat ./data directory of images (.jpg/.jpeg/.png) and videos
(.mp4/.avi), detects/crops faces (dlib), and writes submission.csv with columns
filename, label (0=Real, 1=Fake). Requires the train.py checkpoint
(classifier + frozen LID reference bank) at ./checkpoints/lid_classifier.pth.
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

PRETRAINED        = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_MODEL_DIR = "./checkpoints"
CLASSIFIER_FILE   = "lid_classifier.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi"}

NUM_VIDEO_FRAMES  = 30
RESIZE_FOR_DETECT = 640


def _get_bounding_box(face, width, height):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(cx - size_bb // 2, 0)
    y1 = max(cy - size_bb // 2, 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def detect_and_crop_face(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    orig_h, orig_w = arr.shape[:2]

    if orig_w > RESIZE_FOR_DETECT:
        scale = RESIZE_FOR_DETECT / orig_w
        small = cv2.resize(arr, (RESIZE_FOR_DETECT, int(orig_h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = arr

    detector = dlib.get_frontal_face_detector()
    faces = detector(small, 1)
    if not faces:
        return None

    face = max(faces, key=lambda r: r.width() * r.height())
    face = dlib.rectangle(
        left=int(face.left() / scale), top=int(face.top() / scale),
        right=int(face.right() / scale), bottom=int(face.bottom() / scale),
    )
    x, y, size = _get_bounding_box(face, orig_w, orig_h)
    cropped = arr[y:y + size, x:x + size]
    return Image.fromarray(cropped).resize(target_size, Image.BICUBIC)


def process_file(file_path: Path):
    """Preprocess one file -> (filename, list[face PIL], error)."""
    faces, ext, error = [], file_path.suffix.lower(), None
    try:
        if ext in IMAGE_EXTS:
            face = detect_and_crop_face(Image.open(file_path))
            if face:
                faces.append(face)
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                for idx in np.linspace(0, total - 1, NUM_VIDEO_FRAMES, dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    face = detect_and_crop_face(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if face:
                        faces.append(face)
            cap.release()
    except Exception as exc:
        error = str(exc)
    return file_path.name, faces, error


def run_inference(data_dir="./data", model_dir=DEFAULT_MODEL_DIR,
                  batch_size=64, output="submission.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    classifier_path = os.path.join(model_dir, CLASSIFIER_FILE)
    ckpt = torch.load(classifier_path, map_location=device)
    if "reference_bank" not in ckpt:
        raise KeyError("Checkpoint has no 'reference_bank'; re-train with the current train.py.")
    k = ckpt.get("k", 20)

    processor  = ViTImageProcessor.from_pretrained(PRETRAINED)
    base_model = ViTForImageClassification.from_pretrained(PRETRAINED)
    model      = MultiLayerLIDModel(base_model, k=k).to(device)
    model.load_reference_bank({n: t.to(device) for n, t in ckpt["reference_bank"].items()})
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()

    data_path = Path(data_dir)
    files = sorted(p for p in data_path.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    print(f"Files found: {len(files)}")

    n_workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
    faces_per_file = {}
    with multiprocessing.Pool(processes=n_workers) as pool:
        with tqdm(total=len(files), desc="Preprocessing") as pbar:
            for filename, faces, error in pool.imap_unordered(process_file, files):
                if error:
                    print(f"  [warn] {filename}: {error}")
                faces_per_file[filename] = faces
                pbar.update(1)

    # LID is measured against the fixed reference bank, so each face is scored
    # independently (no batch padding / replication; B=1 is valid).
    def predict_faces(face_imgs):
        if not face_imgs:
            return 0  # no face -> default Real
        pv = processor(images=face_imgs, return_tensors="pt")["pixel_values"].to(device)
        probs = []
        for start in range(0, pv.size(0), batch_size):
            with torch.no_grad():
                logits = model(pv[start:start + batch_size])
                probs.append(F.softmax(logits, dim=1)[:, 1].cpu())
        return int(torch.cat(probs).mean().item() >= 0.5)

    results = {p.name: predict_faces(faces_per_file.get(p.name, []))
               for p in tqdm(files, desc="Classifying")}

    with open(Path(output), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for p in files:
            writer.writerow([p.name, results.get(p.name, 0)])

    n_fake = sum(results.values())
    print(f"\nDone. Real: {len(results) - n_fake}  Fake: {n_fake}")
    print(f"Saved → {Path(output).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-layer LID deepfake inference")
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--output",     default="submission.csv")
    args, _ = parser.parse_known_args()
    run_inference(args.data_dir, args.model_dir, args.batch_size, args.output)
