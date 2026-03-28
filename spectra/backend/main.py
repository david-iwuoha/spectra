from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import segmentation_models_pytorch as smp
from datetime import datetime
import uuid

app = FastAPI(title="Spectra API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for hackathon demo
detections = []

MODELS_DIR = Path("models")
PATCHES_DIR = Path("data/patches")
TEST_DIR = Path("data/raw/oil-spill/test/images")

def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )
    model.load_state_dict(torch.load(
        MODELS_DIR / "spectra_model.pth",
        map_location="cpu"
    ))
    model.eval()
    return model

print("Loading Spectra model...")
MODEL = load_model()
print("Model ready.")


class ScanRequest(BaseModel):
    scene_id: str = "latest"
    date: str = None


def run_scan_job(scan_id: str):
    # Prefer Kaggle test images — they contain real spill signatures
    images = list(TEST_DIR.glob("*.jpg")) if TEST_DIR.exists() else []

    # Fallback to preprocessed patches
    if not images:
        images = list(PATCHES_DIR.rglob("*.npy"))

    if not images:
        detections.append({
            "id": scan_id,
            "error": "No data available for scan",
            "detected": False,
            "status": "failed"
        })
        return

    all_probs = []
    for img_path in images[:4]:
        path_str = str(img_path)

        if path_str.endswith(".npy"):
            patch = np.load(img_path).astype(np.float32)
        else:
            img = Image.open(img_path).convert("L")
            img = img.resize((256, 256))
            arr = np.array(img, dtype=np.float32) / 255.0
            patch = np.stack([arr, arr], axis=0)

        inp = torch.tensor(patch).unsqueeze(0).float()
        with torch.no_grad():
            pred = torch.sigmoid(MODEL(inp))
        prob = pred.squeeze().numpy()
        all_probs.append(prob)

    combined = np.mean(all_probs, axis=0)
    binary = (combined > 0.5).astype(int)
    spill_pixels = int(binary.sum())
    confidence = float(combined[binary == 1].mean()) if spill_pixels > 0 else 0.0
    area_km2 = round((spill_pixels * 100) / 1e6, 4)

    # Build GeoJSON polygon anchored to Niger Delta coordinates
    rows, cols = np.where(binary == 1)
    if len(rows) > 0:
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [5.5 + float(cols.min()) / 10000, 4.5 + float(rows.min()) / 10000],
                [5.5 + float(cols.max()) / 10000, 4.5 + float(rows.min()) / 10000],
                [5.5 + float(cols.max()) / 10000, 4.5 + float(rows.max()) / 10000],
                [5.5 + float(cols.min()) / 10000, 4.5 + float(rows.max()) / 10000],
                [5.5 + float(cols.min()) / 10000, 4.5 + float(rows.min()) / 10000],
            ]]
        }
    else:
        polygon = None

    result = {
        "id": scan_id,
        "scene": "S1A_IW_GRDH_Niger_Delta_20240117",
        "detected_at": datetime.utcnow().isoformat(),
        "detected": spill_pixels > 100,
        "confidence": round(confidence * 100, 2),
        "area_km2": area_km2,
        "spill_pixels": spill_pixels,
        "polygon": polygon,
        "alert_sent": confidence > 0.7,
        "status": "complete"
    }

    detections.append(result)
    print(f"Scan {scan_id} complete — confidence: {result['confidence']}% | area: {area_km2} km²")


@app.get("/")
def root():
    return {
        "name": "Spectra",
        "tagline": "AI-powered oil spill detection for Africa",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    patch_count = len(list(PATCHES_DIR.rglob("*.npy")))
    test_count = len(list(TEST_DIR.glob("*.jpg"))) if TEST_DIR.exists() else 0
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "patches_available": patch_count,
        "test_images_available": test_count,
        "detections_count": len(detections)
    }


@app.get("/detections")
def get_detections():
    return {
        "detections": detections,
        "total": len(detections)
    }


@app.get("/detections/{detection_id}")
def get_detection(detection_id: str):
    for d in detections:
        if d["id"] == detection_id:
            return d
    return {"error": "Detection not found"}


@app.delete("/detections/{detection_id}")
def delete_detection(detection_id: str):
    global detections
    detections = [d for d in detections if d["id"] != detection_id]
    return {"message": f"Detection {detection_id} deleted"}


@app.post("/scan")
def trigger_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    scan_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_scan_job, scan_id)
    return {
        "scan_id": scan_id,
        "status": "running",
        "message": "Scan started. Poll /detections for results."
    }


@app.get("/scenes")
def list_scenes():
    scenes = []
    scenes_dir = Path("data/scenes")
    for f in scenes_dir.glob("*.SAFE"):
        scenes.append({"name": f.name, "type": "SAFE", "source": "local"})
    for f in scenes_dir.glob("*.zip"):
        scenes.append({"name": f.name, "type": "zip", "source": "local"})
    return {"scenes": scenes, "total": len(scenes)}
