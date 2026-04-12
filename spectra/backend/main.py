from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
from typing import Optional, List
import numpy as np
import torch
import segmentation_models_pytorch as smp
from datetime import datetime
from sqlalchemy.orm import Session
import uuid
import json

from backend.database import init_db, get_db, Detection, WatchZone, AlertLog

app = FastAPI(title="Spectra API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
PATCHES_DIR = Path("data/patches")
TEST_DIR = Path("data/raw/oil-spill/test/images")

# Init DB on startup
init_db()

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


# ─── SCHEMAS ───────────────────────────────────────────────

class ScanRequest(BaseModel):
    scene_id: str = "latest"
    watch_zone_id: Optional[str] = None
    date: Optional[str] = None

class WatchZoneCreate(BaseModel):
    name: str
    client_name: str
    priority: str = "medium"
    polygon_geojson: dict
    description: Optional[str] = None

class AlertDispatch(BaseModel):
    detection_id: str
    recipients: List[str]


# ─── DETECTION ENGINE ──────────────────────────────────────

def run_scan_job(scan_id: str, watch_zone_id: Optional[str] = None):
    from backend.database import SessionLocal
    db = SessionLocal()

    try:
        images = list(TEST_DIR.glob("*.jpg")) if TEST_DIR.exists() else []
        if not images:
            images = list(PATCHES_DIR.rglob("*.npy"))

        if not images:
            det = Detection(
                id=scan_id,
                status="failed",
                detected=False,
                detected_at=datetime.utcnow()
            )
            db.add(det)
            db.commit()
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

        det = Detection(
            id=scan_id,
            watch_zone_id=watch_zone_id,
            scene="S1A_IW_GRDH_Niger_Delta_20240117",
            detected_at=datetime.utcnow(),
            detected=spill_pixels > 100,
            confidence=round(confidence * 100, 2),
            area_km2=area_km2,
            spill_pixels=spill_pixels,
            polygon_geojson=json.dumps(polygon) if polygon else None,
            alert_sent=False,
            status="complete"
        )
        db.add(det)
        db.commit()

        print(f"Scan {scan_id} saved — confidence: {det.confidence}% | area: {area_km2} km²")

    except Exception as e:
        print(f"Scan error: {e}")
        db.rollback()
    finally:
        db.close()


# ─── ROUTES ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Spectra",
        "tagline": "AI-powered oil spill detection for Africa",
        "status": "online",
        "version": "2.0.0"
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
        "database": "sqlite"
    }


# ─── WATCH ZONES ───────────────────────────────────────────

@app.post("/watch-zones")
def create_watch_zone(data: WatchZoneCreate, db: Session = Depends(get_db)):
    zone = WatchZone(
        id=str(uuid.uuid4())[:8],
        name=data.name,
        client_name=data.client_name,
        priority=data.priority,
        polygon_geojson=json.dumps(data.polygon_geojson),
        description=data.description,
        created_at=datetime.utcnow(),
        active=True
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return {
        "id": zone.id,
        "name": zone.name,
        "client_name": zone.client_name,
        "priority": zone.priority,
        "created_at": zone.created_at.isoformat(),
        "message": "Watch zone created successfully"
    }


@app.get("/watch-zones")
def get_watch_zones(db: Session = Depends(get_db)):
    zones = db.query(WatchZone).filter(WatchZone.active == True).all()
    return {
        "watch_zones": [
            {
                "id": z.id,
                "name": z.name,
                "client_name": z.client_name,
                "priority": z.priority,
                "polygon": json.loads(z.polygon_geojson),
                "created_at": z.created_at.isoformat(),
                "description": z.description
            }
            for z in zones
        ],
        "total": len(zones)
    }


@app.delete("/watch-zones/{zone_id}")
def delete_watch_zone(zone_id: str, db: Session = Depends(get_db)):
    zone = db.query(WatchZone).filter(WatchZone.id == zone_id).first()
    if not zone:
        return {"error": "Watch zone not found"}
    zone.active = False
    db.commit()
    return {"message": f"Watch zone {zone_id} deactivated"}


# ─── DETECTIONS ────────────────────────────────────────────

@app.get("/detections")
def get_detections(db: Session = Depends(get_db)):
    dets = db.query(Detection).order_by(Detection.detected_at.desc()).all()
    return {
        "detections": [
            {
                "id": d.id,
                "watch_zone_id": d.watch_zone_id,
                "scene": d.scene,
                "detected_at": d.detected_at.isoformat(),
                "detected": d.detected,
                "confidence": d.confidence,
                "area_km2": d.area_km2,
                "spill_pixels": d.spill_pixels,
                "polygon": json.loads(d.polygon_geojson) if d.polygon_geojson else None,
                "alert_sent": d.alert_sent,
                "status": d.status
            }
            for d in dets
        ],
        "total": len(dets)
    }


@app.get("/detections/{detection_id}")
def get_detection(detection_id: str, db: Session = Depends(get_db)):
    d = db.query(Detection).filter(Detection.id == detection_id).first()
    if not d:
        return {"error": "Detection not found"}
    return {
        "id": d.id,
        "watch_zone_id": d.watch_zone_id,
        "scene": d.scene,
        "detected_at": d.detected_at.isoformat(),
        "detected": d.detected,
        "confidence": d.confidence,
        "area_km2": d.area_km2,
        "spill_pixels": d.spill_pixels,
        "polygon": json.loads(d.polygon_geojson) if d.polygon_geojson else None,
        "alert_sent": d.alert_sent,
        "status": d.status
    }


@app.delete("/detections/{detection_id}")
def delete_detection(detection_id: str, db: Session = Depends(get_db)):
    d = db.query(Detection).filter(Detection.id == detection_id).first()
    if not d:
        return {"error": "Not found"}
    db.delete(d)
    db.commit()
    return {"message": f"Detection {detection_id} deleted"}


# ─── SCAN ──────────────────────────────────────────────────

@app.post("/scan")
def trigger_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    scan_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_scan_job, scan_id, request.watch_zone_id)
    return {
        "scan_id": scan_id,
        "status": "running",
        "message": "Scan started. Poll /detections for results."
    }


# ─── ALERTS ────────────────────────────────────────────────

@app.post("/alerts/dispatch")
def dispatch_alerts(data: AlertDispatch, db: Session = Depends(get_db)):
    detection = db.query(Detection).filter(Detection.id == data.detection_id).first()
    if not detection:
        return {"error": "Detection not found"}

    from backend.alerts import send_spill_alert
    det_dict = {
        "id": detection.id,
        "confidence": detection.confidence,
        "area_km2": detection.area_km2,
        "detected_at": detection.detected_at.isoformat(),
        "scene": detection.scene,
        "spill_pixels": detection.spill_pixels
    }

    results = []
    for recipient in data.recipients:
        try:
            send_spill_alert(det_dict)
            log = AlertLog(
                detection_id=detection.id,
                recipient=recipient,
                sent_at=datetime.utcnow(),
                success=True
            )
            db.add(log)
            results.append({"email": recipient, "status": "sent"})
        except Exception as e:
            results.append({"email": recipient, "status": f"failed: {str(e)}"})

    detection.alert_sent = True
    detection.alert_recipients = json.dumps(data.recipients)
    db.commit()

    return {
        "detection_id": data.detection_id,
        "results": results,
        "total_sent": len([r for r in results if r["status"] == "sent"])
    }


@app.get("/alerts/logs")
def get_alert_logs(db: Session = Depends(get_db)):
    logs = db.query(AlertLog).order_by(AlertLog.sent_at.desc()).limit(50).all()
    return {
        "logs": [
            {
                "detection_id": l.detection_id,
                "recipient": l.recipient,
                "sent_at": l.sent_at.isoformat(),
                "success": l.success
            }
            for l in logs
        ]
    }


# ─── SCENES ────────────────────────────────────────────────

@app.get("/scenes")
def list_scenes():
    scenes = []
    scenes_dir = Path("data/scenes")
    for f in scenes_dir.glob("*.SAFE"):
        scenes.append({"name": f.name, "type": "SAFE", "source": "local"})
    for f in scenes_dir.glob("*.zip"):
        scenes.append({"name": f.name, "type": "zip", "source": "local"})
    return {"scenes": scenes, "total": len(scenes)}
