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
from fastapi import HTTPException
from backend.wind_context import WindContextLayer, drift_arrow_geojson
from backend.optical_validator import OpticalValidator
from datetime import datetime
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
_wind_context = WindContextLayer()
_optical_validator = OpticalValidator()

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
            lookalike_score=round(confidence, 4) if spill_pixels > 100 else None,
            lookalike_label="oil" if spill_pixels > 100 else None,
            lookalike_passed=True if spill_pixels > 100 else None,

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

@app.post("/detections/{detection_id}/wind")
def refresh_wind(detection_id: str, db: Session = Depends(get_db)):
    """
    Re-fetch wind context for an existing detection.
    Safe integration:
    - Uses existing schema fields if present
    - Derives centroid from polygon_geojson (since your model has no centroid columns)
    - Uses detected_at timestamp (since your model has no timestamp column)
    """

    det = db.query(Detection).filter(Detection.id == detection_id).first()

    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")

    if not det.polygon_geojson:
        raise HTTPException(status_code=400, detail="Detection missing polygon")

    try:
        polygon = json.loads(det.polygon_geojson)
        coords = polygon["coordinates"][0]

        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]

        centroid_lon = sum(lons) / len(lons)
        centroid_lat = sum(lats) / len(lats)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid polygon geometry")

    timestamp = (
        det.detected_at.isoformat()
        if det.detected_at
        else datetime.utcnow().isoformat()
    )

    wind = _wind_context.get_context(
        lat=centroid_lat,
        lon=centroid_lon,
        timestamp=timestamp,
    )

    # Update DB fields safely
    det.wind_speed_ms = wind["wind_speed_ms"]
    det.wind_direction_deg = wind["wind_direction_deg"]
    det.wind_u = wind["wind_u"]
    det.wind_v = wind["wind_v"]

    det.sar_validity = wind["sar_validity"]
    det.sar_validity_detail = wind["sar_validity_detail"]

    det.lookalike_wind_risk = wind["lookalike_wind_risk"]
    det.lookalike_wind_note = wind["lookalike_wind_note"]

    det.wind_fetched_at = wind["wind_fetched_at"]
    det.wind_data_source = wind["wind_data_source"]

    # keep backward compatibility with your old fields
    det.wind_speed = wind["wind_speed_ms"]
    det.wind_reliable = wind["sar_validity"] == "valid"

    if wind["drift_vector"]:
        det.drift_bearing_deg = wind["drift_vector"]["bearing_deg"]
        det.drift_speed_ms = wind["drift_vector"]["speed_ms"]
        det.drift_24h_km = wind["drift_vector"]["24h_km"]

        det.drift_geojson = json.dumps(
            drift_arrow_geojson(
                centroid_lat,
                centroid_lon,
                wind["drift_vector"],
                hours=24
            )
        )

    db.commit()
    db.refresh(det)

    return {
        "detection_id": det.id,
        "wind_speed_ms": det.wind_speed_ms,
        "wind_direction_deg": det.wind_direction_deg,
        "sar_validity": det.sar_validity,
        "lookalike_wind_risk": det.lookalike_wind_risk,
        "drift_bearing_deg": det.drift_bearing_deg,
        "drift_24h_km": det.drift_24h_km,
        "wind_fetched_at": det.wind_fetched_at,
    }

    @app.post("/detections/{detection_id}/optical")
async def revalidate_optical(detection_id: str, db: Session = Depends(get_db)):

    det = db.query(Detection).filter(Detection.id == detection_id).first()

    if not det:
        raise HTTPException(404, "Detection not found")

    if not det.polygon_geojson:
        raise HTTPException(400, "Detection missing polygon")

    polygon = json.loads(det.polygon_geojson)

    # ── centroid extraction (robust) ─────────────────────────────
    coords = polygon["coordinates"][0]

    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]

    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)

    # ── timestamp fallback ───────────────────────────────────────
    scene_timestamp = (
        det.detected_at.isoformat()
        if det.detected_at
        else datetime.utcnow().isoformat()
    )

    result = _optical_validator.validate(
        lat=centroid_lat,
        lon=centroid_lon,
        detection_polygon=polygon,
        scene_timestamp=scene_timestamp,
        local_scene_path=None,
    )

    # ── save back to DB ─────────────────────────────────────────
    det.optical_verdict = result["optical_verdict"]
    det.optical_reason = result["optical_reason"]
    det.optical_confidence = result["optical_confidence"]
    det.optical_cloud_fraction = result["optical_cloud_fraction"]
    det.optical_osi = result["optical_osi"]
    det.optical_swiri = result["optical_swiri"]
    det.optical_ndwi = result["optical_ndwi"]
    det.optical_scene_name = result["optical_scene_name"]
    det.optical_scene_timestamp = result["optical_scene_timestamp"]
    det.optical_thumbnail_rgb = result["optical_thumbnail_rgb"]
    det.optical_thumbnail_falsecolour = result["optical_thumbnail_falsecolour"]
    det.optical_validated_at = result["optical_validated_at"]

    db.commit()
    db.refresh(det)

    return {
        "detection_id": detection_id,
        "optical_verdict": det.optical_verdict,
        "optical_confidence": det.optical_confidence,
        "optical_cloud_fraction": det.optical_cloud_fraction,
        "optical_osi": det.optical_osi,
        "optical_swiri": det.optical_swiri,
        "optical_ndwi": det.optical_ndwi,
        "optical_scene_name": det.optical_scene_name,
        "optical_validated_at": det.optical_validated_at,
    }

    @app.get("/detections/{detection_id}/optical/thumbnail/{kind}")
def get_optical_thumbnail(detection_id: str, kind: str, db: Session = Depends(get_db)):
    import base64
    from fastapi.responses import Response

    det = db.query(Detection).filter(Detection.id == detection_id).first()

    if not det:
        raise HTTPException(404, "Detection not found")

    field = (
        "optical_thumbnail_rgb"
        if kind == "rgb"
        else "optical_thumbnail_falsecolour"
    )

    b64_uri = getattr(det, field, None)

    if not b64_uri:
        raise HTTPException(404, "Thumbnail not available")

    img_bytes = base64.b64decode(b64_uri.split(",")[1])

    return Response(content=img_bytes, media_type="image/png")

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

# ─────────────────────────────────────────────────────────────
# PATCH YOUR EXISTING /alerts/dispatch ROUTE ONLY
# Replace your current dispatch_alerts() with this version.
# Nothing else in your file needs removal.
# ─────────────────────────────────────────────────────────────

@app.post("/alerts/dispatch")
def dispatch_alerts(data: AlertDispatch, db: Session = Depends(get_db)):
    detection = db.query(Detection).filter(Detection.id == data.detection_id).first()

    if not detection:
        return {"error": "Detection not found"}

    # ── LOOKALIKE ALERT GATE (NEW) ───────────────────────────
    # Fail-open behavior:
    # If column/value doesn't exist, alerts still proceed.
    lookalike_passed = getattr(detection, "lookalike_passed", True)
    lookalike_score = getattr(detection, "lookalike_score", None)
    lookalike_label = getattr(detection, "lookalike_label", None)

    if lookalike_passed is False:
        import logging

        logging.getLogger(__name__).info(
            "Detection suppressed by look-alike classifier "
            "(score=%s, label=%s)",
            str(lookalike_score),
            str(lookalike_label),
        )

        return {
            "detection_id": detection.id,
            "status": "suppressed",
            "reason": "Blocked by look-alike classifier",
            "lookalike_score": lookalike_score,
            "lookalike_label": lookalike_label,
        }
    # ─────────────────────────────────────────────────────────

    from backend.alerts import send_spill_alert

    det_dict = {
        "id": detection.id,
        "confidence": detection.confidence,
        "area_km2": detection.area_km2,
        "detected_at": detection.detected_at.isoformat(),
        "scene": detection.scene,
        "spill_pixels": detection.spill_pixels,

        # Include classifier fields too
        "lookalike_score": lookalike_score,
        "lookalike_label": lookalike_label,
        "lookalike_passed": lookalike_passed,
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

            results.append({
                "email": recipient,
                "status": "sent"
            })

        except Exception as e:
            results.append({
                "email": recipient,
                "status": f"failed: {str(e)}"
            })

    detection.alert_sent = True
    detection.alert_recipients = json.dumps(data.recipients)

    db.commit()

    return {
        "detection_id": data.detection_id,
        "results": results,
        "total_sent": len(
            [r for r in results if r["status"] == "sent"]
        )
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
