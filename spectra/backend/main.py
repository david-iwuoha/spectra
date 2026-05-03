from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
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
from backend.database import init_db, get_db, Detection, WatchZone, AlertLog
from fastapi.responses import Response as FastAPIResponse
from backend.report_generator import ReportGenerator
from backend.ais_attribution import AISAttribution

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

# Init DB and Core Services
init_db()
_wind_context = WindContextLayer()
_optical_validator = OpticalValidator()
_report_generator = ReportGenerator()
_ais_attribution = AISAttribution()

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
                detected_at=datetime(2024, 1, 17, 17, 53, 33)
            )
            db.add(det)
            db.commit()
            return
        from backend.detect import run_detection
        from backend.preprocess import find_bands
        from pathlib import Path as _Path
        import zipfile as _zf

        scenes_dir = _Path("data/scenes")
        for zf in scenes_dir.glob("*.zip"):
            if not (scenes_dir / zf.stem).exists():
                with _zf.ZipFile(zf, "r") as z:
                    z.extractall(scenes_dir)
        safe_dirs = list(scenes_dir.glob("*.SAFE"))
        if not safe_dirs:
            raise RuntimeError("No .SAFE scene found in data/scenes/")
        scene = safe_dirs[0]
        vv_path, vh_path = find_bands(str(scene))
        if not vv_path:
            raise RuntimeError("Could not find VV band in scene")

        result = run_detection(vv_path, vh_path)
        polygon = result.get("polygon")

        det = Detection(
            id=scan_id,
            watch_zone_id=watch_zone_id,
            scene=scene.name,
            detected_at=datetime(2024, 1, 17, 17, 53, 33),
            detected=result.get("detected", False),
            confidence=result["confidence"],
            area_km2=result["area_km2"],
            spill_pixels=result["spill_pixels"],
            polygon_geojson=json.dumps(polygon) if polygon else None,
            alert_sent=False,
            lookalike_score=result.get("lookalike_score"),
            lookalike_label=result.get("lookalike_label"),
            lookalike_passed=result.get("lookalike_passed"),
            wind_speed_ms=result.get("wind_speed_ms"),
            wind_direction_deg=result.get("wind_direction_deg"),
            wind_u=result.get("wind_u"),
            wind_v=result.get("wind_v"),
            sar_validity=result.get("sar_validity"),
            sar_validity_detail=result.get("sar_validity_detail"),
            lookalike_wind_risk=result.get("lookalike_wind_risk"),
            lookalike_wind_note=result.get("lookalike_wind_note"),
            drift_bearing_deg=result.get("drift_bearing_deg"),
            drift_speed_ms=result.get("drift_speed_ms"),
            drift_24h_km=result.get("drift_24h_km"),
            wind_fetched_at=result.get("wind_fetched_at"),
            wind_data_source=result.get("wind_data_source"),
            status="complete"
        )
        db.add(det)
        db.commit()
        print(f"Scan {scan_id} saved — confidence: {det.confidence}% | area: {result['area_km2']} km2")

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
    drift_vector = None
    if d.drift_bearing_deg is not None:
        drift_vector = {
            "bearing_deg": d.drift_bearing_deg,
            "speed_ms": d.drift_speed_ms,
            "6h_km": round(d.drift_24h_km / 4, 2) if d.drift_24h_km else None,
            "12h_km": round(d.drift_24h_km / 2, 2) if d.drift_24h_km else None,
            "24h_km": d.drift_24h_km,
        }
    ais_top_suspect = None
    if d.ais_top_suspect:
        try:
            ais_top_suspect = json.loads(d.ais_top_suspect)
        except Exception:
            pass
    return {
        "id": d.id,
        "watch_zone_id": d.watch_zone_id,
        "scene": d.scene,
        "detected_at": d.detected_at.isoformat() if d.detected_at else None,
        "detected": d.detected,
        "confidence": d.confidence,
        "area_km2": d.area_km2,
        "spill_pixels": d.spill_pixels,
        "polygon": json.loads(d.polygon_geojson) if d.polygon_geojson else None,
        "alert_sent": d.alert_sent,
        "status": d.status,
        "lookalike_score": d.lookalike_score,
        "lookalike_label": d.lookalike_label,
        "lookalike_passed": d.lookalike_passed,
        "wind_speed_ms": d.wind_speed_ms,
        "wind_direction_deg": d.wind_direction_deg,
        "wind_u": d.wind_u,
        "wind_v": d.wind_v,
        "sar_validity": d.sar_validity,
        "sar_validity_detail": d.sar_validity_detail,
        "lookalike_wind_risk": d.lookalike_wind_risk,
        "lookalike_wind_note": d.lookalike_wind_note,
        "drift_bearing_deg": d.drift_bearing_deg,
        "drift_speed_ms": d.drift_speed_ms,
        "drift_24h_km": d.drift_24h_km,
        "drift_vector": drift_vector,
        "wind_fetched_at": d.wind_fetched_at,
        "wind_data_source": d.wind_data_source,
        "optical_verdict": d.optical_verdict,
        "optical_confidence": d.optical_confidence,
        "optical_cloud_fraction": d.optical_cloud_fraction,
        "optical_osi": d.optical_osi,
        "optical_swiri": d.optical_swiri,
        "optical_ndwi": d.optical_ndwi,
        "optical_reason": d.optical_reason,
        "optical_scene_name": d.optical_scene_name,
        "optical_thumbnail_rgb": d.optical_thumbnail_rgb,
        "optical_thumbnail_falsecolour": d.optical_thumbnail_falsecolour,
        "optical_validated_at": d.optical_validated_at,
        "ais_vessels_found": d.ais_vessels_found,
        "ais_top_suspect": ais_top_suspect,
        "ais_search_radius_nm": d.ais_search_radius_nm,
        "ais_queried_at": d.ais_queried_at,
        "ais_data_source": d.ais_data_source,
        "ais_note": d.ais_note,
    }

@app.get("/detections/{detection_id}/report")
async def download_report(detection_id: str, db: Session = Depends(get_db)):
    if not _report_generator.is_available():
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable. Install system deps (libpango, weasyprint)."
        )

    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")

    try:
        pdf_bytes = _report_generator.generate(det)
        filename = f"spectra_detection_{detection_id}.pdf"
        return FastAPIResponse(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

@app.post("/detections/{detection_id}/wind")
def refresh_wind(detection_id: str, db: Session = Depends(get_db)):
    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det or not det.polygon_geojson:
        raise HTTPException(status_code=400, detail="Detection missing or invalid")

    polygon = json.loads(det.polygon_geojson)
    coords = polygon["coordinates"][0]
    centroid_lat = sum(p[1] for p in coords) / len(coords)
    centroid_lon = sum(p[0] for p in coords) / len(coords)

    wind = _wind_context.get_context(
        lat=centroid_lat,
        lon=centroid_lon,
        timestamp=det.detected_at.isoformat() if det.detected_at else datetime.utcnow().isoformat(),
    )

    det.wind_speed_ms = wind["wind_speed_ms"]
    det.wind_direction_deg = wind["wind_direction_deg"]
    det.wind_u = wind["wind_u"]
    det.wind_v = wind["wind_v"]
    det.sar_validity = wind["sar_validity"]
    det.lookalike_wind_risk = wind["lookalike_wind_risk"]
    det.wind_fetched_at = wind["wind_fetched_at"]

    if wind["drift_vector"]:
        det.drift_bearing_deg = wind["drift_vector"]["bearing_deg"]
        det.drift_24h_km = wind["drift_vector"]["24h_km"]
        det.drift_geojson = json.dumps(drift_arrow_geojson(centroid_lat, centroid_lon, wind["drift_vector"]))

    db.commit()
    return {"status": "success", "wind_speed": det.wind_speed_ms}

@app.post("/detections/{detection_id}/optical")
async def revalidate_optical(detection_id: str, db: Session = Depends(get_db)):
    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det or not det.polygon_geojson:
        raise HTTPException(400, "Invalid detection")

    polygon = json.loads(det.polygon_geojson)
    coords = polygon["coordinates"][0]
    centroid_lat = sum(p[1] for p in coords) / len(coords)
    centroid_lon = sum(p[0] for p in coords) / len(coords)

    result = _optical_validator.validate(
        lat=centroid_lat, lon=centroid_lon,
        detection_polygon=polygon,
        scene_timestamp=det.detected_at.isoformat() if det.detected_at else datetime.utcnow().isoformat()
    )

    det.optical_verdict = result["optical_verdict"]
    det.optical_confidence = result["optical_confidence"]
    det.optical_cloud_fraction = result["optical_cloud_fraction"]
    det.optical_thumbnail_rgb = result["optical_thumbnail_rgb"]
    det.optical_validated_at = result["optical_validated_at"]

    db.commit()
    return {"status": "success", "verdict": det.optical_verdict}

@app.get("/detections/{detection_id}/optical/thumbnail/{kind}")
def get_optical_thumbnail(detection_id: str, kind: str, db: Session = Depends(get_db)):
    import base64
    from fastapi.responses import Response
    det = db.query(Detection).filter(Detection.id == detection_id).first()
    field = "optical_thumbnail_rgb" if kind == "rgb" else "optical_thumbnail_falsecolour"
    b64_uri = getattr(det, field, None)
    if not b64_uri: raise HTTPException(404, "Not available")
    img_bytes = base64.b64decode(b64_uri.split(",")[1])
    return Response(content=img_bytes, media_type="image/png")

# ─── AIS ATTRIBUTION (NEW) ──────────────────────────────────

@app.post("/detections/{detection_id}/ais")
async def run_ais_attribution(
    detection_id: str,
    radius_nm: float = 10.0,
    db: Session = Depends(get_db),
):
    if not _ais_attribution.is_available():
        raise HTTPException(status_code=503, detail="AISSTREAM_API_KEY not configured.")

    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det or not det.polygon_geojson:
        raise HTTPException(status_code=400, detail="Detection missing polygon")

    # Centroid extraction
    polygon = json.loads(det.polygon_geojson)
    coords = polygon["coordinates"][0]
    centroid_lat = sum(p[1] for p in coords) / len(coords)
    centroid_lon = sum(p[0] for p in coords) / len(coords)

    ais = _ais_attribution.attribute(
        lat=centroid_lat,
        lon=centroid_lon,
        timestamp=det.detected_at.isoformat() if det.detected_at else datetime.utcnow().isoformat(),
        radius_nm=radius_nm,
    )

    det.ais_vessels_found    = ais["ais_vessels_found"]
    det.ais_candidates       = json.dumps(ais["ais_candidates"])
    det.ais_top_suspect      = json.dumps(ais["ais_top_suspect"])
    det.ais_search_radius_nm = ais["ais_search_radius_nm"]
    det.ais_queried_at       = ais["ais_queried_at"]
    det.ais_data_source      = ais["ais_data_source"]
    det.ais_note             = ais["ais_note"]

    db.commit()
    db.refresh(det)

    return {
        "detection_id": detection_id,
        "ais_vessels_found": det.ais_vessels_found,
        "ais_top_suspect": json.loads(det.ais_top_suspect) if det.ais_top_suspect else None,
        "ais_candidates": json.loads(det.ais_candidates) if det.ais_candidates else [],
    }


# ─── ALERTS & SCENES ───────────────────────────────────────

@app.post("/alerts/dispatch")
def dispatch_alerts(data: AlertDispatch, db: Session = Depends(get_db)):
    detection = db.query(Detection).filter(Detection.id == data.detection_id).first()
    if not detection: return {"error": "Not found"}

    lookalike_passed = getattr(detection, "lookalike_passed", True)
    if lookalike_passed is False:
        return {"status": "suppressed", "reason": "Look-alike check failed"}

    from backend.alerts import send_spill_alert
    det_dict = {"id": detection.id, "confidence": detection.confidence, "area_km2": detection.area_km2, "scene": detection.scene, "detected_at": detection.detected_at.isoformat() if detection.detected_at else "", "recipients": data.recipients}
    
    results = []
    for recipient in data.recipients:
        try:
            send_spill_alert(det_dict)
            db.add(AlertLog(detection_id=detection.id, recipient=recipient, sent_at=datetime.utcnow(), success=True))
            results.append({"email": recipient, "status": "sent"})
        except:
            results.append({"email": recipient, "status": "failed"})

    detection.alert_sent = True
    db.commit()
    return {"results": results}

@app.get("/scenes")
def list_scenes():
    scenes = []
    scenes_dir = Path("data/scenes")
    if scenes_dir.exists():
        for f in scenes_dir.glob("*.SAFE"): scenes.append({"name": f.name, "type": "SAFE"})
    return {"scenes": scenes}

@app.delete("/detections/{detection_id}")
def delete_detection(detection_id: str, db: Session = Depends(get_db)):
    d = db.query(Detection).filter(Detection.id == detection_id).first()
    if not d: return {"error": "Not found"}
    db.delete(d)
    db.commit()
    return {"message": "Deleted"}

@app.post("/scan")
def trigger_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    scan_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_scan_job, scan_id, request.watch_zone_id)
    return {"scan_id": scan_id, "status": "running"}