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


# ─── SERIALIZATION ─────────────────────────────────────────

def polygon_centroid(polygon: Optional[dict]):
    """Mean vertex of a GeoJSON polygon ring, or (None, None)."""
    if not polygon or not polygon.get("coordinates"):
        return None, None
    ring = polygon["coordinates"][0]
    if not ring:
        return None, None
    lat = sum(p[1] for p in ring) / len(ring)
    lon = sum(p[0] for p in ring) / len(ring)
    return lat, lon


def serialize_detection(d: Detection, include_thumbnails: bool = True) -> dict:
    """
    Single source of truth for the detection JSON shape, so the list and
    detail endpoints can't drift apart. Thumbnails are base64 blobs, so
    the list endpoint omits them to keep the payload small.
    """
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

    polygon = json.loads(d.polygon_geojson) if d.polygon_geojson else None
    centroid_lat, centroid_lon = polygon_centroid(polygon)

    payload = {
        "id": d.id,
        "watch_zone_id": d.watch_zone_id,
        "scene": d.scene,
        "detected_at": d.detected_at.isoformat() if d.detected_at else None,
        "detected": d.detected,
        "confidence": d.confidence,
        "area_km2": d.area_km2,
        "spill_pixels": d.spill_pixels,
        "polygon": polygon,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
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
        "optical_validated_at": d.optical_validated_at,
        "ais_vessels_found": d.ais_vessels_found,
        "ais_top_suspect": ais_top_suspect,
        "ais_search_radius_nm": d.ais_search_radius_nm,
        "ais_queried_at": d.ais_queried_at,
        "ais_data_source": d.ais_data_source,
        "ais_note": d.ais_note,
    }

    if include_thumbnails:
        payload["optical_thumbnail_rgb"] = d.optical_thumbnail_rgb
        payload["optical_thumbnail_falsecolour"] = d.optical_thumbnail_falsecolour

    return payload


# ─── DETECTION ENGINE ──────────────────────────────────────

# Column names on the Detection model, used to filter run_detection()'s
# result dict — it carries a few keys (ais_match_found, ais_reason) that
# aren't persisted, and silently setattr-ing those would be a no-op at best.
DETECTION_COLUMNS = {c.name for c in Detection.__table__.columns}

# Set by run_detection() / handled explicitly, so the generic mapper skips them.
_MANUALLY_MAPPED = {"polygon", "detected_at", "polygon_geojson", "alert_sent", "status", "id"}


def _parse_scene_timestamp(value: Optional[str]) -> Optional[datetime]:
    """ISO-8601 'Z' string from the scene filename -> naive UTC datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def apply_result_to_detection(det: Detection, result: dict) -> None:
    """Copy run_detection() output onto the ORM row, JSON-encoding containers."""
    for key, value in result.items():
        if key in _MANUALLY_MAPPED or key not in DETECTION_COLUMNS:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        setattr(det, key, value)


def run_scan_job(scan_id: str, watch_zone_id: Optional[str] = None):
    from backend.database import SessionLocal
    db = SessionLocal()

    # Create the row up front with status="running" so the client can poll
    # this id for progress instead of guessing how long the scan will take.
    det = Detection(
        id=scan_id,
        watch_zone_id=watch_zone_id,
        status="running",
        detected=False,
        detected_at=datetime.utcnow(),
    )
    db.add(det)
    db.commit()

    try:
        import zipfile
        from backend.preprocess import find_bands
        from backend.detect import run_detection, _extract_scene_timestamp

        scenes_dir = Path("data/scenes")
        for zf in scenes_dir.glob("*.zip"):
            if not (scenes_dir / zf.stem).exists():
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(scenes_dir)

        safe_dirs = sorted(scenes_dir.glob("*.SAFE"))
        if not safe_dirs:
            raise RuntimeError("No .SAFE scene found in data/scenes/")

        scene = safe_dirs[0]
        vv_path, vh_path = find_bands(str(scene))
        if not vv_path:
            raise RuntimeError(f"Could not find VV band in {scene.name}")

        # run_detection() is the real pipeline: it georeferences the mask into
        # actual polygons, runs the look-alike classifier, and pulls wind /
        # optical / AIS context. Keep this as the single detection path so the
        # scan endpoint can't drift away from it.
        result = run_detection(vv_path, vh_path)

        det.detected_at = _parse_scene_timestamp(_extract_scene_timestamp(vv_path))
        det.polygon_geojson = json.dumps(result["polygon"]) if result.get("polygon") else None
        det.alert_sent = False
        apply_result_to_detection(det, result)
        det.status = "complete"
        db.commit()
        print(f"Scan {scan_id} saved — confidence: {det.confidence}% | area: {det.area_km2} km2")
    except Exception as e:
        print(f"Scan error: {e}")
        db.rollback()
        # Surface the failure to the client instead of leaving it polling forever.
        failed = db.query(Detection).filter(Detection.id == scan_id).first()
        if failed:
            failed.status = "failed"
            db.commit()
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
        "detections": [serialize_detection(d, include_thumbnails=False) for d in dets],
        "total": len(dets)
    }

@app.get("/detections/{detection_id}")
def get_detection(detection_id: str, db: Session = Depends(get_db)):
    d = db.query(Detection).filter(Detection.id == detection_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Detection not found")
    return serialize_detection(d)

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