"""
Spectra Phase E — Sentinel-2 Cross-Validation
================================================
After U-Net + LookalikeClassifier + WindContext produce a detection,
this module asks: does Sentinel-2 optical data confirm it?

Pipeline position:
    U-Net → Lookalike → Wind → OpticalValidator (Phase E) → alert gate → DB

What it produces per detection:
    optical_verdict         "confirmed" | "unconfirmed" | "inconclusive" | "unavailable"
    optical_confidence      0-1 score
    optical_cloud_fraction  % cloud cover over detection area (from SCL)
    optical_osi             Oil Spill Index (B03+B04)/B02
    optical_swiri           SWIR suppression ratio B11/(B8A+B11)
    optical_ndwi            Normalised Difference Water Index
    optical_thumbnail_rgb         base64 PNG — true colour crop (B04/B03/B02)
    optical_thumbnail_falsecolour base64 PNG — SWIR false colour (B11/B8A/B04)
    optical_scene_name      name of the S2 scene used
    optical_validated_at    ISO-8601 timestamp

Spectral science:
    OSI = (B03 + B04) / B02
        Oil raises visible reflectance ratio vs clean water. OSI > 1.8 = possible oil,
        OSI > 2.5 = high confidence oil.

    SWIRI = B11 / (B8A + B11)
        Oil suppresses SWIR reflectance. SWIRI < 0.30 over water = oil signature.
        Source: E-OSI framework (88% accuracy, >95% specificity vs look-alikes).

    NDWI = (B03 - B08) / (B03 + B08)
        Confirms the detection area is over water (not land misidentification).

    False-colour composite: B11/B8A/B04 (SWIR-NIR-Red)
        Oil appears deep black/dark blue. Industry standard for visual confirmation.

Usage:
    from backend.optical_validator import OpticalValidator
    validator = OpticalValidator()
    result = validator.validate(
        lat=3.5, lon=6.2,
        detection_polygon=geojson_polygon,
        scene_timestamp="2024-11-14T09:30:00Z",
        local_scene_path=None,   # or Path("data/scenes/S2A_MSIL2A_...SAFE")
    )
    detection.update(result)

CDSE credentials:
    export COPERNICUS_USER=your@email.com
    export COPERNICUS_PASSWORD=yourpassword

WAF note:
    GitHub Codespace IPs are blocked by Copernicus WAF for S2 downloads.
    Scene SEARCH works fine. Place downloaded .SAFE or .zip in data/scenes/
    and the validator will auto-detect it. Or pass local_scene_path explicitly.

Dependencies:
    pip install rasterio pillow shapely --break-system-packages
"""

import base64
import io
import logging
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spectral thresholds (peer-reviewed literature)
# ---------------------------------------------------------------------------

OSI_OIL_MIN          = 1.8    # (B03+B04)/B02 — possible oil
OSI_OIL_CONFIDENT    = 2.5    # strong oil signature
SWIRI_OIL_MAX        = 0.30   # B11/(B8A+B11) — above: not oil
SWIRI_OIL_CONFIDENT  = 0.20   # below: strong SWIR suppression = oil
NDWI_WATER_MIN       = 0.0    # must be positive to confirm water body
CLOUD_COVER_MAX      = 60.0   # skip scene if scene-level cloud > this %
CLOUD_OVER_SPILL_MAX = 30.0   # inconclusive if cloud % over polygon > this
TEMPORAL_WINDOW_DAYS = 3      # search ±3 days of SAR acquisition
SCL_CLOUD_CLASSES    = {8, 9, 10}  # medium cloud, high cloud, thin cirrus


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OpticalValidator:
    """
    Sentinel-2 optical cross-validator. Instantiate once at module level.
    Thread-safe. All heavy work is inside validate().
    """

    def __init__(self):
        self._cdse_user     = os.environ.get("COPERNICUS_USER", "")
        self._cdse_password = os.environ.get("COPERNICUS_PASSWORD", "")
        self._has_credentials = bool(self._cdse_user and self._cdse_password)

        if not self._has_credentials:
            logger.warning(
                "CDSE_USER / CDSE_PASSWORD not set. "
                "Optical validator will use local scenes only (data/scenes/)."
            )

        self._rasterio_ok = self._check_deps()

    def _check_deps(self) -> bool:
        missing = []
        for pkg in ("rasterio", "PIL", "shapely"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            logger.warning(
                "Missing packages for optical validation: %s. "
                "Run: pip install rasterio pillow shapely --break-system-packages",
                missing,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(
        self,
        lat: float,
        lon: float,
        detection_polygon: dict,
        scene_timestamp: str,
        local_scene_path: Optional[Path] = None,
    ) -> dict:
        """
        Run optical cross-validation. Returns dict with all optical_* fields.

        Args:
            lat, lon:             Spill centroid (decimal degrees)
            detection_polygon:    GeoJSON Polygon geometry of the SAR detection
            scene_timestamp:      ISO-8601 UTC of Sentinel-1 acquisition
            local_scene_path:     Path to .SAFE dir or .zip — use for WAF workaround.
                                  If None, auto-searches data/scenes/ then tries CDSE API.
        """
        if not self._rasterio_ok:
            return self._unavailable("Dependencies not installed (rasterio/pillow/shapely)")

        try:
            scene_path = self._resolve_scene(lat, lon, scene_timestamp, local_scene_path)
            if scene_path is None:
                return self._unavailable(
                    f"No S2 L2A scene within ±{TEMPORAL_WINDOW_DAYS} days. "
                    "Download manually and place in data/scenes/ or pass local_scene_path."
                )
            return self._process_scene(scene_path, lat, lon, detection_polygon, scene_timestamp)

        except Exception as exc:
            logger.error("Optical validation error: %s", exc, exc_info=True)
            return self._unavailable(f"Processing error: {exc}")

    # ------------------------------------------------------------------
    # Scene resolution
    # ------------------------------------------------------------------

    def _resolve_scene(self, lat, lon, timestamp, local_path) -> Optional[Path]:
        if local_path is not None:
            p = Path(local_path)
            if p.exists():
                logger.info("Using provided local S2 scene: %s", p.name)
                return p
            logger.warning("Provided local_scene_path does not exist: %s", p)

        auto = self._find_local_scene(timestamp)
        if auto:
            return auto

        if self._has_credentials:
            return self._search_and_download(lat, lon, timestamp)

        return None

    def _find_local_scene(self, timestamp: str) -> Optional[Path]:
        """Scan data/scenes/ for any S2 L2A scene within the temporal window."""
        scenes_dir = Path("data/scenes")
        if not scenes_dir.exists():
            return None

        dt           = _parse_ts(timestamp)
        window_start = dt - timedelta(days=TEMPORAL_WINDOW_DAYS)
        window_end   = dt + timedelta(days=TEMPORAL_WINDOW_DAYS)

        candidates = (
            list(scenes_dir.glob("S2*MSIL2A*.SAFE")) +
            list(scenes_dir.glob("S2*MSIL2A*.zip"))
        )

        for path in sorted(candidates):
            parts = path.name.split("_")
            if len(parts) < 3:
                continue
            try:
                acq = datetime.strptime(parts[2][:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if window_start <= acq <= window_end:
                logger.info("Auto-detected local S2 scene: %s", path.name)
                return path

        return None

    def _search_and_download(self, lat, lon, timestamp) -> Optional[Path]:
        try:
            import requests
        except ImportError:
            return None

        dt        = _parse_ts(timestamp)
        start_str = (dt - timedelta(days=TEMPORAL_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = (dt + timedelta(days=TEMPORAL_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        buf       = 0.15
        bbox_wkt  = (
            f"POLYGON(({lon-buf} {lat-buf},{lon+buf} {lat-buf},"
            f"{lon+buf} {lat+buf},{lon-buf} {lat+buf},{lon-buf} {lat-buf}))"
        )

        url = (
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
            f"?$filter=Collection/Name eq 'SENTINEL-2'"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
            f" and ContentDate/Start gt {start_str}"
            f" and ContentDate/Start lt {end_str}"
            f" and Attributes/OData.CSC.DoubleAttribute/any("
            f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {CLOUD_COVER_MAX})"
            f" and contains(Name,'MSIL2A')"
            f"&$orderby=ContentDate/Start asc&$top=1"
        )

        try:
            items = requests.get(url, timeout=30).json().get("value", [])
        except Exception as exc:
            logger.warning("CDSE S2 search failed: %s", exc)
            return None

        if not items:
            logger.info("No S2 scene found in CDSE for (%.4f, %.4f) ±%d days", lat, lon, TEMPORAL_WINDOW_DAYS)
            return None

        product    = items[0]
        product_id = product["Id"]
        name       = product["Name"]
        cloud_pct  = next(
            (a["Value"] for a in product.get("Attributes", []) if a.get("Name") == "cloudCover"),
            "?"
        )
        logger.info("Found S2 scene: %s (cloud=%.1f%%)", name, float(cloud_pct) if cloud_pct != "?" else 0)

        return self._download_scene(product_id, name)

    def _download_scene(self, product_id: str, name: str) -> Optional[Path]:
        try:
            import requests
        except ImportError:
            return None

        scenes_dir = Path("data/scenes")
        scenes_dir.mkdir(parents=True, exist_ok=True)
        out_zip = scenes_dir / f"{name}.zip"

        token = self._get_token()
        if not token:
            return None

        dl_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        try:
            resp = requests.get(
                dl_url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True, timeout=120,
            )
            if resp.status_code == 403:
                logger.warning(
                    "Copernicus WAF blocked download (403). "
                    "Download manually and place in data/scenes/. Scene: %s", name
                )
                return None
            resp.raise_for_status()
            with open(out_zip, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        except Exception as exc:
            logger.error("S2 download error: %s", exc)
            if out_zip.exists():
                out_zip.unlink()
            return None

        with zipfile.ZipFile(out_zip) as zf:
            zf.extractall(scenes_dir)
        out_zip.unlink(missing_ok=True)

        safes = list(scenes_dir.glob(f"{name.replace('.zip','')}*.SAFE"))
        return safes[0] if safes else None

    def _get_token(self) -> Optional[str]:
        try:
            import requests
            resp = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token",
                data={
                    "client_id":  "cdse-public",
                    "username":   self._cdse_user,
                    "password":   self._cdse_password,
                    "grant_type": "password",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["access_token"]
        except Exception as exc:
            logger.error("CDSE token failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Scene processing
    # ------------------------------------------------------------------

    def _process_scene(self, scene_path, lat, lon, polygon, timestamp) -> dict:
        if str(scene_path).endswith(".zip"):
            scene_path = self._unzip_safe(scene_path)
            if scene_path is None:
                return self._unavailable("Failed to unzip scene")

        bands = self._locate_bands(scene_path)
        if not bands:
            return self._unavailable(f"No band files found in {scene_path.name}")

        logger.info("Processing bands from %s", scene_path.name)

        cloud_fraction = self._cloud_fraction_over_polygon(bands.get("SCL"), polygon)

        if cloud_fraction > CLOUD_OVER_SPILL_MAX:
            thumbnails = self._generate_thumbnails(bands, lat, lon)
            return self._build_result(
                verdict="inconclusive",
                reason=f"Cloud cover over detection: {cloud_fraction:.1f}% (threshold {CLOUD_OVER_SPILL_MAX}%)",
                cloud_fraction=cloud_fraction,
                scene_name=scene_path.name,
                scene_timestamp=timestamp,
                thumbnails=thumbnails,
                indices={},
                confidence=0.0,
            )

        indices    = self._compute_indices(bands, lat, lon)
        verdict, reason, confidence = _assess_verdict(indices, cloud_fraction)
        thumbnails = self._generate_thumbnails(bands, lat, lon)

        logger.info("Optical verdict: %s (%.2f) — %s", verdict, confidence, reason[:80])

        return self._build_result(
            verdict=verdict, reason=reason, confidence=confidence,
            cloud_fraction=cloud_fraction, scene_name=scene_path.name,
            scene_timestamp=timestamp, thumbnails=thumbnails, indices=indices,
        )

    def _locate_bands(self, scene_path: Path) -> dict:
        target = {
            "B02": None, "B03": None, "B04": None,
            "B08": None, "B8A": None, "B11": None, "B12": None,
            "SCL": None,
        }
        for jp2 in scene_path.rglob("*.jp2"):
            stem = jp2.stem
            for band in target:
                if target[band] is None and (f"_{band}_" in stem or stem.endswith(f"_{band}")):
                    target[band] = jp2
        return {k: v for k, v in target.items() if v is not None}

    def _cloud_fraction_over_polygon(self, scl_path, polygon) -> float:
        if scl_path is None or not polygon:
            return 0.0
        try:
            import rasterio
            from rasterio.mask import mask as rio_mask
            from shapely.geometry import shape

            with rasterio.open(scl_path) as src:
                geom = shape(polygon)
                out, _ = rio_mask(src, [geom.__geo_interface__], crop=True, nodata=255)
                scl    = out[0].astype(int)
                valid  = scl[scl != 255]
                if valid.size == 0:
                    return 0.0
                cloud = np.isin(valid, list(SCL_CLOUD_CLASSES)).sum()
                return float(cloud / valid.size * 100)
        except Exception as exc:
            logger.debug("Cloud fraction check failed: %s", exc)
            return 0.0

    def _compute_indices(self, bands: dict, lat: float, lon: float) -> dict:
        RADIUS = 30   # pixels — ~600m at 20m res, ~300m at 10m res

        def sample(path):
            if path is None:
                return None
            try:
                import rasterio
                with rasterio.open(path) as src:
                    row, col = src.index(lon, lat)
                    r0 = max(0, row - RADIUS); r1 = min(src.height, row + RADIUS)
                    c0 = max(0, col - RADIUS); c1 = min(src.width,  col + RADIUS)
                    win  = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)
                    data = src.read(1, window=win).astype(float)
                    data[data == 0] = np.nan
                    return float(np.nanmedian(data / 10000.0))
            except Exception as exc:
                logger.debug("Band sample failed: %s", exc)
                return None

        b02 = sample(bands.get("B02"))
        b03 = sample(bands.get("B03"))
        b04 = sample(bands.get("B04"))
        b08 = sample(bands.get("B08"))
        b8a = sample(bands.get("B8A"))
        b11 = sample(bands.get("B11"))

        osi   = round((b03 + b04) / b02, 4) if (b02 and b02 > 0 and b03 and b04) else None
        swiri = round(b11 / (b8a + b11), 4) if (b8a and b11 and (b8a + b11) > 0) else None
        ndwi  = round((b03 - b08) / (b03 + b08), 4) if (b03 and b08 and (b03 + b08) > 0) else None

        return {
            "b02": round(b02, 5) if b02 else None,
            "b03": round(b03, 5) if b03 else None,
            "b04": round(b04, 5) if b04 else None,
            "b08": round(b08, 5) if b08 else None,
            "b8a": round(b8a, 5) if b8a else None,
            "b11": round(b11, 5) if b11 else None,
            "osi": osi, "swiri": swiri, "ndwi": ndwi,
        }

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------

    def _generate_thumbnails(self, bands: dict, lat: float, lon: float) -> dict:
        CROP_PX = 500   # ~5km at 10m res
        out = {}
        combos = {
            "rgb":          [bands.get("B04"), bands.get("B03"), bands.get("B02")],
            "false_colour": [bands.get("B11"), bands.get("B8A"), bands.get("B04")],
        }
        for name, paths in combos.items():
            try:
                b64 = self._make_thumbnail(paths, lat, lon, CROP_PX)
                if b64:
                    out[name] = b64
            except Exception as exc:
                logger.debug("Thumbnail %s failed: %s", name, exc)
        return out

    def _make_thumbnail(self, band_paths, lat, lon, crop_px) -> Optional[str]:
        if any(p is None for p in band_paths):
            return None

        import rasterio
        from PIL import Image

        arrays = []
        for path in band_paths:
            with rasterio.open(path) as src:
                row, col = src.index(lon, lat)
                r0 = max(0, row - crop_px // 2)
                c0 = max(0, col - crop_px // 2)
                win  = rasterio.windows.Window(c0, r0, crop_px, crop_px)
                data = src.read(1, window=win).astype(float)
                arrays.append(data)

        rgb_channels = []
        for arr in arrays:
            valid = arr[arr > 0]
            if valid.size == 0:
                rgb_channels.append(np.zeros_like(arr, dtype=np.uint8))
                continue
            p2, p98  = np.percentile(valid, (2, 98))
            stretched = np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)
            rgb_channels.append((stretched * 255).astype(np.uint8))

        img = Image.fromarray(np.stack(rgb_channels, axis=-1), mode="RGB")
        img = img.resize((512, 512), Image.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_result(self, verdict, reason, confidence, cloud_fraction,
                      scene_name, scene_timestamp, thumbnails, indices=None) -> dict:
        return {
            "optical_verdict":               verdict,
            "optical_reason":                reason,
            "optical_confidence":            round(confidence, 3),
            "optical_cloud_fraction":        round(cloud_fraction, 1),
            "optical_scene_name":            scene_name,
            "optical_scene_timestamp":       scene_timestamp,
            "optical_osi":                   (indices or {}).get("osi"),
            "optical_swiri":                 (indices or {}).get("swiri"),
            "optical_ndwi":                  (indices or {}).get("ndwi"),
            "optical_thumbnail_rgb":         thumbnails.get("rgb"),
            "optical_thumbnail_falsecolour": thumbnails.get("false_colour"),
            "optical_validated_at":          datetime.now(timezone.utc).isoformat(),
        }

    def _unavailable(self, reason: str) -> dict:
        return {
            "optical_verdict":               "unavailable",
            "optical_reason":                reason,
            "optical_confidence":            0.0,
            "optical_cloud_fraction":        None,
            "optical_scene_name":            None,
            "optical_scene_timestamp":       None,
            "optical_osi":                   None,
            "optical_swiri":                 None,
            "optical_ndwi":                  None,
            "optical_thumbnail_rgb":         None,
            "optical_thumbnail_falsecolour": None,
            "optical_validated_at":          datetime.now(timezone.utc).isoformat(),
        }

    def _unzip_safe(self, zip_path: Path) -> Optional[Path]:
        try:
            extract_dir = zip_path.parent
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            safes = list(extract_dir.glob("*.SAFE"))
            return safes[0] if safes else None
        except Exception as exc:
            logger.error("Unzip failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Spectral verdict (pure function)
# ---------------------------------------------------------------------------

def _assess_verdict(indices: dict, cloud_fraction: float) -> tuple:
    """
    Returns (verdict, reason, confidence).
    verdict: "confirmed" | "unconfirmed" | "inconclusive"
    """
    osi   = indices.get("osi")
    swiri = indices.get("swiri")
    ndwi  = indices.get("ndwi")

    evidence = []
    signals  = []

    if ndwi is not None:
        if ndwi < NDWI_WATER_MIN:
            return (
                "inconclusive",
                f"NDWI={ndwi:.3f} — sample area includes land pixels, not open water",
                0.0,
            )
        evidence.append(f"NDWI={ndwi:.3f} (water confirmed)")

    if osi is not None:
        if osi >= OSI_OIL_CONFIDENT:
            signals.append(2)
            evidence.append(f"OSI={osi:.3f} — strong oil signature (>{OSI_OIL_CONFIDENT})")
        elif osi >= OSI_OIL_MIN:
            signals.append(1)
            evidence.append(f"OSI={osi:.3f} — moderate oil signature")
        else:
            signals.append(-1)
            evidence.append(f"OSI={osi:.3f} — below oil threshold, suggests clean water")
    else:
        evidence.append("OSI unavailable")

    if swiri is not None:
        if swiri <= SWIRI_OIL_CONFIDENT:
            signals.append(2)
            evidence.append(f"SWIRI={swiri:.3f} — strong SWIR suppression (oil)")
        elif swiri <= SWIRI_OIL_MAX:
            signals.append(1)
            evidence.append(f"SWIRI={swiri:.3f} — moderate SWIR suppression")
        else:
            signals.append(-1)
            evidence.append(f"SWIRI={swiri:.3f} — high SWIR reflectance, inconsistent with oil")
    else:
        evidence.append("SWIRI unavailable")

    if not signals:
        return "inconclusive", "Insufficient band data for spectral assessment", 0.0

    score      = sum(signals)
    confidence = max(0.0, min(1.0, score / (len(signals) * 2)))

    if score >= 2:
        verdict = "confirmed"
    elif score <= -1:
        verdict = "unconfirmed"
    else:
        verdict = "inconclusive"

    reason = " | ".join(evidence)
    if cloud_fraction > 0:
        reason += f" | Cloud over spill: {cloud_fraction:.1f}%"

    return verdict, reason, confidence


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)