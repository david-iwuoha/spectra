"""
Spectra Phase D — Wind Context Layer
======================================
Fetches ERA5 10m wind data (u10, v10) from the Copernicus Climate Data Store
for a given detection's location and timestamp, then derives:

  1. wind_speed_ms       — scalar wind speed in m/s
  2. wind_direction_deg  — meteorological convention (direction wind is coming FROM), 0-360
  3. sar_validity        — "valid" | "too_low" | "too_high" | "borderline"
  4. lookalike_wind_risk — "high" | "medium" | "low" (feeds back to Phase C context)
  5. drift_vector        — {"bearing_deg": float, "speed_ms": float, "6h_km": float, ...}
  6. wind_u / wind_v     — raw ERA5 components, stored for audit trail

Pipeline position:
    U-Net → LookalikeClassifier → WindContext (Phase D) → alert gate → DB

Usage in detect.py:
    from backend.wind_context import WindContextLayer
    wind = WindContextLayer()
    ctx = wind.get_context(lat=3.5, lon=6.2, timestamp="2024-11-14T09:30:00Z")
    detection.update(ctx)

On-demand re-fetch is exposed via FastAPI endpoint (see main.py integration below).

CDS API Setup (one-time):
    1. Register at https://cds.climate.copernicus.eu
    2. Accept the ERA5 dataset license at:
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
       (click "Terms of use" tab, accept, then the API will work)
    3. Get your Personal Access Token from https://cds.climate.copernicus.eu/profile
    4. Create the config file:
           nano ~/.cdsapirc
       Paste this (replace with your actual token):
           url: https://cds.climate.copernicus.eu/api
           key: <YOUR-PERSONAL-ACCESS-TOKEN>
    5. Install dependencies:
           pip install cdsapi xarray netcdf4 scipy

ERA5 variables used:
    10m_u_component_of_wind  (u10) — eastward wind component at 10m height
    10m_v_component_of_wind  (v10) — northward wind component at 10m height
    Spatial resolution: 0.25 x 0.25 degrees (~28km at equator)
    Temporal resolution: hourly
    Latency: ~5 days behind real time (reanalysis) — fine for Sentinel-1's 6-day revisit
"""

import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wind speed thresholds (m/s) from SAR oil spill detection literature
# ---------------------------------------------------------------------------

WIND_MIN_VISIBLE  = 2.0   # below: oil film may not dampen capillary waves enough
WIND_BORDERLINE   = 3.0   # 2-3 m/s: marginal detection zone
WIND_MAX_VALID    = 10.0  # above: wave breaking mixes/obscures the slick
WIND_HIGH_RISK    = 7.0   # above: elevated look-alike risk from wave patterns

# Drift: 3% empirical wind drift factor (standard in operational spill modeling)
DRIFT_FACTOR      = 0.03
DRIFT_DEFLECTION  = 15.0  # degrees right of downwind (Coriolis, northern hemisphere)

ERA5_DATASET = "reanalysis-era5-single-levels"


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class WindContextLayer:
    """
    Fetch ERA5 wind and compute all operationally relevant wind context
    for an oil spill detection.

    Instantiate once at module level in detect.py. Fails gracefully if CDS
    is not configured — all fields return None and wind_data_source is None,
    so the rest of the pipeline is unaffected.
    """

    def __init__(self):
        self._cds_available = self._check_cds_available()

    def _check_cds_available(self) -> bool:
        try:
            import cdsapi  # noqa: F401
        except ImportError:
            logger.warning(
                "cdsapi not installed. Wind context unavailable. "
                "Run: pip install cdsapi xarray netcdf4 scipy"
            )
            return False

        cdsapirc = Path.home() / ".cdsapirc"
        if not cdsapirc.exists():
            logger.warning(
                "~/.cdsapirc not found. Wind context unavailable. "
                "See backend/wind_context.py docstring for setup instructions."
            )
            return False

        logger.info("ERA5/CDS configured. Wind context layer ready.")
        return True

    def is_available(self) -> bool:
        return self._cds_available

    # ------------------------------------------------------------------
    # Public interface — called from detect.py and the API endpoint
    # ------------------------------------------------------------------

    def get_context(
        self,
        lat: float,
        lon: float,
        timestamp: str,
    ) -> dict:
        """
        Fetch ERA5 wind and return a context dict to merge into the detection result.

        Args:
            lat:       Spill centroid latitude (decimal degrees)
            lon:       Spill centroid longitude (decimal degrees)
            timestamp: ISO-8601 UTC string of Sentinel-1 scene acquisition time

        Returns:
            Dict with all wind fields. All values are None if CDS is unavailable.
        """
        if not self._cds_available:
            return self._unavailable_context("CDS API not configured")

        try:
            u10, v10 = self._fetch_era5_wind(lat, lon, timestamp)
            return self._compute_context(lat, lon, timestamp, u10, v10)
        except Exception as exc:
            logger.error("Wind fetch failed for (%.4f, %.4f): %s", lat, lon, exc, exc_info=True)
            return self._unavailable_context(f"Fetch error: {exc}")

    # ------------------------------------------------------------------
    # ERA5 fetch
    # ------------------------------------------------------------------

    def _fetch_era5_wind(self, lat: float, lon: float, timestamp: str):
        import cdsapi
        import xarray as xr

        dt      = _parse_timestamp(timestamp)
        snapped = dt.replace(minute=0, second=0, microsecond=0)

        # 1-degree bounding box around point
        north = round(lat + 0.5, 2)
        south = round(lat - 0.5, 2)
        west  = round(lon - 0.5, 2)
        east  = round(lon + 0.5, 2)

        request = {
            "product_type":    ["reanalysis"],
            "variable":        ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year":            [str(snapped.year)],
            "month":           [f"{snapped.month:02d}"],
            "day":             [f"{snapped.day:02d}"],
            "time":            [f"{snapped.hour:02d}:00"],
            "area":            [north, west, south, east],
            "data_format":     "netcdf",
            "download_format": "unarchived",
        }

        logger.info(
            "ERA5 request: (%.4f, %.4f) at %s UTC",
            lat, lon, snapped.strftime("%Y-%m-%d %H:%M"),
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(ERA5_DATASET, request, tmp_path)

            ds = xr.open_dataset(tmp_path)
            u10 = float(ds["u10"].interp(latitude=lat, longitude=lon, method="nearest").values.flat[0])
            v10 = float(ds["v10"].interp(latitude=lat, longitude=lon, method="nearest").values.flat[0])
            ds.close()
            return u10, v10
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def _compute_context(self, lat, lon, timestamp, u10, v10) -> dict:
        speed     = math.sqrt(u10**2 + v10**2)
        direction = (math.degrees(math.atan2(-u10, -v10)) + 360) % 360

        validity       = _assess_sar_validity(speed)
        lookalike_risk = _assess_lookalike_risk(speed)
        drift          = _compute_drift(speed, direction, lat)

        return {
            "wind_u":             round(u10, 3),
            "wind_v":             round(v10, 3),
            "wind_speed_ms":      round(speed, 2),
            "wind_direction_deg": round(direction, 1),

            "sar_validity":        validity["status"],
            "sar_validity_detail": validity["detail"],

            "lookalike_wind_risk": lookalike_risk["level"],
            "lookalike_wind_note": lookalike_risk["note"],

            "drift_vector": drift,

            "wind_fetched_at":  datetime.now(timezone.utc).isoformat(),
            "wind_data_source": "ERA5 reanalysis (ECMWF/Copernicus CDS)",
        }

    def _unavailable_context(self, reason: str) -> dict:
        return {
            "wind_u": None, "wind_v": None,
            "wind_speed_ms": None, "wind_direction_deg": None,
            "sar_validity": "unavailable", "sar_validity_detail": reason,
            "lookalike_wind_risk": "unknown", "lookalike_wind_note": "",
            "drift_vector": None,
            "wind_fetched_at": None, "wind_data_source": None,
        }


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O — independently testable)
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _assess_sar_validity(speed: float) -> dict:
    if speed < WIND_MIN_VISIBLE:
        return {
            "status": "too_low",
            "detail": (
                f"Wind {speed:.1f} m/s is below the {WIND_MIN_VISIBLE} m/s detection minimum. "
                "Oil film may not sufficiently suppress capillary waves. "
                "Detection confidence is reduced — treat with caution."
            ),
        }
    if speed < WIND_BORDERLINE:
        return {
            "status": "borderline",
            "detail": (
                f"Wind {speed:.1f} m/s is in the marginal {WIND_MIN_VISIBLE}-{WIND_BORDERLINE} m/s zone. "
                "Detection is possible but with reduced contrast."
            ),
        }
    if speed <= WIND_MAX_VALID:
        return {
            "status": "valid",
            "detail": (
                f"Wind {speed:.1f} m/s is within the optimal {WIND_BORDERLINE}-{WIND_MAX_VALID} m/s window. "
                "SAR oil detection conditions are good."
            ),
        }
    return {
        "status": "too_high",
        "detail": (
            f"Wind {speed:.1f} m/s exceeds {WIND_MAX_VALID} m/s. "
            "Wave breaking may disperse the slick and obscure its signature. "
            "Dark patch could be wind roughness variation rather than oil."
        ),
    }


def _assess_lookalike_risk(speed: float) -> dict:
    if speed < WIND_MIN_VISIBLE:
        return {
            "level": "high",
            "note": (
                "Calm water and biogenic films are visually indistinguishable from oil "
                "at this wind speed. High false-positive risk."
            ),
        }
    if speed < WIND_BORDERLINE:
        return {
            "level": "medium",
            "note": "Marginal wind speed. Biogenic films may mimic oil slicks.",
        }
    if speed > WIND_HIGH_RISK:
        return {
            "level": "medium",
            "note": (
                f"Wind {speed:.1f} m/s. Wind shadows and wave streak patterns "
                "can produce dark features similar to oil slicks."
            ),
        }
    return {
        "level": "low",
        "note": "Wind conditions favour reliable oil vs look-alike discrimination.",
    }


def _compute_drift(speed: float, wind_direction_deg: float, lat: float) -> dict:
    """
    Estimate oil drift using 3% empirical wind drift factor.
    Drift direction = downwind + Coriolis deflection (right in NH, left in SH).
    """
    downwind        = (wind_direction_deg + 180.0) % 360.0
    deflection      = DRIFT_DEFLECTION if lat >= 0 else -DRIFT_DEFLECTION
    drift_bearing   = (downwind + deflection) % 360.0
    drift_speed_ms  = speed * DRIFT_FACTOR
    drift_speed_kmh = drift_speed_ms * 3.6

    return {
        "bearing_deg": round(drift_bearing, 1),
        "speed_ms":    round(drift_speed_ms, 4),
        "speed_kmh":   round(drift_speed_kmh, 3),
        "6h_km":       round(drift_speed_kmh * 6, 2),
        "12h_km":      round(drift_speed_kmh * 12, 2),
        "24h_km":      round(drift_speed_kmh * 24, 2),
        "note": (
            f"Predicted drift bearing {drift_bearing:.0f}° at {drift_speed_ms:.3f} m/s "
            f"(3% wind drift, {deflection:+.0f}° Coriolis)"
        ),
    }


def drift_arrow_geojson(lat: float, lon: float, drift_vector: dict, hours: int = 24) -> dict:
    """
    Build GeoJSON LineString for the predicted drift path.
    Used by the Leaflet frontend to render an arrow from detection centroid.
    """
    distance_km = drift_vector.get(f"{hours}h_km", drift_vector.get("24h_km", 0))
    bearing_rad = math.radians(drift_vector["bearing_deg"])

    delta_lat = (distance_km / 111.32) * math.cos(bearing_rad)
    delta_lon = (distance_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(bearing_rad)

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat], [lon + delta_lon, lat + delta_lat]],
        },
        "properties": {
            "type":        "drift_arrow",
            "bearing_deg": drift_vector["bearing_deg"],
            "distance_km": distance_km,
            "hours":       hours,
            "speed_ms":    drift_vector["speed_ms"],
        },
    }