"""
Spectra Phase G — AIS Vessel Attribution
==========================================
Cross-references oil spill detections against AIS vessel tracking data
to identify ships that were in the vicinity at the time of detection.

Pipeline position (on-demand, not blocking the scan loop):
    Detection record → AISAttribution → vessel candidates → DB + PDF report

What it produces per detection:
    ais_vessels_found     int  — total vessels in search radius
    ais_candidates        list — ranked vessel dicts (closest first)
    ais_top_suspect       dict — highest-suspicion vessel (or None)
    ais_search_radius_nm  float — search radius used (nautical miles)
    ais_queried_at        str  — ISO-8601 timestamp of the query
    ais_data_source       str  — which API provided the data

Each vessel candidate dict contains:
    mmsi, imo, name, callsign, vessel_type, vessel_type_code,
    latitude, longitude, speed_knots, heading_deg, nav_status,
    distance_nm, bearing_from_spill, suspicion_score, suspicion_flags

Suspicion scoring (0-10):
    +3  tanker type (80-89) within radius
    +2  cargo/bulk type (70-79) within radius
    +2  speed < 1 knot (vessel drifting / anchored at scene)
    +2  within 3 nautical miles of detection centroid
    +1  within 5 nautical miles of detection centroid
    +1  AIS nav_status = 0 (underway using engine) with low speed — possibly loitering

API: AISHub (free, requires registration)
    Register at: https://www.aishub.net/register
    API docs:    https://www.aishub.net/api
    Endpoint:    https://data.aishub.net/ws.php
    Key fields:  USERNAME (your AISHub username)

AISHub limitation:
    AISHub provides CURRENT positions only (not historical).
    For a scan run 6 days ago, vessels may have moved.
    For production legal use, upgrade to a paid historical API:
        - Datalastic: https://datalastic.com (cheapest option with history)
        - MarineTraffic: https://servicedocs.marinetraffic.com
        - VT Explorer: https://www.vtexplorer.com
    The module is built with a clean _fetch_vessels() interface so
    replacing the data source requires changing only that one method.

Usage:
    from backend.ais_attribution import AISAttribution
    ais = AISAttribution()
    result = ais.attribute(lat=3.5, lon=6.2, timestamp="2024-11-14T09:30:00Z")
    detection.update(result)

    # Or on-demand via API endpoint (see phase_g_integration_patch.py):
    POST /detections/{id}/ais

Environment:
    export AISHUB_USERNAME=your_aishub_username

Dependencies:
    pip install requests  (already in your venv)
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AISHUB_API_URL      = "https://data.aishub.net/ws.php"
SEARCH_RADIUS_NM    = 10.0   # nautical miles — search box around detection centroid
CLOSE_RADIUS_NM     = 3.0    # high-suspicion proximity threshold
NM_TO_DEG           = 1 / 60  # 1 nautical mile ≈ 1/60 degree latitude

REQUEST_TIMEOUT_S   = 30
MAX_CANDIDATES      = 20     # cap returned vessels for DB / report

# AIS vessel type code ranges
TANKER_TYPES        = set(range(80, 90))   # 80-89: all tankers
CARGO_TYPES         = set(range(70, 80))   # 70-79: cargo and bulk
FISHING_TYPES       = {30}                 # 30: fishing
SPECIAL_TYPES       = set(range(50, 60))   # 50-59: special craft (OSVs, tugs)

# All vessel types worth flagging near an oil spill
SUSPICIOUS_TYPES    = TANKER_TYPES | CARGO_TYPES | FISHING_TYPES | SPECIAL_TYPES

# AIS nav status codes
NAV_UNDERWAY_ENGINE = 0
NAV_ANCHORED        = 1
NAV_NOT_COMMAND     = 2
NAV_MOORED          = 5

# ---------------------------------------------------------------------------
# AIS vessel type label lookup
# ---------------------------------------------------------------------------

def _vessel_type_label(type_code: int) -> str:
    if type_code in TANKER_TYPES:
        return "Tanker"
    if type_code in CARGO_TYPES:
        return "Cargo"
    if type_code in FISHING_TYPES:
        return "Fishing"
    if type_code in SPECIAL_TYPES:
        return "Special/OSV"
    if 60 <= type_code <= 69:
        return "Passenger"
    if 40 <= type_code <= 49:
        return "High-Speed Craft"
    if type_code == 52:
        return "Tug"
    if type_code == 0:
        return "Unknown"
    return f"Type {type_code}"


def _nav_status_label(code: int) -> str:
    labels = {
        0: "Underway (engine)",
        1: "Anchored",
        2: "Not under command",
        3: "Restricted manoeuvrability",
        4: "Constrained by draught",
        5: "Moored",
        6: "Aground",
        7: "Fishing",
        8: "Under sail",
        15: "Undefined",
    }
    return labels.get(code, f"Status {code}")


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------

def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R = 3440.065  # Earth radius in nautical miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 (degrees, 0=N)."""
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(math.radians(lat2))
    y = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Suspicion scoring
# ---------------------------------------------------------------------------

def _score_vessel(vessel: dict, spill_lat: float, spill_lon: float) -> tuple:
    """
    Compute suspicion score (0-10) and list of flag strings.
    Returns (score, flags).
    """
    score = 0
    flags = []

    type_code  = vessel.get("vessel_type_code", 0)
    dist_nm    = vessel.get("distance_nm", 999)
    speed_kts  = vessel.get("speed_knots", 99)
    nav_stat   = vessel.get("nav_status", 15)

    # Vessel type — tankers are highest suspicion near oil spills
    if type_code in TANKER_TYPES:
        score += 3
        flags.append("TANKER")
    elif type_code in CARGO_TYPES:
        score += 2
        flags.append("CARGO/BULK")
    elif type_code in FISHING_TYPES:
        score += 1
        flags.append("FISHING")
    elif type_code in SPECIAL_TYPES:
        score += 1
        flags.append("OSV/SPECIAL")

    # Proximity
    if dist_nm <= CLOSE_RADIUS_NM:
        score += 2
        flags.append(f"WITHIN {CLOSE_RADIUS_NM}nm")
    elif dist_nm <= 5.0:
        score += 1
        flags.append("WITHIN 5nm")

    # Speed anomaly — very slow or stopped near a spill
    if speed_kts < 0.5:
        score += 2
        flags.append("STATIONARY")
    elif speed_kts < 1.5:
        score += 1
        flags.append("NEAR-STATIONARY")

    # Nav status anomaly — underway engine at very low speed = possible loitering
    if nav_stat == NAV_UNDERWAY_ENGINE and speed_kts < 1.5:
        score += 1
        flags.append("LOITERING")

    return min(score, 10), flags


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AISAttribution:
    """
    AIS vessel attribution for Spectra detections.
    Instantiate once at module level.
    """

    def __init__(self):
        self._username = os.environ.get("AISHUB_USERNAME", "").strip()
        if not self._username:
            logger.warning(
                "AISHUB_USERNAME not set. AIS attribution unavailable. "
                "Register at https://www.aishub.net/register and set: "
                "export AISHUB_USERNAME=your_username"
            )

    def is_available(self) -> bool:
        return bool(self._username)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attribute(
        self,
        lat: float,
        lon: float,
        timestamp: str,
        radius_nm: float = SEARCH_RADIUS_NM,
    ) -> dict:
        """
        Query AIS data for vessels near the detection and score them.

        Args:
            lat, lon:    Detection centroid (decimal degrees)
            timestamp:   ISO-8601 UTC string of Sentinel-1 acquisition
            radius_nm:   Search radius in nautical miles (default 10)

        Returns:
            Dict with all ais_* fields. All None if unavailable.
        """
        if not self.is_available():
            return self._unavailable("AISHUB_USERNAME not configured")

        try:
            raw_vessels = self._fetch_vessels(lat, lon, radius_nm)
        except Exception as exc:
            logger.error("AIS fetch failed: %s", exc, exc_info=True)
            return self._unavailable(f"API error: {exc}")

        if raw_vessels is None:
            return self._unavailable("No AIS data returned from API")

        candidates = self._process_vessels(raw_vessels, lat, lon, radius_nm)

        top = candidates[0] if candidates else None

        logger.info(
            "AIS attribution: %d vessels in %.1fnm of (%.4f, %.4f). "
            "Top suspect: %s (score=%s)",
            len(candidates), radius_nm, lat, lon,
            top.get("name", "—") if top else "none",
            top.get("suspicion_score", 0) if top else 0,
        )

        return {
            "ais_vessels_found":    len(candidates),
            "ais_candidates":       candidates[:MAX_CANDIDATES],
            "ais_top_suspect":      top,
            "ais_search_radius_nm": radius_nm,
            "ais_queried_at":       datetime.now(timezone.utc).isoformat(),
            "ais_data_source":      "AISHub (real-time)",
            "ais_note":             (
                "AISHub provides current vessel positions only. "
                "Positions shown are from the time of query, not the time of detection. "
                "For historical attribution, upgrade to Datalastic or MarineTraffic paid API."
            ),
        }

    # ------------------------------------------------------------------
    # AISHub fetch
    # ------------------------------------------------------------------

    def _fetch_vessels(self, lat: float, lon: float, radius_nm: float) -> list:
        """
        Query AISHub bounding box API.
        Returns list of raw vessel dicts (AISHub JSON format).
        """
        # Convert radius to degrees (approximate — fine at equatorial latitudes)
        deg_buf = radius_nm * NM_TO_DEG * 1.2  # 20% padding

        params = {
            "username": self._username,
            "format":   "1",      # AIS format (includes vessel name / type)
            "output":   "json",
            "compress": "0",      # no compression
            "latmin":   round(lat - deg_buf, 4),
            "latmax":   round(lat + deg_buf, 4),
            "lonmin":   round(lon - deg_buf, 4),
            "lonmax":   round(lon + deg_buf, 4),
        }

        logger.info(
            "AISHub query: bbox (%.4f-%.4f, %.4f-%.4f)",
            params["latmin"], params["latmax"],
            params["lonmin"], params["lonmax"],
        )

        resp = requests.get(AISHUB_API_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()

        data = resp.json()

        # AISHub wraps response: [{ERROR: false, ...}, [vessel, vessel, ...]]
        if isinstance(data, list) and len(data) >= 2:
            meta    = data[0]
            vessels = data[1] if isinstance(data[1], list) else []
            if meta.get("ERROR"):
                raise RuntimeError(f"AISHub error: {meta.get('ERROR_MESSAGE', 'unknown')}")
            logger.info("AISHub returned %d vessels", len(vessels))
            return vessels

        # Fallback: sometimes returns flat list
        if isinstance(data, list) and data and isinstance(data[0], dict) and "MMSI" in data[0]:
            return data

        logger.warning("AISHub returned unexpected format: %s", type(data))
        return []

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_vessels(
        self,
        raw_vessels: list,
        spill_lat: float,
        spill_lon: float,
        radius_nm: float,
    ) -> list:
        """
        Parse, filter by actual radius, score, and sort vessel list.
        Returns list of candidate dicts sorted by suspicion_score desc, distance asc.
        """
        candidates = []

        for raw in raw_vessels:
            try:
                parsed = self._parse_vessel(raw, spill_lat, spill_lon, radius_nm)
                if parsed:
                    candidates.append(parsed)
            except Exception as exc:
                logger.debug("Vessel parse error: %s — %s", exc, raw)

        # Sort: highest suspicion first, then closest
        candidates.sort(key=lambda v: (-v["suspicion_score"], v["distance_nm"]))
        return candidates

    def _parse_vessel(
        self,
        raw: dict,
        spill_lat: float,
        spill_lon: float,
        radius_nm: float,
    ) -> Optional[dict]:
        """Parse one raw AISHub vessel dict. Returns None if outside radius."""
        # AISHub field names differ between format=1 and format=2
        # format=1 returns NAME, TYPE, LATITUDE, LONGITUDE, SOG, COG, HEADING etc.
        v_lat = raw.get("LATITUDE") or raw.get("latitude")
        v_lon = raw.get("LONGITUDE") or raw.get("longitude")

        if v_lat is None or v_lon is None:
            return None

        v_lat = float(v_lat)
        v_lon = float(v_lon)

        dist_nm = _haversine_nm(spill_lat, spill_lon, v_lat, v_lon)
        if dist_nm > radius_nm:
            return None  # outside actual circular radius (bbox query is square)

        bearing = _bearing_deg(spill_lat, spill_lon, v_lat, v_lon)

        mmsi        = raw.get("MMSI") or raw.get("mmsi") or 0
        imo         = raw.get("IMO")  or raw.get("imo")  or 0
        name        = (raw.get("NAME") or raw.get("name") or "").strip() or "UNKNOWN"
        callsign    = (raw.get("CALLSIGN") or raw.get("callsign") or "").strip()
        type_code   = int(raw.get("TYPE") or raw.get("type") or 0)
        speed_raw   = raw.get("SOG") or raw.get("sog") or 0
        heading_raw = raw.get("HEADING") or raw.get("heading") or 0
        nav_stat    = int(raw.get("NAVSTAT") or raw.get("navstat") or 15)
        destination = (raw.get("DEST") or raw.get("dest") or "").strip()

        # SOG in AISHub is in tenths of a knot (or already in knots depending on format)
        # format=1 returns raw AIS SOG (tenths of knot, 0-1022)
        speed_kts = float(speed_raw) / 10.0 if float(speed_raw) > 102 else float(speed_raw)

        vessel = {
            "mmsi":              int(mmsi),
            "imo":               int(imo) if imo else None,
            "name":              name,
            "callsign":          callsign or None,
            "vessel_type_code":  type_code,
            "vessel_type":       _vessel_type_label(type_code),
            "latitude":          round(v_lat, 5),
            "longitude":         round(v_lon, 5),
            "speed_knots":       round(speed_kts, 1),
            "heading_deg":       int(heading_raw) if int(heading_raw) < 360 else None,
            "nav_status":        nav_stat,
            "nav_status_label":  _nav_status_label(nav_stat),
            "destination":       destination or None,
            "distance_nm":       round(dist_nm, 2),
            "bearing_from_spill": round(bearing, 1),
            "suspicion_score":   0,
            "suspicion_flags":   [],
        }

        score, flags = _score_vessel(vessel, spill_lat, spill_lon)
        vessel["suspicion_score"] = score
        vessel["suspicion_flags"] = flags

        return vessel

    # ------------------------------------------------------------------
    # Fallback result
    # ------------------------------------------------------------------

    def _unavailable(self, reason: str) -> dict:
        return {
            "ais_vessels_found":    None,
            "ais_candidates":       [],
            "ais_top_suspect":      None,
            "ais_search_radius_nm": SEARCH_RADIUS_NM,
            "ais_queried_at":       None,
            "ais_data_source":      None,
            "ais_note":             reason,
        }


# ---------------------------------------------------------------------------
# Convenience: format candidate for PDF report section
# ---------------------------------------------------------------------------

def format_candidates_for_report(candidates: list, max_rows: int = 10) -> list:
    """
    Return a cleaned list of vessel dicts suitable for table rendering in the PDF.
    Trims to max_rows and formats numeric fields.
    """
    out = []
    for v in candidates[:max_rows]:
        out.append({
            "Name":          v.get("name", "—"),
            "MMSI":          str(v.get("mmsi", "—")),
            "IMO":           str(v.get("imo")) if v.get("imo") else "—",
            "Type":          v.get("vessel_type", "—"),
            "Distance":      f"{v.get('distance_nm', '?')} nm",
            "Speed":         f"{v.get('speed_knots', '?')} kts",
            "Status":        v.get("nav_status_label", "—"),
            "Destination":   v.get("destination") or "—",
            "Suspicion":     f"{v.get('suspicion_score', 0)}/10",
            "Flags":         ", ".join(v.get("suspicion_flags", [])) or "—",
        })
    return out