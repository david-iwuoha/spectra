"""
Spectra Phase G — AIS Vessel Attribution (AISStream.io)
=========================================================
Cross-references oil spill detections against live AIS vessel tracking data
to identify ships that were in the vicinity of the detection.

Data source: AISStream.io (https://aisstream.io)
    - Free tier, no station required
    - WebSocket-based, bounding box subscription
    - Register at https://aisstream.io and get API key from
      https://aisstream.io/apikeys

Environment:
    AISSTREAM_API_KEY=your_key_here   (in your .env file)

Dependencies:
    pip install websockets python-dotenv --break-system-packages

Pipeline position (on-demand, not blocking the scan loop):
    Detection record → AISAttribution → vessel candidates → DB + PDF report

What it produces per detection:
    ais_vessels_found     int  — total vessels found in search radius
    ais_candidates        list — ranked vessel dicts (highest suspicion first)
    ais_top_suspect       dict — highest-suspicion vessel (or None)
    ais_search_radius_nm  float — search radius used (nautical miles)
    ais_queried_at        str  — ISO-8601 timestamp of the query
    ais_data_source       str  — "AISStream.io (real-time)"

Each vessel candidate dict contains:
    mmsi, name, vessel_type, vessel_type_code,
    latitude, longitude, speed_knots, heading_deg,
    nav_status, nav_status_label, course_over_ground,
    distance_nm, bearing_from_spill,
    suspicion_score (0-10), suspicion_flags

Suspicion scoring:
    +3  tanker type (80-89)
    +2  cargo/bulk type (70-79)
    +1  fishing (30) or special/OSV (50-59)
    +2  within 3 nautical miles of spill centroid
    +1  within 5 nautical miles
    +2  speed < 0.5 knots (stationary at scene)
    +1  speed < 1.5 knots (near-stationary / loitering)
    +1  underway (nav_status=0) at very low speed

Important:
    AISStream provides CURRENT vessel positions, not historical.
    For a Sentinel-1 scene acquired 6 days ago, vessels will have moved.
    Run attribution at scan time AND re-run on-demand via the API endpoint.
    For historical legal-grade attribution, upgrade to Datalastic paid API —
    the _fetch_vessels() method is the only thing that needs changing.

Usage:
    from backend.ais_attribution import AISAttribution
    ais = AISAttribution()
    result = ais.attribute(lat=4.915, lon=6.275, timestamp="2024-11-14T09:30:00Z")
    detection.update(result)
"""

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AISSTREAM_WS_URL   = "wss://stream.aisstream.io/v0/stream"
SEARCH_RADIUS_NM   = 10.0    # nautical miles — default search radius
CLOSE_RADIUS_NM    = 3.0     # high-suspicion proximity threshold
NM_TO_DEG          = 1 / 60  # 1 nm ≈ 1/60 degree latitude (approximate)
COLLECT_TIMEOUT_S  = 15      # how long to listen for AIS messages (seconds)
MAX_CANDIDATES     = 20      # cap returned vessels

# AIS vessel type code ranges
TANKER_TYPES   = set(range(80, 90))
CARGO_TYPES    = set(range(70, 80))
FISHING_TYPES  = {30}
SPECIAL_TYPES  = set(range(50, 60))

# Nav status codes
NAV_UNDERWAY_ENGINE = 0
NAV_ANCHORED        = 1
NAV_MOORED          = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """
    Load AISSTREAM_API_KEY from environment.
    Supports both raw env var and .env file (via python-dotenv if installed).
    """
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    if key:
        return key

    # Try loading from .env file in repo root
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    except ImportError:
        pass

    return key


def _vessel_type_label(code: int) -> str:
    if code in TANKER_TYPES:   return "Tanker"
    if code in CARGO_TYPES:    return "Cargo"
    if code in FISHING_TYPES:  return "Fishing"
    if code in SPECIAL_TYPES:  return "Special/OSV"
    if 60 <= code <= 69:       return "Passenger"
    if code == 0:              return "Unknown"
    return f"Type {code}"


def _nav_status_label(code: int) -> str:
    return {
        0:  "Underway (engine)",
        1:  "Anchored",
        2:  "Not under command",
        3:  "Restricted manoeuvrability",
        4:  "Constrained by draught",
        5:  "Moored",
        6:  "Aground",
        7:  "Fishing",
        8:  "Under sail",
        15: "Undefined",
    }.get(code, f"Status {code}")


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 (degrees, 0=North)."""
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(math.radians(lat2))
    y = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _score_vessel(vessel: dict) -> tuple:
    """
    Compute suspicion score (0-10) and list of flag strings.
    Returns (score, flags).
    """
    score = 0
    flags = []

    type_code = vessel.get("vessel_type_code", 0)
    dist_nm   = vessel.get("distance_nm", 999)
    speed_kts = vessel.get("speed_knots", 99)
    nav_stat  = vessel.get("nav_status", 15)

    # Vessel type
    if type_code in TANKER_TYPES:
        score += 3; flags.append("TANKER")
    elif type_code in CARGO_TYPES:
        score += 2; flags.append("CARGO/BULK")
    elif type_code in FISHING_TYPES:
        score += 1; flags.append("FISHING")
    elif type_code in SPECIAL_TYPES:
        score += 1; flags.append("OSV/SPECIAL")

    # Proximity
    if dist_nm <= CLOSE_RADIUS_NM:
        score += 2; flags.append(f"WITHIN {CLOSE_RADIUS_NM}nm")
    elif dist_nm <= 5.0:
        score += 1; flags.append("WITHIN 5nm")

    # Speed anomaly
    if speed_kts < 0.5:
        score += 2; flags.append("STATIONARY")
    elif speed_kts < 1.5:
        score += 1; flags.append("NEAR-STATIONARY")

    # Loitering — engine on but barely moving
    if nav_stat == NAV_UNDERWAY_ENGINE and speed_kts < 1.5:
        score += 1; flags.append("LOITERING")

    return min(score, 10), flags


# ---------------------------------------------------------------------------
# AISStream fetch (async, run via asyncio.run())
# ---------------------------------------------------------------------------

async def _fetch_from_aisstream(
    api_key: str,
    lat: float,
    lon: float,
    radius_nm: float,
    timeout_s: float,
) -> list:
    """
    Open AISStream WebSocket, subscribe to bounding box, collect
    vessel position messages for timeout_s seconds, return raw list.
    """
    try:
        import websockets
    except ImportError:
        raise RuntimeError(
            "websockets not installed. "
            "Run: pip install websockets --break-system-packages"
        )

    # Build bounding box with 20% padding
    deg_buf = radius_nm * NM_TO_DEG * 1.2
    bbox = [
        [lat - deg_buf, lon - deg_buf],  # [min_lat, min_lon]
        [lat + deg_buf, lon + deg_buf],  # [max_lat, max_lon]
    ]

    subscribe_msg = json.dumps({
        "APIKey":        api_key,
        "BoundingBoxes": [bbox],
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    })

    vessels_seen = {}  # keyed by MMSI to deduplicate
    deadline = asyncio.get_event_loop().time() + timeout_s

    logger.info(
        "AISStream: connecting, bbox lat[%.3f-%.3f] lon[%.3f-%.3f], "
        "listening %.0fs",
        lat - deg_buf, lat + deg_buf,
        lon - deg_buf, lon + deg_buf,
        timeout_s,
    )

    try:
        async with websockets.connect(
            AISSTREAM_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=15,
        ) as ws:
            await ws.send(subscribe_msg)

            while asyncio.get_event_loop().time() < deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                except asyncio.TimeoutError:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("MessageType", "")

                if msg_type == "PositionReport":
                    meta    = msg.get("MetaData", {})
                    payload = msg.get("Message", {}).get("PositionReport", {})
                    mmsi    = meta.get("MMSI") or payload.get("UserID")
                    if not mmsi:
                        continue

                    entry = vessels_seen.get(mmsi, {})
                    entry.update({
                        "mmsi":       int(mmsi),
                        "name":       (meta.get("ShipName") or entry.get("name") or "").strip() or "UNKNOWN",
                        "latitude":   meta.get("latitude")  or payload.get("Latitude"),
                        "longitude":  meta.get("longitude") or payload.get("Longitude"),
                        "speed_knots": (payload.get("Sog") or 0),
                        "heading_deg": payload.get("TrueHeading"),
                        "course_over_ground": payload.get("Cog"),
                        "nav_status": payload.get("NavigationalStatus", 15),
                    })
                    vessels_seen[mmsi] = entry

                elif msg_type == "ShipStaticData":
                    meta    = msg.get("MetaData", {})
                    payload = msg.get("Message", {}).get("ShipStaticData", {})
                    mmsi    = meta.get("MMSI") or payload.get("UserID")
                    if not mmsi:
                        continue

                    entry = vessels_seen.get(mmsi, {})
                    entry.update({
                        "mmsi":             int(mmsi),
                        "name":             (payload.get("Name") or entry.get("name") or "").strip() or "UNKNOWN",
                        "vessel_type_code": payload.get("Type", 0),
                        "callsign":         payload.get("CallSign", "").strip() or None,
                        "imo":              payload.get("ImoNumber") or None,
                        "destination":      payload.get("Destination", "").strip() or None,
                    })
                    vessels_seen[mmsi] = entry

    except Exception as exc:
        logger.warning("AISStream WebSocket error: %s", exc)

    logger.info("AISStream: collected %d unique vessels", len(vessels_seen))
    return list(vessels_seen.values())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AISAttribution:
    """
    AIS vessel attribution for Spectra detections via AISStream.io.
    Instantiate once at module level in detect.py / main.py.
    """

    def __init__(self):
        self._api_key = _load_api_key()
        if not self._api_key:
            logger.warning(
                "AISSTREAM_API_KEY not set. AIS attribution unavailable. "
                "Add AISSTREAM_API_KEY=your_key to your .env file. "
                "Get a key at https://aisstream.io/apikeys"
            )

    def is_available(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Public interface — synchronous wrapper around async fetch
    # ------------------------------------------------------------------

    def attribute(
        self,
        lat: float,
        lon: float,
        timestamp: str,
        radius_nm: float = SEARCH_RADIUS_NM,
    ) -> dict:
        """
        Query AISStream for vessels near the detection and score them.

        Args:
            lat, lon:    Detection centroid (decimal degrees)
            timestamp:   ISO-8601 UTC string of Sentinel-1 acquisition
            radius_nm:   Search radius in nautical miles (default 10)

        Returns:
            Dict with all ais_* fields. Values are None if unavailable.
        """
        if not self.is_available():
            return self._unavailable("AISSTREAM_API_KEY not configured")

        try:
            raw_vessels = asyncio.run(
                _fetch_from_aisstream(
                    api_key=self._api_key,
                    lat=lat,
                    lon=lon,
                    radius_nm=radius_nm,
                    timeout_s=COLLECT_TIMEOUT_S,
                )
            )
        except Exception as exc:
            logger.error("AIS fetch failed: %s", exc, exc_info=True)
            return self._unavailable(f"AISStream error: {exc}")

        candidates = self._process_vessels(raw_vessels, lat, lon, radius_nm)
        top = candidates[0] if candidates else None

        logger.info(
            "AIS attribution complete: %d vessels in %.1fnm of (%.4f, %.4f). "
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
            "ais_data_source":      "AISStream.io (real-time)",
            "ais_note": (
                "AISStream provides current vessel positions only. "
                "Positions shown are from the time of query, not the time of detection. "
                "For historical attribution, upgrade to Datalastic paid API."
            ),
        }

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
        """Parse, filter to actual circle, score, and sort."""
        candidates = []

        for raw in raw_vessels:
            try:
                parsed = self._parse_vessel(raw, spill_lat, spill_lon, radius_nm)
                if parsed:
                    candidates.append(parsed)
            except Exception as exc:
                logger.debug("Vessel parse error: %s", exc)

        # Sort: highest suspicion first, then by distance
        candidates.sort(key=lambda v: (-v["suspicion_score"], v["distance_nm"]))
        return candidates

    def _parse_vessel(
        self,
        raw: dict,
        spill_lat: float,
        spill_lon: float,
        radius_nm: float,
    ) -> Optional[dict]:
        """Parse one raw vessel dict. Returns None if outside radius or missing position."""
        v_lat = raw.get("latitude")
        v_lon = raw.get("longitude")
        if v_lat is None or v_lon is None:
            return None  # position report not yet received for this MMSI

        v_lat = float(v_lat)
        v_lon = float(v_lon)

        dist_nm = _haversine_nm(spill_lat, spill_lon, v_lat, v_lon)
        if dist_nm > radius_nm:
            return None  # outside actual circular radius

        bearing     = _bearing_deg(spill_lat, spill_lon, v_lat, v_lon)
        type_code   = int(raw.get("vessel_type_code") or 0)
        speed_raw   = raw.get("speed_knots") or 0
        heading_raw = raw.get("heading_deg")
        nav_stat    = int(raw.get("nav_status") or 15)

        # AISStream SOG is already in knots (0.1 knot resolution)
        speed_kts = float(speed_raw)

        vessel = {
            "mmsi":                int(raw.get("mmsi", 0)),
            "imo":                 raw.get("imo") or None,
            "name":                raw.get("name", "UNKNOWN"),
            "callsign":            raw.get("callsign") or None,
            "vessel_type_code":    type_code,
            "vessel_type":         _vessel_type_label(type_code),
            "latitude":            round(v_lat, 5),
            "longitude":           round(v_lon, 5),
            "speed_knots":         round(speed_kts, 1),
            "heading_deg":         int(heading_raw) if heading_raw and int(heading_raw) < 360 else None,
            "course_over_ground":  raw.get("course_over_ground"),
            "nav_status":          nav_stat,
            "nav_status_label":    _nav_status_label(nav_stat),
            "destination":         raw.get("destination") or None,
            "distance_nm":         round(dist_nm, 2),
            "bearing_from_spill":  round(bearing, 1),
            "suspicion_score":     0,
            "suspicion_flags":     [],
        }

        score, flags = _score_vessel(vessel)
        vessel["suspicion_score"] = score
        vessel["suspicion_flags"] = flags

        return vessel

    # ------------------------------------------------------------------
    # Fallback
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
# Report helper (used by report_generator.py Phase G section)
# ---------------------------------------------------------------------------

def format_candidates_for_report(candidates: list, max_rows: int = 10) -> list:
    """Return cleaned vessel dicts for table rendering in the PDF report."""
    out = []
    for v in candidates[:max_rows]:
        out.append({
            "Name":        v.get("name", "—"),
            "MMSI":        str(v.get("mmsi", "—")),
            "IMO":         str(v.get("imo")) if v.get("imo") else "—",
            "Type":        v.get("vessel_type", "—"),
            "Distance":    f"{v.get('distance_nm', '?')} nm",
            "Speed":       f"{v.get('speed_knots', '?')} kts",
            "Status":      v.get("nav_status_label", "—"),
            "Destination": v.get("destination") or "—",
            "Suspicion":   f"{v.get('suspicion_score', 0)}/10",
            "Flags":       ", ".join(v.get("suspicion_flags", [])) or "—",
        })
    return out