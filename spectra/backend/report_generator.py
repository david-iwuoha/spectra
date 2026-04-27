"""
Spectra Phase F — Professional Report Generator
=================================================
Generates a PDF evidence report for a single detection.

Pipeline position (on-demand, not in the scan loop):
    Detection record (DB) → ReportGenerator → PDF bytes → HTTP response

The report is a self-contained legal/regulatory document containing:
  - Cover block: detection ID, timestamp, coordinates, scan metadata
  - Evidence chain summary: all phases that ran and their verdicts
  - Phase C section: look-alike classifier verdict + confidence bar
  - Phase D section: ERA5 wind speed, direction, SAR validity, drift table
  - Phase E section: S2 optical verdict, spectral indices (OSI/SWIRI/NDWI),
                     true-colour and false-colour thumbnails
  - Detection polygon map link (cannot embed live Leaflet — links to centroid)
  - Methodology notes: what each phase does and why, for regulators
  - Footer: generation timestamp, Spectra version, data sources

Usage from main.py:
    from backend.report_generator import ReportGenerator
    rg = ReportGenerator()
    pdf_bytes = rg.generate(detection)   # detection is a dict or ORM object
    return Response(pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="spectra_{detection_id}.pdf"'})

Dependencies:
    System (run once in Codespace terminal):
        sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0

    Python:
        pip install weasyprint --break-system-packages
"""

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Spectra version string embedded in every report footer
SPECTRA_VERSION = "1.0.0-phase-f"


class ReportGenerator:
    """
    Generates a PDF evidence report for a single Spectra detection.
    Instantiate once at module level — WeasyPrint imports are cached.
    """

    def __init__(self):
        self._weasyprint_ok = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            from weasyprint import HTML  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "weasyprint not installed. "
                "Run: pip install weasyprint --break-system-packages"
            )
            return False
        except OSError as e:
            logger.warning(
                "WeasyPrint system dependencies missing: %s. "
                "Run: sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0",
                e,
            )
            return False

    def is_available(self) -> bool:
        return self._weasyprint_ok

    def generate(self, detection: Union[dict, object]) -> bytes:
        """
        Generate a PDF report for the given detection.

        Args:
            detection: dict or SQLAlchemy model instance with all detection fields.

        Returns:
            PDF as bytes. Raise RuntimeError if WeasyPrint unavailable.
        """
        if not self._weasyprint_ok:
            raise RuntimeError(
                "WeasyPrint not available. "
                "Install system deps and pip install weasyprint."
            )

        d = _to_dict(detection)
        html = _render_html(d)

        from weasyprint import HTML as WPHtml, CSS
        pdf_bytes = WPHtml(string=html, base_url=None).write_pdf(
            stylesheets=[CSS(string=_PDF_PRINT_CSS)]
        )
        logger.info(
            "Generated PDF report for detection %s (%d bytes)",
            d.get("id", "?"),
            len(pdf_bytes),
        )
        return pdf_bytes


# ---------------------------------------------------------------------------
# HTML template renderer
# ---------------------------------------------------------------------------

def _to_dict(detection) -> dict:
    """Convert ORM object to plain dict. Already a dict → return as-is."""
    if isinstance(detection, dict):
        return detection
    return {c.name: getattr(detection, c.name) for c in detection.__table__.columns}


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts


def _confidence_bar(score: Optional[float], color: str = "#00d4aa") -> str:
    if score is None:
        return ""
    pct = min(100, max(0, round(score * 100)))
    return f"""
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
      <span class="conf-bar-label">{pct}%</span>
    </div>"""


def _verdict_chip(text: str, color: str) -> str:
    return f'<span class="chip" style="background:{color}22;color:{color};border:1px solid {color}66;">{text}</span>'


def _thumbnail_img(b64_uri: Optional[str], label: str) -> str:
    if not b64_uri:
        return f'<div class="thumb-placeholder"><span>Image unavailable</span><small>{label}</small></div>'
    return f'<div class="thumb-wrap"><img src="{b64_uri}" alt="{label}"/><div class="thumb-label">{label}</div></div>'


def _render_html(d: dict) -> str:
    detection_id   = d.get("id", "UNKNOWN")
    detected_at    = _fmt_ts(d.get("detected_at") or d.get("timestamp"))
    confidence     = d.get("confidence")
    area_km2       = d.get("area_km2")
    spill_pixels   = d.get("spill_pixels")
    centroid_lat   = d.get("centroid_lat")
    centroid_lon   = d.get("centroid_lon")
    generated_at   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Phase C
    lookalike_score  = d.get("lookalike_score")
    lookalike_label  = d.get("lookalike_label") or "—"
    lookalike_passed = d.get("lookalike_passed")

    # Phase D
    wind_speed      = d.get("wind_speed_ms")
    wind_dir        = d.get("wind_direction_deg")
    sar_validity    = d.get("sar_validity") or "unavailable"
    sar_detail      = d.get("sar_validity_detail") or ""
    wind_risk       = d.get("lookalike_wind_risk") or "unknown"
    wind_risk_note  = d.get("lookalike_wind_note") or ""
    drift_bearing   = d.get("drift_bearing_deg") or (d.get("drift_vector") or {}).get("bearing_deg")
    drift_24h       = d.get("drift_24h_km")       or (d.get("drift_vector") or {}).get("24h_km")
    drift_6h        = (d.get("drift_vector") or {}).get("6h_km")
    drift_speed     = d.get("drift_speed_ms")     or (d.get("drift_vector") or {}).get("speed_ms")

    # Phase E
    optical_verdict    = d.get("optical_verdict") or "unavailable"
    optical_confidence = d.get("optical_confidence")
    optical_cloud      = d.get("optical_cloud_fraction")
    optical_osi        = d.get("optical_osi")
    optical_swiri      = d.get("optical_swiri")
    optical_ndwi       = d.get("optical_ndwi")
    optical_reason     = d.get("optical_reason") or ""
    optical_scene      = d.get("optical_scene_name") or "—"
    thumb_rgb          = d.get("optical_thumbnail_rgb")
    thumb_fc           = d.get("optical_thumbnail_falsecolour")

    # Verdict colours
    c_verdict = {
        "confirmed":    "#00d4aa",
        "unconfirmed":  "#ff4444",
        "inconclusive": "#f5a623",
        "unavailable":  "#6b8299",
    }
    sar_valid_color = {
        "valid": "#00d4aa", "borderline": "#f5a623",
        "too_low": "#ff4444", "too_high": "#ff4444",
        "unavailable": "#6b8299",
    }
    risk_color = {"low": "#00d4aa", "medium": "#f5a623", "high": "#ff4444", "unknown": "#6b8299"}
    la_color   = "#00d4aa" if lookalike_passed else "#ff4444"

    # Evidence summary line (for cover block)
    phases_run = []
    if lookalike_score is not None:
        phases_run.append(("Phase C Look-alike", "PASS" if lookalike_passed else "SUPPRESSED", la_color))
    if wind_speed is not None:
        phases_run.append(("Phase D Wind", sar_validity.upper().replace("_", " "), sar_valid_color.get(sar_validity, "#6b8299")))
    if optical_verdict != "unavailable":
        phases_run.append(("Phase E Optical S2", optical_verdict.upper(), c_verdict.get(optical_verdict, "#6b8299")))

    phases_html = "".join(
        f'<div class="ev-pill" style="border-color:{col}44;">'
        f'<span class="ev-phase">{ph}</span>'
        f'<span class="ev-verdict" style="color:{col};">{vd}</span>'
        f'</div>'
        for ph, vd, col in phases_run
    ) or '<p style="color:#6b8299;font-size:11px;">No post-processing phases run</p>'

    # Map link if centroid available
    map_link = ""
    if centroid_lat and centroid_lon:
        zoom = 11
        osm  = f"https://www.openstreetmap.org/?mlat={centroid_lat}&mlon={centroid_lon}#map={zoom}/{centroid_lat}/{centroid_lon}"
        map_link = f'<p class="map-link">Centroid location: <strong>{centroid_lat:.5f}°N, {centroid_lon:.5f}°E</strong> — <a href="{osm}">[View on OpenStreetMap]</a></p>'

    # OSI/SWIRI/NDWI colours
    def osi_color(v):
        if v is None: return "#6b8299"
        return "#00d4aa" if v >= 2.5 else "#f5a623" if v >= 1.8 else "#6b8299"

    def swiri_color(v):
        if v is None: return "#6b8299"
        return "#00d4aa" if v <= 0.20 else "#f5a623" if v <= 0.30 else "#6b8299"

    def ndwi_color(v):
        if v is None: return "#6b8299"
        return "#00d4aa" if v >= 0 else "#6b8299"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spectra Detection Report — {detection_id}</title>
<style>
  {_BASE_CSS}
</style>
</head>
<body>

<!-- ========================================================
     COVER BLOCK
     ======================================================== -->
<div class="cover">
  <div class="cover-header">
    <div class="cover-logo">SPECTRA</div>
    <div class="cover-sub">Oil Spill Detection Platform &mdash; Evidence Report</div>
  </div>
  <div class="cover-id">DETECTION {detection_id}</div>
  <div class="cover-meta-grid">
    <div class="cover-meta-item"><div class="cmi-label">Detected At</div><div class="cmi-value">{detected_at}</div></div>
    <div class="cover-meta-item"><div class="cmi-label">Confidence</div><div class="cmi-value">{f"{confidence}%" if confidence is not None else "—"}</div></div>
    <div class="cover-meta-item"><div class="cmi-label">Area</div><div class="cmi-value">{f"{area_km2} km²" if area_km2 is not None else "—"}</div></div>
    <div class="cover-meta-item"><div class="cmi-label">Spill Pixels</div><div class="cmi-value">{f"{int(spill_pixels):,}" if spill_pixels else "—"}</div></div>
    <div class="cover-meta-item"><div class="cmi-label">Centroid Lat</div><div class="cmi-value">{f"{centroid_lat:.5f}°N" if centroid_lat else "—"}</div></div>
    <div class="cover-meta-item"><div class="cmi-label">Centroid Lon</div><div class="cmi-value">{f"{centroid_lon:.5f}°E" if centroid_lon else "—"}</div></div>
  </div>
  {map_link}
</div>

<!-- Evidence chain summary -->
<div class="section">
  <div class="section-title">Evidence Chain</div>
  <div class="ev-chain">{phases_html}</div>
</div>

<hr class="divider"/>

<!-- ========================================================
     PHASE C — LOOK-ALIKE CLASSIFIER
     ======================================================== -->
<div class="section">
  <div class="phase-header">
    <div class="phase-badge" style="background:#ff444422;color:#ff6b6b;border-color:#ff444466;">C</div>
    <div class="phase-title">Look-alike Classifier</div>
    <div class="phase-subtitle">MobileNet CNN — oil vs look-alike discrimination</div>
  </div>

  {"<p class='unavail'>Look-alike classifier did not run for this detection.</p>" if lookalike_score is None else f"""
  <div class="two-col">
    <div>
      <table class="data-table">
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Classification</td><td class="val" style="color:{la_color};">{lookalike_label.upper()}</td></tr>
        <tr><td>Oil probability</td><td class="val">{f"{round(lookalike_score * 100, 1)}%" if lookalike_score is not None else "—"}</td></tr>
        <tr><td>Alert gate</td><td class="val" style="color:{la_color};">{"PASSED" if lookalike_passed else "SUPPRESSED"}</td></tr>
        <tr><td>Threshold</td><td class="val">0.60 (60% minimum)</td></tr>
      </table>
    </div>
    <div>
      <p class="field-label">Oil Probability Score</p>
      {_confidence_bar(lookalike_score, la_color)}
      <p class="field-note">Score &ge; 0.60 passes the gate. Score &lt; 0.60 suppresses the alert and classifies the detection as a look-alike (algae, calm water, wind shadow, or rain cell).</p>
    </div>
  </div>
  """}
</div>

<hr class="divider"/>

<!-- ========================================================
     PHASE D — WIND CONTEXT
     ======================================================== -->
<div class="section">
  <div class="phase-header">
    <div class="phase-badge" style="background:#f5a62322;color:#f5a623;border-color:#f5a62366;">D</div>
    <div class="phase-title">Wind Context Layer</div>
    <div class="phase-subtitle">ERA5 reanalysis (ECMWF / Copernicus CDS) &mdash; 10m wind at detection location and time</div>
  </div>

  {"<p class='unavail'>ERA5 wind data was not fetched for this detection. Configure CDS API credentials to enable wind context.</p>" if wind_speed is None else f"""
  <div class="two-col">
    <div>
      <table class="data-table">
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Wind speed</td><td class="val">{wind_speed} m/s</td></tr>
        <tr><td>Wind direction</td><td class="val">{f"{wind_dir}°" if wind_dir is not None else "—"} (coming FROM)</td></tr>
        <tr><td>SAR validity</td><td class="val" style="color:{sar_valid_color.get(sar_validity,'#6b8299')};">{sar_validity.upper().replace("_"," ")}</td></tr>
        <tr><td>Look-alike wind risk</td><td class="val" style="color:{risk_color.get(wind_risk,'#6b8299')};">{wind_risk.upper()}</td></tr>
      </table>
      {f'<p class="field-note">{sar_detail}</p>' if sar_detail else ""}
      {f'<p class="field-note">{wind_risk_note}</p>' if wind_risk_note else ""}
    </div>
    <div>
      <p class="field-label">Predicted Drift Vector (3% empirical wind drift)</p>
      <table class="data-table">
        <tr><th>Horizon</th><th>Distance</th></tr>
        <tr><td>Bearing</td><td class="val">{f"{drift_bearing}°" if drift_bearing is not None else "—"}</td></tr>
        <tr><td>Drift speed</td><td class="val">{f"{drift_speed} m/s" if drift_speed is not None else "—"}</td></tr>
        <tr><td>6h displacement</td><td class="val">{f"{drift_6h} km" if drift_6h is not None else "—"}</td></tr>
        <tr><td>24h displacement</td><td class="val">{f"{drift_24h} km" if drift_24h is not None else "—"}</td></tr>
      </table>
      <p class="field-note">Drift is calculated using a 3% empirical wind drift factor with Coriolis deflection (+15° in northern hemisphere). This is the industry standard used in operational spill trajectory modelling.</p>
    </div>
  </div>
  """}
</div>

<hr class="divider"/>

<!-- ========================================================
     PHASE E — SENTINEL-2 OPTICAL CROSS-VALIDATION
     ======================================================== -->
<div class="section">
  <div class="phase-header">
    <div class="phase-badge" style="background:#4a9eff22;color:#4a9eff;border-color:#4a9eff66;">E</div>
    <div class="phase-title">Sentinel-2 Optical Cross-Validation</div>
    <div class="phase-subtitle">S2 L2A scene &mdash; spectral oil spill indices + visual confirmation</div>
  </div>

  {"<p class='unavail'>No Sentinel-2 scene was available within &plusmn;3 days of this detection. Place a scene in data/scenes/ and re-validate to add optical confirmation.</p>" if optical_verdict == "unavailable" else f"""
  <div class="two-col">
    <div>
      <table class="data-table">
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Verdict</td><td class="val" style="color:{c_verdict.get(optical_verdict,'#6b8299')};">{optical_verdict.upper()}</td></tr>
        <tr><td>Confidence</td><td class="val">{f"{round((optical_confidence or 0)*100)}%" if optical_confidence is not None else "—"}</td></tr>
        <tr><td>Cloud cover</td><td class="val">{f"{optical_cloud:.1f}%" if optical_cloud is not None else "—"}</td></tr>
        <tr><td>Scene</td><td class="val mono">{optical_scene}</td></tr>
      </table>
    </div>
    <div>
      <p class="field-label">Spectral Indices</p>
      <table class="data-table">
        <tr><th>Index</th><th>Value</th><th>Interpretation</th></tr>
        <tr>
          <td>OSI <small>(B03+B04)/B02</small></td>
          <td class="val" style="color:{osi_color(optical_osi)};">{f"{optical_osi:.4f}" if optical_osi is not None else "—"}</td>
          <td><small>Oil: &gt;1.8 possible, &gt;2.5 confident</small></td>
        </tr>
        <tr>
          <td>SWIRI <small>B11/(B8A+B11)</small></td>
          <td class="val" style="color:{swiri_color(optical_swiri)};">{f"{optical_swiri:.4f}" if optical_swiri is not None else "—"}</td>
          <td><small>Oil: &lt;0.30 possible, &lt;0.20 confident</small></td>
        </tr>
        <tr>
          <td>NDWI <small>(B03-B08)/(B03+B08)</small></td>
          <td class="val" style="color:{ndwi_color(optical_ndwi)};">{f"{optical_ndwi:.4f}" if optical_ndwi is not None else "—"}</td>
          <td><small>&gt;0.0 = water confirmed</small></td>
        </tr>
      </table>
      {f'<p class="field-note">{optical_reason}</p>' if optical_reason else ""}
    </div>
  </div>

  <div class="thumb-grid">
    {_thumbnail_img(thumb_rgb, "True Colour (B4/B3/B2)")}
    {_thumbnail_img(thumb_fc, "SWIR False Colour (B11/B8A/B4) — Oil appears black")}
  </div>
  """}
</div>

<hr class="divider"/>

<!-- ========================================================
     METHODOLOGY
     ======================================================== -->
<div class="section methodology">
  <div class="section-title">Methodology Notes</div>
  <p>This report was generated by <strong>Spectra</strong>, an AI-powered oil spill detection platform. All detections are processed through the following peer-reviewed pipeline:</p>
  <ul>
    <li><strong>SAR Segmentation (U-Net):</strong> A ResNet34-encoder U-Net (IoU 0.7066) segments spill pixels from Sentinel-1 GRD SAR data in VV+VH polarisation at 256&times;256 patch resolution.</li>
    <li><strong>Look-alike Classification (Phase C):</strong> A MobileNet-style binary CNN classifies the detected region as confirmed oil or a look-alike. Oil probability must exceed 0.60 to trigger an alert.</li>
    <li><strong>Wind Context (Phase D):</strong> ERA5 reanalysis 10m wind data is fetched from the Copernicus Climate Data Store at the detection location and acquisition time. SAR oil detection is reliable only between 2&ndash;10 m/s. A 3% empirical drift factor computes slick trajectory.</li>
    <li><strong>Optical Cross-Validation (Phase E):</strong> The nearest Sentinel-2 L2A scene within &plusmn;3 days is processed. The Oil Spill Index (OSI), SWIR suppression ratio (SWIRI), and NDWI are computed at the detection centroid. SWIRI &lt;0.20 indicates strong SWIR backscatter suppression consistent with oil &mdash; peer-reviewed accuracy &gt;88%, specificity &gt;95%.</li>
  </ul>
  <p>All detections are stored permanently in the Spectra database with a full evidence trail for legal and regulatory use.</p>
</div>

<!-- ========================================================
     FOOTER
     ======================================================== -->
<div class="report-footer">
  <div>Spectra v{SPECTRA_VERSION} &mdash; Generated {generated_at}</div>
  <div>Data sources: Sentinel-1 (ESA/Copernicus) &middot; ERA5 (ECMWF/Copernicus CDS) &middot; Sentinel-2 L2A (ESA/Copernicus)</div>
  <div>This document is computer-generated. All timestamps are UTC. Retain the original database record as the authoritative evidence source.</div>
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_BASE_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&family=Barlow:wght@300;400;500&display=swap');

  :root {
    --bg:      #0a0e13;
    --panel:   #0f1520;
    --panel2:  #141c2a;
    --border:  #1e2d42;
    --teal:    #00d4aa;
    --red:     #ff4444;
    --amber:   #f5a623;
    --blue:    #4a9eff;
    --text:    #c8d8e8;
    --text-dim:#6b8299;
    --mono:    'Share Tech Mono', monospace;
    --display: 'Barlow Condensed', sans-serif;
    --body:    'Barlow', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--body);
    font-size: 12px;
    line-height: 1.6;
    padding: 0;
  }

  /* ── Cover ── */
  .cover {
    background: var(--panel);
    border-bottom: 2px solid var(--teal);
    padding: 40px 48px 32px;
    page-break-after: avoid;
  }

  .cover-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 24px;
  }

  .cover-logo {
    font-family: var(--display);
    font-size: 38px;
    font-weight: 700;
    letter-spacing: 8px;
    color: var(--teal);
    text-transform: uppercase;
  }

  .cover-sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 1px;
  }

  .cover-id {
    font-family: var(--display);
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 4px;
    color: var(--text-bright);
    text-transform: uppercase;
    margin-bottom: 20px;
    color: #e8f4ff;
  }

  .cover-meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .cover-meta-item {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
  }

  .cmi-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .cmi-value {
    font-family: var(--display);
    font-size: 16px;
    font-weight: 600;
    color: var(--teal);
  }

  .map-link {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 8px;
  }

  .map-link a { color: var(--blue); }

  /* ── Sections ── */
  .section {
    padding: 28px 48px;
    page-break-inside: avoid;
  }

  .section-title {
    font-family: var(--display);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 16px;
  }

  .phase-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
  }

  .phase-badge {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    border: 1px solid;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--display);
    font-size: 16px;
    font-weight: 700;
    flex-shrink: 0;
  }

  .phase-title {
    font-family: var(--display);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #e8f4ff;
  }

  .phase-subtitle {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    margin-left: auto;
  }

  /* ── Evidence chain ── */
  .ev-chain {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }

  .ev-pill {
    border: 1px solid;
    border-radius: 4px;
    padding: 8px 14px;
    background: rgba(255,255,255,0.02);
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 160px;
  }

  .ev-phase {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .ev-verdict {
    font-family: var(--display);
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
  }

  /* ── Data tables ── */
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }

  .data-table th {
    font-family: var(--mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    background: var(--panel2);
  }

  .data-table td {
    padding: 7px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text-dim);
  }

  .data-table .val {
    font-family: var(--display);
    font-size: 14px;
    font-weight: 600;
    color: #e8f4ff;
  }

  .data-table .mono {
    font-family: var(--mono);
    font-size: 9px;
    word-break: break-all;
  }

  /* ── Two column layout ── */
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }

  /* ── Confidence bar ── */
  .conf-bar-wrap {
    height: 8px;
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 4px;
    position: relative;
    overflow: hidden;
    margin: 8px 0 4px;
  }

  .conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }

  .conf-bar-label {
    position: absolute;
    right: 6px;
    top: -14px;
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
  }

  /* ── Chips ── */
  .chip {
    display: inline-block;
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 3px;
  }

  /* ── Field labels / notes ── */
  .field-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    margin-top: 12px;
  }

  .field-note {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 8px;
    line-height: 1.5;
  }

  /* ── Thumbnails ── */
  .thumb-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 20px;
  }

  .thumb-wrap {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .thumb-wrap img {
    width: 100%;
    display: block;
  }

  .thumb-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
    text-align: center;
    padding: 6px;
    background: var(--panel2);
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .thumb-placeholder {
    border: 1px solid var(--border);
    border-radius: 4px;
    height: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: var(--panel2);
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 10px;
  }

  .thumb-placeholder small {
    font-size: 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* ── Misc ── */
  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0 48px;
  }

  .unavail {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    font-style: italic;
    padding: 12px;
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 4px;
  }

  /* ── Methodology ── */
  .methodology p {
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 10px;
    line-height: 1.7;
  }

  .methodology ul {
    padding-left: 18px;
    margin-bottom: 10px;
  }

  .methodology li {
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 6px;
    line-height: 1.6;
  }

  .methodology li strong {
    color: var(--text);
  }

  /* ── Footer ── */
  .report-footer {
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 16px 48px;
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
    display: flex;
    flex-direction: column;
    gap: 4px;
    letter-spacing: 0.5px;
  }
"""

# Supplemental CSS applied only during PDF rendering for pagination control
_PDF_PRINT_CSS = """
  @page {
    size: A4;
    margin: 0;
  }
  .section { page-break-inside: avoid; }
  .cover   { page-break-after: always; }
  .report-footer { position: running(footer); }
"""