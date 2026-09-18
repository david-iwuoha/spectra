import { detections } from '../state.js';
import { driftLayerGroup } from '../map/init.js';
import { addDriftArrow } from '../map/layers.js';
import { requestWind, fetchDetection } from '../api.js';
import { bearingToCardinal, validityIcon, validityChipClass, riskChipClass, compassSVG } from '../utils/format.js';

export function renderWindPanel(d) {
  const spd = d.wind_speed_ms, dir = d.wind_direction_deg, val = d.sar_validity,
        vdet = d.sar_validity_detail, risk = d.lookalike_wind_risk, rnote = d.lookalike_wind_note,
        dv = d.drift_vector, fetchedAt = d.wind_fetched_at;

  if (spd == null && val !== 'error') {
    return '<div class="wind-panel"><div class="sub-panel-header"><span class="sub-panel-icon">◎</span><span class="sub-panel-title">Wind Context</span><span class="sub-panel-badge">ERA5</span></div>' +
      '<div class="wind-unavailable-msg">CDS API not configured. <a href="https://cds.climate.copernicus.eu" target="_blank">Set up →</a></div></div>';
  }

  const vc = validityChipClass(val), rc = riskChipClass(risk), compass = compassSVG(dir);
  const cardinal = bearingToCardinal(dir || 0);
  const validityLabel = (val || 'unavailable').replace(/_/g, ' ').toUpperCase();
  const riskLabel = (risk || 'unknown').toUpperCase();
  const driftHtml = dv
    ? '<div class="wind-drift-section"><div class="wind-drift-label">Predicted Drift</div><div class="wind-drift-grid">' +
      '<div class="drift-cell"><span class="drift-val">' + dv.bearing_deg + '°</span><span class="drift-lbl">Bearing</span></div>' +
      '<div class="drift-cell"><span class="drift-val">' + dv['6h_km'] + '</span><span class="drift-lbl">km / 6h</span></div>' +
      '<div class="drift-cell"><span class="drift-val">' + dv['24h_km'] + '</span><span class="drift-lbl">km / 24h</span></div>' +
      '</div></div>'
    : '';
  const fetchedHtml = fetchedAt ? '<div class="wind-fetch-time">ERA5 · ' + new Date(fetchedAt).toLocaleString() + '</div>' : '';
  const detailHtml = vdet ? '<div class="wind-detail-txt">' + vdet + '</div>' : '';

  return '<div class="wind-panel">' +
    '<div class="sub-panel-header"><span class="sub-panel-icon">◎</span><span class="sub-panel-title">Wind Context</span><span class="sub-panel-badge">ERA5</span></div>' +
    '<div class="wind-main-row">' + compass + '<div><div class="wind-speed-big">' + spd + '<span>m/s</span></div><div class="wind-dir-text">From ' + cardinal + ' · ' + dir + '°</div></div></div>' +
    '<div class="wind-chips-row">' +
    '<div class="wind-validity-chip ' + vc + '" title="' + (vdet || '') + '">' + validityIcon(val) + ' SAR ' + validityLabel + '</div>' +
    '<div class="wind-risk-chip ' + rc + '" title="' + (rnote || '') + '">◇ Risk: ' + riskLabel + '</div>' +
    '</div>' + detailHtml + driftHtml + fetchedHtml + '</div>';
}

export function renderSARValidityBadge(d) {
  if (!d.sar_validity || d.sar_validity === 'unavailable') return '';
  const cls = validityChipClass(d.sar_validity), label = d.sar_validity.replace(/_/g, ' ').toUpperCase();
  const vbClass = cls === 'wv-valid' ? 'vb-valid' : cls === 'wv-borderline' ? 'vb-borderline' : 'vb-invalid';
  return '<span class="verdict-badge ' + vbClass + '" title="Wind ' + d.wind_speed_ms + ' m/s">' + validityIcon(d.sar_validity) + ' ' + label + '</span>';
}

export async function refreshWindForDetection(id, el) {
  const btn = el.querySelector('.action-btn[data-action="wind"]');
  if (btn) { btn.textContent = 'Fetching...'; btn.disabled = true; }
  try {
    await requestWind(id);
    const full = await fetchDetection(id);
    detections[id] = full;
    const container = el.querySelector('.wind-panel-container');
    if (container) container.innerHTML = renderWindPanel(full);
    driftLayerGroup.clearLayers();
    Object.values(detections).forEach((d) => { if (d.drift_vector) addDriftArrow(d); });
  } catch (e) { console.warn('Wind refresh error:', e); }
  if (btn) { btn.textContent = 'Wind'; btn.disabled = false; }
}
