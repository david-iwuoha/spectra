import { detections } from '../state.js';
import { requestAIS, fetchDetection } from '../api.js';

export function renderAISPanel(d) {
  const found = d.ais_vessels_found, top = d.ais_top_suspect,
        note = d.ais_note || '', radius = d.ais_search_radius_nm || 10;

  if (found == null) {
    return '<div class="ais-panel ais-unavailable">' +
      '<div class="sub-panel-header"><span class="sub-panel-icon">⊕</span><span class="sub-panel-title">AIS Vessel Attribution</span><span class="sub-panel-badge">AIS</span></div>' +
      '<div class="ais-unavail-msg">Not yet run. Use Re-attribute.</div></div>';
  }
  if (!top || !found) {
    return '<div class="ais-panel">' +
      '<div class="sub-panel-header"><span class="sub-panel-icon">⊕</span><span class="sub-panel-title">AIS Vessel Attribution</span><span class="sub-panel-badge">AIS</span></div>' +
      '<div class="ais-unavail-msg">No vessels found within ' + radius + ' nm.</div>' +
      '<div class="ais-count-note">' + note + '</div></div>';
  }

  const score = top.suspicion_score || 0;
  const scoreClass = score >= 6 ? 'ais-score-high' : score >= 3 ? 'ais-score-medium' : 'ais-score-low';
  const flags = (top.suspicion_flags || []).join(' · ') || '—';

  return '<div class="ais-panel">' +
    '<div class="sub-panel-header"><span class="sub-panel-icon">⊕</span><span class="sub-panel-title">AIS Vessel Attribution</span><span class="sub-panel-badge">AIS</span></div>' +
    '<div class="ais-suspect-row">' +
    '<div><div class="ais-suspect-name">' + top.name + '</div><div class="ais-suspect-type">' + top.vessel_type + ' · MMSI ' + top.mmsi + '</div></div>' +
    '<span class="ais-score-chip ' + scoreClass + '">' + score + '/10</span>' +
    '</div>' +
    '<div class="ais-detail-row">' +
    '<div class="ais-detail-cell"><div class="ais-detail-label">Distance</div><div class="ais-detail-value">' + top.distance_nm + ' nm</div></div>' +
    '<div class="ais-detail-cell"><div class="ais-detail-label">Speed</div><div class="ais-detail-value">' + top.speed_knots + ' kts</div></div>' +
    '<div class="ais-detail-cell"><div class="ais-detail-label">Status</div><div class="ais-detail-value" style="font-size:10px;">' + top.nav_status_label + '</div></div>' +
    '</div>' +
    '<div class="ais-count-note">' + found + ' vessel(s) in ' + radius + ' nm · Flags: ' + flags + '</div>' +
    '<div class="ais-count-note">' + note + '</div></div>';
}

export function renderAISVerdictBadge(d) {
  if (d.ais_vessels_found == null) return '';
  const top = d.ais_top_suspect;
  if (!top) return '<span class="verdict-badge vb-neutral">⊕ AIS 0</span>';
  const score = top.suspicion_score || 0;
  const cls = score >= 6 ? 'vb-invalid' : score >= 3 ? 'vb-borderline' : 'vb-neutral';
  return '<span class="verdict-badge ' + cls + '">⊕ AIS ' + score + '/10</span>';
}

export async function reattributeAIS(id, el) {
  const btn = el.querySelector('.action-btn[data-action="ais"]');
  if (btn) { btn.textContent = 'Querying...'; btn.disabled = true; }
  try {
    await requestAIS(id);
    const full = await fetchDetection(id);
    if (!full) throw new Error('Detection ' + id + ' no longer exists');
    detections[id] = full;
    const container = el.querySelector('.ais-panel-container');
    if (container) container.innerHTML = renderAISPanel(full);
  } catch (e) { console.warn('AIS error:', e); }
  if (btn) { btn.textContent = 'AIS'; btn.disabled = false; }
}
