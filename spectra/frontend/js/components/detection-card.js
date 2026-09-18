import { detections, layers } from '../state.js';
import { fetchDetections } from '../api.js';
import { addDetectionToMap, buildHeatmap } from '../map/layers.js';
import { updateStats } from './stats.js';
import { renderWindPanel, renderSARValidityBadge } from './wind-panel.js';
import { renderOpticalPanel, renderOpticalVerdictBadge } from './optical-panel.js';
import { renderAISPanel, renderAISVerdictBadge } from './ais-panel.js';

export function renderLookalikeChip(d) {
  if (d.lookalike_score == null) {
    return '<div class="lookalike-chip lookalike-pending">&#9675; &nbsp;Classifier pending</div>';
  }
  const s = d.lookalike_score, passed = d.lookalike_passed, pct = Math.round(s * 100);
  const cls = passed ? (s >= 0.85 ? 'lookalike-confirmed' : 'lookalike-probable') : 'lookalike-rejected';
  const icon = passed ? '◆' : '◇';
  const label = passed ? 'Oil confirmed &nbsp;' + pct + '%' : 'Look-alike &nbsp;' + pct + '%';
  return '<div class="lookalike-chip ' + cls + '" title="' + (passed ? 'Confirmed (' + pct + '%)' : 'Suppressed: ' + d.lookalike_label + ' (' + pct + '%)') + '">' +
    '<span>' + icon + ' &nbsp;' + label + '</span>' +
    '<div class="chip-bar"><div class="chip-fill" style="width:' + pct + '%"></div></div></div>';
}

export function addDetectionCard(d) {
  const list = document.getElementById('detectionList');
  const empty = list.querySelector('.empty-state');
  if (empty) empty.remove();

  const conf = d.confidence, level = conf >= 70 ? 'high' : conf >= 50 ? 'medium' : 'low';
  const time = new Date(d.detected_at).toLocaleTimeString();
  detections[d.id] = d;

  const badge = d.lookalike_passed === false
    ? '<span class="card-badge badge-suppressed">Suppressed</span>'
    : d.alert_sent
      ? '<span class="card-badge badge-alert">Alert Sent</span>'
      : '<span class="card-badge badge-ok">Monitoring</span>';

  const sarBadge = renderSARValidityBadge(d);
  const opticalBadge = renderOpticalVerdictBadge(d);
  const aisBadge = renderAISVerdictBadge(d);

  const card = document.createElement('div');
  card.className = 'detection-card ' + level;
  card.dataset.detectionId = d.id;
  card.innerHTML =
    '<div class="card-id">' + d.id + ' · ' + time + '</div>' +
    '<div class="card-confidence">' + conf + '<span>% confidence</span></div>' +
    '<div class="card-meta">' +
    '<div class="card-stat"><strong>' + d.area_km2 + '</strong> km²</div>' +
    '<div class="card-stat"><strong>' + (d.spill_pixels ? d.spill_pixels.toLocaleString() : '—') + '</strong> px</div>' +
    '</div>' +
    renderLookalikeChip(d) +
    '<div class="wind-panel-container">' + renderWindPanel(d) + '</div>' +
    '<div class="optical-panel-container">' + renderOpticalPanel(d) + '</div>' +
    '<div class="ais-panel-container">' + renderAISPanel(d) + '</div>' +
    '<div class="card-footer">' +
    '<div class="card-footer-badges">' + badge + sarBadge + opticalBadge + aisBadge + '</div>' +
    '<div class="card-footer-actions">' +
    '<button class="action-btn" data-action="wind" onclick="refreshWindForDetection(\'' + d.id + '\',this.closest(\'.detection-card\'))">Wind</button>' +
    '<button class="action-btn" data-action="optical" onclick="revalidateOptical(\'' + d.id + '\',this.closest(\'.detection-card\'))">S2</button>' +
    '<button class="action-btn" data-action="ais" onclick="reattributeAIS(\'' + d.id + '\',this.closest(\'.detection-card\'))">AIS</button>' +
    '<button class="action-btn" onclick="downloadReport(\'' + d.id + '\')" title="Download PDF evidence report">↓ PDF</button>' +
    '<button class="action-btn" onclick="openAlertModal(window._detections[\'' + d.id + '\'])">Dispatch</button>' +
    '</div></div>';

  list.prepend(card);

  if (d.alert_sent) {
    const log = document.getElementById('alertLog');
    const emptyLog = log.querySelector('.empty-state');
    if (emptyLog) emptyLog.remove();
    const entry = document.createElement('div');
    entry.className = 'alert-log-entry fired';
    entry.textContent = time + ' — Alert fired (' + conf + '%)';
    log.prepend(entry);
  }
}

export async function loadDetections() {
  try {
    const data = await fetchDetections();
    data.detections.forEach(function (d) {
      addDetectionToMap(d);
      addDetectionCard(d);
      updateStats(d);
    });
    if (layers.heatmap && Object.keys(detections).length > 0) buildHeatmap();
  } catch (e) {
    console.log('Backend not reachable on startup — dashboard is ready when you start the server.');
  }
}
