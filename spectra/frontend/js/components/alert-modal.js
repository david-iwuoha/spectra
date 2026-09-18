import { alertState } from '../state.js';
import { dispatchAlertRequest, downloadReport } from '../api.js';
import { bearingToCardinal, validityIcon, verdictIcon } from '../utils/format.js';
import { sleep } from '../utils/async.js';

export function downloadCurrentReport() {
  downloadReport(alertState.currentDetection && alertState.currentDetection.id);
}

function showBanner(el, html) {
  if (html == null) { el.classList.remove('active'); return; }
  el.innerHTML = html;
  el.classList.add('active');
}

export function openAlertModal(det) {
  alertState.currentDetection = det;
  document.getElementById('alertModal').classList.add('active');

  document.getElementById('modalDetectionInfo').innerHTML =
    'ID: ' + det.id + ' &nbsp;|&nbsp; Confidence: <strong style="color:var(--alert-2)">' + det.confidence + '%</strong>' +
    ' &nbsp;|&nbsp; Area: <strong>' + det.area_km2 + ' km²</strong>' +
    ' &nbsp;|&nbsp; ' + new Date(det.detected_at).toLocaleString();

  const classifierBanner = document.getElementById('modalClassifierBanner');
  if (det.lookalike_score != null) {
    const passed = det.lookalike_passed, pct = Math.round(det.lookalike_score * 100);
    showBanner(classifierBanner, passed
      ? '◆ Look-alike: <strong>Oil confirmed</strong> &nbsp;— ' + pct + '% &nbsp;· Alert gate open'
      : '◇ Look-alike: <strong>Suppressed</strong> &nbsp;— ' + det.lookalike_label + ' (' + pct + '%) &nbsp;· Manual override available');
  } else { showBanner(classifierBanner, null); }

  const windBanner = document.getElementById('modalWindBanner');
  if (det.wind_speed_ms != null) {
    const driftNote = det.drift_vector ? ' &nbsp;· Drift: ' + det.drift_vector['24h_km'] + ' km @ ' + det.drift_vector.bearing_deg + '°' : '';
    showBanner(windBanner, '◎ ERA5 Wind: <strong>' + det.wind_speed_ms + ' m/s · ' + bearingToCardinal(det.wind_direction_deg) + '</strong>' +
      ' &nbsp;| SAR: <strong>' + validityIcon(det.sar_validity) + ' ' + (det.sar_validity || '').replace(/_/g, ' ').toUpperCase() + '</strong>' +
      ' &nbsp;| Risk: <strong>' + (det.lookalike_wind_risk || '—').toUpperCase() + '</strong>' + driftNote);
  } else { showBanner(windBanner, null); }

  const opticalBanner = document.getElementById('modalOpticalBanner');
  if (det.optical_verdict && det.optical_verdict !== 'unavailable') {
    showBanner(opticalBanner, '◉ Sentinel-2: <strong>' + verdictIcon(det.optical_verdict) + ' ' + det.optical_verdict.toUpperCase() +
      (det.optical_confidence != null ? ' ' + Math.round(det.optical_confidence * 100) + '%' : '') + '</strong>' +
      (det.optical_osi != null ? ' &nbsp;| OSI: <strong>' + det.optical_osi.toFixed(3) + '</strong>' : '') +
      (det.optical_swiri != null ? ' &nbsp;| SWIRI: <strong>' + det.optical_swiri.toFixed(3) + '</strong>' : '') +
      (det.optical_cloud_fraction != null ? ' &nbsp;| Cloud: ' + det.optical_cloud_fraction.toFixed(1) + '%' : ''));
  } else { showBanner(opticalBanner, null); }

  const aisBanner = document.getElementById('modalAISBanner'), top = det.ais_top_suspect;
  if (top) {
    const score = top.suspicion_score || 0;
    showBanner(aisBanner, '⊕ AIS Vessel: <strong>' + top.name + '</strong> &nbsp;| Type: <strong>' + top.vessel_type + '</strong>' +
      ' &nbsp;| Distance: <strong>' + top.distance_nm + ' nm</strong>' +
      ' &nbsp;| Suspicion: <strong>' + score + '/10</strong>');
  } else { showBanner(aisBanner, null); }

  renderFlow();
}

export function closeAlertModal() {
  document.getElementById('alertModal').classList.remove('active');
  alertState.emails = [];
  renderEmailTags();
  renderFlow();
}

export function addEmail() {
  const input = document.getElementById('emailInput'), email = input.value.trim();
  if (!email || !email.includes('@')) return;
  if (alertState.emails.includes(email)) { input.value = ''; return; }
  alertState.emails.push(email);
  input.value = '';
  renderEmailTags();
  renderFlow();
}

export function removeEmail(email) {
  alertState.emails = alertState.emails.filter((x) => x !== email);
  renderEmailTags();
  renderFlow();
}

function renderEmailTags() {
  document.getElementById('emailTags').innerHTML = alertState.emails.map((e) =>
    '<div class="email-tag">' + e + '<span class="email-tag-remove" onclick="removeEmail(\'' + e + '\')">&#215;</span></div>'
  ).join('');
}

function renderFlow() {
  const svg = document.getElementById('flowSvg'), W = svg.parentElement.offsetWidth - 40, H = 190;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const emails = alertState.emails.length > 0 ? alertState.emails : ['(no recipients)'];
  const sx = 90, sy = H / 2, ex = W - 70, count = emails.length;
  const sp = Math.min(55, (H - 40) / Math.max(count, 1)), sy0 = H / 2 - ((count - 1) * sp) / 2;
  const hasRecipients = alertState.emails.length > 0;

  let html = '<rect x="' + (sx - 38) + '" y="' + (sy - 24) + '" width="76" height="48" rx="2" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>' +
    '<text x="' + sx + '" y="' + (sy - 6) + '" text-anchor="middle" font-family="DM Mono,monospace" font-size="9" fill="rgba(255,255,255,0.5)" letter-spacing="2">SPECTRA</text>' +
    '<text x="' + sx + '" y="' + (sy + 8) + '" text-anchor="middle" font-family="DM Mono,monospace" font-size="8" fill="rgba(255,255,255,0.25)">SERVER</text>' +
    '<text x="' + sx + '" y="' + (sy + 22) + '" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.4)">⚡</text>';

  emails.forEach(function (e, i) {
    const ey = sy0 + i * sp, mx = sx + (ex - sx) / 2;
    html += '<path id="line-' + i + '" d="M' + (sx + 38) + ',' + sy + ' C' + mx + ',' + sy + ' ' + mx + ',' + ey + ' ' + (ex - 26) + ',' + ey + '" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="4,3"/>' +
      '<circle cx="' + ex + '" cy="' + ey + '" r="22" fill="rgba(255,255,255,0.02)" stroke="' + (hasRecipients ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)') + '" stroke-width="1"/>' +
      '<text x="' + ex + '" y="' + (ey - 3) + '" text-anchor="middle" font-size="12" fill="' + (hasRecipients ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.15)') + '">✉</text>' +
      '<text x="' + ex + '" y="' + (ey + 11) + '" text-anchor="middle" font-family="DM Mono,monospace" font-size="7" fill="' + (hasRecipients ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.1)') + '">' + (hasRecipients ? e.split('@')[0].substring(0, 10) : '—') + '</text>';
  });

  svg.innerHTML = html;
}

async function animatePackets() {
  const svg = document.getElementById('flowSvg'), count = alertState.emails.length;
  for (let i = 0; i < count; i++) {
    const path = svg.querySelector('#line-' + i);
    if (!path) continue;
    path.setAttribute('stroke', 'rgba(45,106,79,0.4)');
    path.setAttribute('stroke-width', '1.5');
    const packet = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    packet.setAttribute('font-size', '10');
    packet.setAttribute('fill', 'rgba(45,106,79,0.9)');
    packet.textContent = '📄';
    const len = path.getTotalLength(), dur = 700;
    let start = null;
    await new Promise((resolve) => {
      function step(ts) {
        if (!start) start = ts;
        const t = Math.min((ts - start) / dur, 1), pt = path.getPointAtLength(t * len);
        packet.setAttribute('x', pt.x - 5);
        packet.setAttribute('y', pt.y + 5);
        if (!packet.parentNode) svg.appendChild(packet);
        if (t < 1) requestAnimationFrame(step);
        else { packet.remove(); resolve(); }
      }
      requestAnimationFrame(step);
    });
    await sleep(120);
  }
}

export async function dispatchAlerts() {
  if (alertState.emails.length === 0) { alert('Please add at least one recipient email.'); return; }
  const btn = document.getElementById('dispatchBtn');
  btn.textContent = 'Dispatching...';
  btn.disabled = true;
  await animatePackets();

  try {
    const result = await dispatchAlertRequest(alertState.currentDetection.id, alertState.emails);
    console.log('Dispatch result:', result);
  } catch (e) { console.warn('Dispatch error:', e); }

  const svg = document.getElementById('flowSvg');
  alertState.emails.forEach(function (_, i) {
    const l = svg.querySelector('#line-' + i);
    if (l) { l.setAttribute('stroke', 'rgba(45,106,79,0.6)'); l.setAttribute('stroke-dasharray', 'none'); }
  });
  btn.textContent = 'Sent';
  btn.style.background = 'var(--confirm)';
  btn.style.color = 'var(--white)';

  alertState.emails.forEach(function (e) {
    const log = document.getElementById('alertLog');
    const emptyLog = log.querySelector('.empty-state');
    if (emptyLog) emptyLog.remove();
    const entry = document.createElement('div');
    entry.className = 'alert-log-entry dispatched';
    entry.textContent = new Date().toLocaleTimeString() + ' → ' + e;
    log.prepend(entry);
  });

  const dispatchedCount = parseInt(document.getElementById('statAlerts').textContent, 10) + alertState.emails.length;
  document.getElementById('statAlerts').textContent = dispatchedCount;

  await sleep(1800);
  closeAlertModal();
  btn.textContent = 'Dispatch Alert';
  btn.style.background = '';
  btn.style.color = '';
  btn.disabled = false;
}
