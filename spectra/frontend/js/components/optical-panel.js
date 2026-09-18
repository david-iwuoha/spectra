import { detections } from '../state.js';
import { requestOptical, fetchDetection } from '../api.js';
import { verdictIcon, verdictChipClass, indexClass } from '../utils/format.js';

export function renderOpticalPanel(d) {
  const v = d.optical_verdict, cf = d.optical_confidence, cloud = d.optical_cloud_fraction,
        osi = d.optical_osi, swiri = d.optical_swiri, ndwi = d.optical_ndwi,
        sceneName = d.optical_scene_name, reason = d.optical_reason, validatedAt = d.optical_validated_at,
        rgb = !!d.optical_thumbnail_rgb, falseColour = !!d.optical_thumbnail_falsecolour;

  if (!v || v === 'unavailable') {
    return '<div class="optical-panel optical-unavailable">' +
      '<div class="sub-panel-header"><span class="sub-panel-icon">◉</span><span class="sub-panel-title">Sentinel-2 Cross-Validation</span><span class="sub-panel-badge">S2</span></div>' +
      '<div class="optical-unavail-msg">' + (reason || 'No S2 scene available within ±3 days.') + '<br><span style="color:var(--text-faint);font-size:8px;">Place .SAFE in data/scenes/ or use Re-validate</span></div></div>';
  }

  const pct = cf != null ? Math.round(cf * 100) : null;
  const vc = verdictChipClass(v), icon = verdictIcon(v), label = v.toUpperCase();
  const osiClass = indexClass(osi, { good: [2.5, null], warn: [1.8, 2.5] });
  const swiriClass = indexClass(swiri, { good: [null, 0.20], warn: [0.20, 0.30] });
  const ndwiClass = indexClass(ndwi, { good: [0, null], warn: [null, 0] });
  const cloudLabel = cloud != null ? cloud.toFixed(1) + '% cloud' : '';
  const sceneShort = sceneName ? sceneName.substring(0, 26) + '…' : '';
  const footerTime = validatedAt ? new Date(validatedAt).toLocaleString() : '';

  let thumbs = '';
  if (rgb || falseColour) {
    thumbs = '<div class="optical-thumbs-row">';
    if (rgb) thumbs += '<div class="optical-thumb" onclick="openThumbnailModal(\'' + d.id + '\',\'rgb\')" title="True colour"><img src="' + d.optical_thumbnail_rgb + '" alt="RGB"/><span class="thumb-label">True Colour</span></div>';
    if (falseColour) thumbs += '<div class="optical-thumb" onclick="openThumbnailModal(\'' + d.id + '\',\'falsecolour\')" title="SWIR false colour"><img src="' + d.optical_thumbnail_falsecolour + '" alt="FC"/><span class="thumb-label">SWIR False Colour</span></div>';
    thumbs += '</div>';
  }

  return '<div class="optical-panel">' +
    '<div class="sub-panel-header"><span class="sub-panel-icon">◉</span><span class="sub-panel-title">Sentinel-2 Cross-Validation</span><span class="sub-panel-badge">S2 L2A</span></div>' +
    '<div class="optical-verdict-row">' +
    '<div class="optical-verdict-chip ' + vc + '" title="' + (reason || '') + '">' + icon + ' &nbsp;' + label + (pct != null ? ' <span class="optical-conf-pct">' + pct + '%</span>' : '') + ' </div>' +
    (cloudLabel ? '<div class="optical-cloud-chip">☁ ' + cloudLabel + '</div>' : '') +
    '</div>' +
    '<div class="optical-indices-row">' +
    '<div class="optical-idx-cell ' + osiClass + '" title="Oil Spill Index"><span class="idx-val">' + (osi != null ? osi.toFixed(3) : '—') + '</span><span class="idx-lbl">OSI</span></div>' +
    '<div class="optical-idx-cell ' + swiriClass + '" title="SWIR Index"><span class="idx-val">' + (swiri != null ? swiri.toFixed(3) : '—') + '</span><span class="idx-lbl">SWIRI</span></div>' +
    '<div class="optical-idx-cell ' + ndwiClass + '" title="Water Index"><span class="idx-val">' + (ndwi != null ? ndwi.toFixed(3) : '—') + '</span><span class="idx-lbl">NDWI</span></div>' +
    '</div>' + thumbs +
    ((sceneShort || footerTime) ? '<div class="optical-footer">' + (sceneShort ? '<span title="' + (sceneName || '') + '">' + sceneShort + '</span>' : '') + (footerTime ? '<span>' + footerTime + '</span>' : '') + ' </div>' : '') +
    '</div>';
}

export function renderOpticalVerdictBadge(d) {
  const v = d.optical_verdict;
  if (!v || v === 'unavailable') return '';
  const cls = verdictChipClass(v);
  const vbClass = cls === 'ov-confirmed' ? 'vb-valid' : cls === 'ov-unconfirmed' ? 'vb-invalid' : 'vb-borderline';
  return '<span class="verdict-badge ' + vbClass + '" title="S2: ' + (d.optical_reason || '') + '">' + verdictIcon(v) + ' S2 ' + v.toUpperCase() + '</span>';
}

export function openThumbnailModal(id, kind) {
  const d = detections[id];
  if (!d) return;
  const src = kind === 'rgb' ? d.optical_thumbnail_rgb : d.optical_thumbnail_falsecolour;
  if (!src) return;
  document.getElementById('thumbnailModalImg').src = src;
  document.getElementById('thumbnailModalCaption').textContent = kind === 'rgb' ? 'True Colour (B4/B3/B2)' : 'False Colour SWIR (B11/B8A/B4)';
  document.getElementById('thumbnailModalSubtitle').textContent = kind === 'rgb'
    ? 'RGB composite — oil appears silvery or dark against blue water'
    : 'SWIR false colour — oil appears deep black, a diagnostic signature for spill confirmation';
  document.getElementById('thumbnailModal').classList.add('active');
}

export function closeThumbnailModal() {
  document.getElementById('thumbnailModal').classList.remove('active');
  document.getElementById('thumbnailModalImg').src = '';
}

export async function revalidateOptical(id, el) {
  const btn = el.querySelector('.action-btn[data-action="optical"]');
  if (btn) { btn.textContent = 'Validating...'; btn.disabled = true; }
  try {
    await requestOptical(id);
    const full = await fetchDetection(id);
    detections[id] = full;
    const container = el.querySelector('.optical-panel-container');
    if (container) container.innerHTML = renderOpticalPanel(full);
  } catch (e) { console.warn('Optical error:', e); }
  if (btn) { btn.textContent = 'S2'; btn.disabled = false; }
}
