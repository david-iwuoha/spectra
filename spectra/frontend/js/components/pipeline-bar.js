import { incrementScanCount } from '../state.js';
import { triggerScanRequest, fetchDetection } from '../api.js';
import { addDetectionToMap } from '../map/layers.js';
import { addDetectionCard } from './detection-card.js';
import { updateStats } from './stats.js';
import { sleep } from '../utils/async.js';
import { detections } from '../state.js';

const STAGE_ORDER = ['ps-sar', 'ps-classify', 'ps-wind', 'ps-optical', 'ps-ais', 'ps-done'];

function setPipelineStage(activeStageId) {
  const activeIdx = STAGE_ORDER.indexOf(activeStageId);
  STAGE_ORDER.forEach(function (id, i) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (i < activeIdx) el.classList.add('done');
    else if (id === activeStageId) el.classList.add('active');
  });
  const fill = document.getElementById('pipelineFill');
  if (fill) fill.style.width = ((activeIdx + 1) / STAGE_ORDER.length * 100) + '%';
}

function clearPipelineStages() {
  STAGE_ORDER.forEach(function (id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active', 'done');
  });
  const fill = document.getElementById('pipelineFill');
  if (fill) fill.style.width = '0%';
}

function setTopbarStatus(state) {
  const dot = document.getElementById('pipeStatusDot');
  const text = document.getElementById('pipeStatusText');
  if (state === 'running') {
    dot.className = 'status-dot warn';
    text.textContent = 'Pipeline Running';
  } else if (state === 'done') {
    dot.className = 'status-dot';
    text.textContent = 'Pipeline Complete';
  } else {
    dot.className = 'status-dot idle';
    text.textContent = 'Pipeline Idle';
  }
}

// NOTE: stage timing below is still a fixed setTimeout choreography, not a
// reflection of real backend progress — see the project plan's credibility
// findings (fake pipeline-stage animation). This is carried over unchanged
// for this mechanical split and replaced with real job-status polling once
// the backend scan endpoint exists.
const STAGE_TIMELINE = [
  { id: 'ps-sar', delay: 800 },
  { id: 'ps-classify', delay: 900 },
  { id: 'ps-wind', delay: 1000 },
  { id: 'ps-optical', delay: 1100 },
  { id: 'ps-ais', delay: 900 },
  { id: 'ps-done', delay: 500 },
];

export async function triggerScan() {
  const btn = document.getElementById('scanBtn');
  const bar = document.getElementById('pipelineBar');
  btn.classList.add('scanning');
  btn.textContent = 'Scanning...';
  btn.disabled = true;
  bar.classList.add('active');
  clearPipelineStages();
  setTopbarStatus('running');
  document.getElementById('statScans').textContent = incrementScanCount();

  for (const stage of STAGE_TIMELINE) {
    await sleep(stage.delay);
    setPipelineStage(stage.id);
  }

  try {
    const data = await triggerScanRequest();
    const scanId = data.scan_id;

    await sleep(15000);
    const det = await fetchDetection(scanId);
    if (det && det.id && !detections[scanId]) {
      addDetectionToMap(det);
      addDetectionCard(det);
      updateStats(det);
    }
  } catch (e) {
    console.warn('Backend unreachable:', e);
    setTopbarStatus('idle');
    bar.classList.remove('active');
    clearPipelineStages();
    btn.classList.remove('scanning');
    btn.textContent = 'Run Scan';
    btn.disabled = false;
    alert('Could not reach the backend. Ensure the API server is running and the URL in js/config.js is correct.');
    return;
  }

  await sleep(400);
  setTopbarStatus('done');
  bar.classList.remove('active');
  clearPipelineStages();
  btn.classList.remove('scanning');
  btn.textContent = 'Run Scan';
  btn.disabled = false;
}
