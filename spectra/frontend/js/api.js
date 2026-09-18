import { API } from './config.js';

export async function fetchDetections() {
  const res = await fetch(API + '/detections');
  return res.json();
}

export async function fetchDetection(id) {
  const res = await fetch(API + '/detections/' + id);
  return res.json();
}

export async function triggerScanRequest() {
  const res = await fetch(API + '/scan', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_id: 'latest' }),
  });
  return res.json();
}

export async function requestWind(id) {
  return fetch(API + '/detections/' + id + '/wind', { method: 'POST' });
}

export async function requestOptical(id) {
  return fetch(API + '/detections/' + id + '/optical', { method: 'POST' });
}

export async function requestAIS(id) {
  return fetch(API + '/detections/' + id + '/ais', { method: 'POST' });
}

export async function dispatchAlertRequest(detectionId, recipients) {
  const res = await fetch(API + '/alerts/dispatch', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ detection_id: detectionId, recipients }),
  });
  return res.json();
}

export async function saveWatchZoneRequest(payload) {
  const res = await fetch(API + '/watch-zones', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function fetchWatchZones() {
  const res = await fetch(API + '/watch-zones');
  return res.json();
}

export function downloadReport(id) {
  if (!id) return;
  const a = document.createElement('a');
  a.href = API + '/detections/' + id + '/report';
  a.setAttribute('download', 'spectra_' + id + '.pdf');
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
