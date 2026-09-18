// Shared, mutable application state. Modules import what they need from
// here instead of reaching for globals directly.

export const detections = {};

export const layers = { spills: true, heatmap: false, pipelines: true, historical: true, drift: true };

export const alertState = {
  emails: [],
  currentDetection: null,
};

export let scanCount = 0;
export function incrementScanCount() {
  scanCount += 1;
  return scanCount;
}

export let pendingPolygon = null;
export function setPendingPolygon(geom) { pendingPolygon = geom; }
export function getPendingPolygon() { return pendingPolygon; }
