import { detections } from './state.js';
import { wireCoordDisplay } from './map/init.js';
import { toggleLayer } from './map/layers.js';
import { wireDrawEvents, loadWatchZones, closeWatchZoneModal, saveWatchZone } from './map/watch-zones.js';
import { loadDetections } from './components/detection-card.js';
import { refreshWindForDetection } from './components/wind-panel.js';
import { revalidateOptical, openThumbnailModal, closeThumbnailModal } from './components/optical-panel.js';
import { reattributeAIS } from './components/ais-panel.js';
import { openAlertModal, closeAlertModal, addEmail, removeEmail, dispatchAlerts, downloadCurrentReport } from './components/alert-modal.js';
import { triggerScan } from './components/pipeline-bar.js';
import { downloadReport } from './api.js';

// The markup still uses inline onclick="..." handlers (carried over as-is
// from the original single-file build to keep this split behavior-neutral).
// ES modules aren't global, so bridge the handful of functions the HTML
// references onto window. `window._detections` is the same object as
// state.js's `detections` — an alias kept only so the onclick strings
// generated in detection-card.js keep working unchanged.
window._detections = detections;
window.toggleLayer = toggleLayer;
window.triggerScan = triggerScan;
window.closeWatchZoneModal = closeWatchZoneModal;
window.saveWatchZone = saveWatchZone;
window.refreshWindForDetection = refreshWindForDetection;
window.revalidateOptical = revalidateOptical;
window.openThumbnailModal = openThumbnailModal;
window.closeThumbnailModal = closeThumbnailModal;
window.reattributeAIS = reattributeAIS;
window.openAlertModal = openAlertModal;
window.closeAlertModal = closeAlertModal;
window.addEmail = addEmail;
window.removeEmail = removeEmail;
window.dispatchAlerts = dispatchAlerts;
window.downloadReport = downloadReport;
window.downloadCurrentReport = downloadCurrentReport;

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toUTCString().split(' ')[4] + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

wireCoordDisplay();
wireDrawEvents();
loadWatchZones();
loadDetections();
