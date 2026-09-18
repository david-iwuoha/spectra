// NOTE: `knownPipelines` and `documentedIncidents` below are carried over
// unchanged from the original single-file build for this mechanical split.
// They are hardcoded, uncited coordinates attributed to real organizations
// (SPDC, NOSDRA, UNEP, Shell) — see the project plan's credibility findings.
// They are removed / replaced with real, sourced data in the follow-up pass;
// do not treat them as verified records in the meantime.

export const map = L.map('map', { center: [5.0, 6.5], zoom: 8, zoomControl: true, attributionControl: false });

const baseLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 18, subdomains: 'abcd' });
baseLayer.on('tileerror', function () {
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);
});
baseLayer.addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18, opacity: 0.5 }).addTo(map);

export const spillLayerGroup = L.layerGroup().addTo(map);
export const pipelineLayerGroup = L.layerGroup().addTo(map);
export const historicalLayerGroup = L.layerGroup().addTo(map);
export const driftLayerGroup = L.layerGroup().addTo(map);
export let heatmapLayer = null;
export function setHeatmapLayer(layer) { heatmapLayer = layer; }

L.polygon([[4.0, 5.0], [4.0, 8.5], [6.5, 8.5], [6.5, 5.0], [4.0, 5.0]], {
  color: 'rgba(255,255,255,0.12)', weight: 1, fillColor: 'rgba(255,255,255,0.01)',
  fillOpacity: 1, dashArray: '4,6',
}).addTo(map);

const knownPipelines = [
  [[5.1, 5.2], [5.3, 5.8], [5.6, 6.2], [5.8, 6.8], [6.0, 7.2]],
  [[4.8, 5.5], [5.0, 6.0], [5.2, 6.5], [5.4, 7.0], [5.5, 7.5]],
  [[5.5, 5.8], [5.6, 6.3], [5.7, 6.9], [5.8, 7.4], [5.9, 8.0]],
  [[4.6, 6.0], [4.9, 6.4], [5.1, 6.8], [5.3, 7.2]],
  [[5.2, 5.3], [5.4, 5.9], [5.7, 6.4], [6.0, 6.8], [6.2, 7.1]],
];
knownPipelines.forEach(function (coords) {
  L.polyline(coords, { color: 'rgba(180,180,160,0.35)', weight: 1.5, opacity: 1, dashArray: '6,4' }).addTo(pipelineLayerGroup);
});

const documentedIncidents = [
  { lat: 4.85, lng: 6.35, name: 'Bonga OML 118', year: 2011, source: 'SPDC' },
  { lat: 5.10, lng: 6.65, name: 'Jesse, Ughelli North', year: 2012, source: 'NOSDRA' },
  { lat: 4.62, lng: 7.05, name: 'Bodo Creek, Ogoniland', year: 2008, source: 'UNEP' },
  { lat: 5.45, lng: 6.90, name: 'Forcados Export Terminal', year: 2016, source: 'Shell' },
  { lat: 5.78, lng: 7.35, name: 'Trans Niger Pipeline, Rivers', year: 2019, source: 'NOSDRA' },
  { lat: 4.95, lng: 5.85, name: 'Warri Refinery Corridor', year: 2020, source: 'NOSDRA' },
  { lat: 4.72, lng: 6.50, name: 'Sangana Offshore Pipeline', year: 2020, source: 'Open source SAR' },
  { lat: 4.80, lng: 6.42, name: 'Sangana Platform Leak', year: 2021, source: 'Open source SAR' },
];
documentedIncidents.forEach(function (s) {
  L.circleMarker([s.lat, s.lng], {
    radius: 6, color: 'rgba(139,115,85,0.6)', fillColor: '#8b7355', fillOpacity: 0.5, weight: 1,
  }).bindPopup(
    '<div class="popup-title">Documented Incident</div>' +
    '<div class="popup-row"><strong>Site:</strong> ' + s.name + '</div>' +
    '<div class="popup-row"><strong>Year:</strong> ' + s.year + '</div>' +
    '<div class="popup-row"><strong>Source:</strong> ' + s.source + '</div>'
  ).addTo(historicalLayerGroup);
});

export function wireCoordDisplay() {
  map.on('mousemove', function (e) {
    document.getElementById('coordDisplay').innerHTML =
      'LAT ' + e.latlng.lat.toFixed(5) + '°N<br>LON ' + e.latlng.lng.toFixed(5) + '°E';
  });
}
