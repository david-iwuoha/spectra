import { map } from './init.js';
import { setPendingPolygon, getPendingPolygon } from '../state.js';
import { saveWatchZoneRequest, fetchWatchZones } from '../api.js';

const drawnItems = new L.FeatureGroup().addTo(map);
const drawControl = new L.Control.Draw({
  position: 'topright',
  draw: {
    polygon: { shapeOptions: { color: 'rgba(255,255,255,0.5)', fillColor: 'rgba(255,255,255,0.05)', fillOpacity: 1, weight: 1.5 }, showArea: true, allowIntersection: false },
    rectangle: { shapeOptions: { color: 'rgba(255,255,255,0.5)', fillColor: 'rgba(255,255,255,0.05)', fillOpacity: 1, weight: 1.5 } },
    circle: false, circlemarker: false, marker: false, polyline: false,
  },
  edit: { featureGroup: drawnItems, remove: true },
});
map.addControl(drawControl);

function priorityColor(priority) {
  return priority === 'high' ? 'rgba(198,18,31,0.6)' : priority === 'medium' ? 'rgba(199,123,42,0.6)' : 'rgba(150,150,150,0.4)';
}

export function wireDrawEvents() {
  map.on(L.Draw.Event.CREATED, function (e) {
    drawnItems.addLayer(e.layer);
    setPendingPolygon(e.layer.toGeoJSON().geometry);
    document.getElementById('watchZoneModal').classList.add('active');
  });
}

export function closeWatchZoneModal() {
  document.getElementById('watchZoneModal').classList.remove('active');
  setPendingPolygon(null);
  drawnItems.clearLayers();
}

export async function saveWatchZone() {
  const name = document.getElementById('wzName').value.trim();
  const client = document.getElementById('wzClient').value.trim();
  const priority = document.getElementById('wzPriority').value;
  const desc = document.getElementById('wzDescription').value.trim();
  if (!name || !client) { alert('Zone name and client name are required.'); return; }

  try {
    const data = await saveWatchZoneRequest({
      name, client_name: client, priority, description: desc, polygon_geojson: getPendingPolygon(),
    });
    const color = priorityColor(priority);
    drawnItems.eachLayer(function (l) {
      l.setStyle({ color, fillColor: color.replace('0.6', '0.06'), weight: 1.5, dashArray: '6,4' });
      l.bindPopup(
        '<div class="popup-title">' + name + '</div>' +
        '<div class="popup-row"><strong>Client:</strong> ' + client + '</div>' +
        '<div class="popup-row"><strong>Priority:</strong> ' + priority + '</div>' +
        '<div class="popup-row"><strong>Zone ID:</strong> ' + data.id + '</div>'
      ).openPopup();
    });
    closeWatchZoneModal();
    loadWatchZones();
  } catch (e) {
    alert('Failed to save watch zone. Is the backend running?');
  }
}

export async function loadWatchZones() {
  try {
    const data = await fetchWatchZones();
    data.watch_zones.forEach(function (z) {
      if (!z.polygon) return;
      const color = priorityColor(z.priority);
      L.geoJSON(z.polygon, { style: { color, fillColor: color.replace('0.6', '0.04'), weight: 1.5, dashArray: '6,4' } })
        .bindPopup(
          '<div class="popup-title">' + z.name + '</div>' +
          '<div class="popup-row"><strong>Client:</strong> ' + z.client_name + '</div>' +
          '<div class="popup-row"><strong>Priority:</strong> ' + z.priority + '</div>' +
          '<div class="popup-row"><strong>ID:</strong> ' + z.id + '</div>'
        ).addTo(map);
    });
  } catch (e) { /* backend not reachable yet */ }
}
