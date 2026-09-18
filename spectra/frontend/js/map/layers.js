import { map, spillLayerGroup, pipelineLayerGroup, historicalLayerGroup, driftLayerGroup, heatmapLayer, setHeatmapLayer } from './init.js';
import { layers, detections } from '../state.js';
import { bearingToCardinal } from '../utils/format.js';

export function toggleLayer(name) {
  layers[name] = !layers[name];
  document.getElementById('toggle-' + name).classList.toggle('on', layers[name]);
  if (name === 'pipelines') { layers[name] ? pipelineLayerGroup.addTo(map) : map.removeLayer(pipelineLayerGroup); }
  if (name === 'historical') { layers[name] ? historicalLayerGroup.addTo(map) : map.removeLayer(historicalLayerGroup); }
  if (name === 'spills') { layers[name] ? spillLayerGroup.addTo(map) : map.removeLayer(spillLayerGroup); }
  if (name === 'drift') { layers[name] ? driftLayerGroup.addTo(map) : map.removeLayer(driftLayerGroup); }
  if (name === 'heatmap') {
    if (layers[name]) { buildHeatmap(); }
    else if (heatmapLayer) { map.removeLayer(heatmapLayer); setHeatmapLayer(null); }
  }
}

// Lightweight canvas-free heatmap: stacked translucent circleMarkers.
export function buildHeatmap() {
  if (heatmapLayer) { map.removeLayer(heatmapLayer); setHeatmapLayer(null); }
  const pts = [];
  Object.values(detections).forEach(function (d) {
    if (d.centroid_lat && d.centroid_lon) {
      pts.push([d.centroid_lat, d.centroid_lon, Math.min(d.confidence / 100, 1)]);
    }
  });
  if (pts.length === 0) return;

  const heatGroup = L.layerGroup();
  pts.forEach(function (p) {
    const intensity = p[2];
    const color = intensity > 0.7 ? 'rgba(198,18,31,' : intensity > 0.5 ? 'rgba(199,123,42,' : 'rgba(200,190,150,';
    for (let r = 40; r >= 4; r -= 4) {
      const alpha = ((40 - r) / 40) * intensity * 0.25;
      L.circleMarker([p[0], p[1]], {
        radius: r, color: 'transparent', fillColor: color + alpha + ')', fillOpacity: 1, weight: 0,
      }).addTo(heatGroup);
    }
  });
  setHeatmapLayer(heatGroup);
  heatGroup.addTo(map);
}

export function addDriftArrow(d) {
  if (!d.drift_vector) return;
  let lat = d.centroid_lat, lon = d.centroid_lon;
  if ((lat == null || lon == null) && d.polygon && d.polygon.coordinates) {
    const co = d.polygon.coordinates[0];
    lat = co.reduce((s, c) => s + c[1], 0) / co.length;
    lon = co.reduce((s, c) => s + c[0], 0) / co.length;
  }
  if (lat == null || lon == null) return;

  const dv = d.drift_vector, dist = dv['24h_km'], br = dv.bearing_deg;
  if (!dist) return;

  const rad = (br * Math.PI) / 180;
  const dLat = (dist / 111.32) * Math.cos(rad);
  const dLon = (dist / (111.32 * Math.cos((lat * Math.PI) / 180))) * Math.sin(rad);
  const shaft = L.polyline([[lat, lon], [lat + dLat, lon + dLon]], { color: 'rgba(233,151,47,0.7)', weight: 1.5, opacity: 1, dashArray: '6 3' });
  const arrowIcon = L.divIcon({ className: '', html: '<div style="color:var(--warn-2);font-size:11px;transform:rotate(' + br + 'deg);transform-origin:center;line-height:1;">▲</div>', iconAnchor: [6, 6] });
  const arrowMarker = L.marker([lat + dLat, lon + dLon], { icon: arrowIcon });
  const popupContent = '<div class="drift-popup"><strong style="color:var(--warn-2);">Wind Drift (24h)</strong><br>Bearing: ' + br + '°<br>Distance: ' + dist + ' km<br>Speed: ' + dv.speed_ms + ' m/s<br><span style="color:var(--mid);font-size:9px;">3% wind drift · Coriolis corrected</span></div>';
  shaft.bindPopup(popupContent);
  arrowMarker.bindPopup(popupContent);
  shaft.addTo(driftLayerGroup);
  arrowMarker.addTo(driftLayerGroup);
}

export function addDetectionToMap(d) {
  if (!d.polygon) return;
  const conf = d.confidence;
  const colorHex = conf >= 70 ? '#e63946' : conf >= 50 ? '#e9972f' : '#b4b4a0';
  const ll = d.polygon.coordinates
    ? d.polygon.coordinates[0].map((c) => [c[1], c[0]])
    : [[d.centroid_lat - 0.05, d.centroid_lon - 0.05], [d.centroid_lat - 0.05, d.centroid_lon + 0.05], [d.centroid_lat + 0.05, d.centroid_lon + 0.05], [d.centroid_lat + 0.05, d.centroid_lon - 0.05]];
  const fillOpacity = (d.lookalike_passed === false) ? 0.05 : 0.22;
  const dashArray = (d.lookalike_passed === false) ? '5,4' : null;

  const lookalikeLine = d.lookalike_score != null
    ? '<div class="popup-row"><strong>Classifier:</strong> ' + d.lookalike_label + ' (' + Math.round(d.lookalike_score * 100) + '%)</div>' +
      '<div class="popup-row"><strong>Alert gate:</strong> ' + (d.lookalike_passed ? 'Passed' : 'Suppressed') + '</div>'
    : '';
  const windLine = d.wind_speed_ms != null
    ? '<div class="popup-row"><strong>Wind:</strong> ' + d.wind_speed_ms + ' m/s · ' + bearingToCardinal(d.wind_direction_deg) + ' (' + d.wind_direction_deg + '°)</div>' +
      '<div class="popup-row"><strong>SAR validity:</strong> ' + (d.sar_validity || '').replace(/_/g, ' ') + '</div>' +
      '<div class="popup-row"><strong>Drift 24h:</strong> ' + (d.drift_vector ? d.drift_vector['24h_km'] + ' km @ ' + d.drift_vector.bearing_deg + '°' : '—') + '</div>'
    : '';
  const opticalLine = d.optical_verdict && d.optical_verdict !== 'unavailable'
    ? '<div class="popup-row"><strong>S2 verdict:</strong> ' + d.optical_verdict.toUpperCase() + (d.optical_confidence != null ? ' (' + Math.round(d.optical_confidence * 100) + '%)' : '') + ' </div>' +
      (d.optical_osi != null ? '<div class="popup-row"><strong>OSI/SWIRI:</strong> ' + d.optical_osi + ' / ' + d.optical_swiri + '</div>' : '')
    : '';
  const aisLine = d.ais_top_suspect
    ? '<div class="popup-row"><strong>AIS suspect:</strong> ' + d.ais_top_suspect.name + ' (' + d.ais_top_suspect.vessel_type + ', ' + d.ais_top_suspect.distance_nm + ' nm, score ' + d.ais_top_suspect.suspicion_score + '/10)</div>'
    : '';

  L.polygon(ll, { color: colorHex, fillColor: colorHex, fillOpacity, weight: 1.5, opacity: 0.9, dashArray })
    .bindPopup(
      '<div class="popup-title">' + (d.lookalike_passed === false ? '◇ Suppressed' : '⚠ Spill Detected') + '</div>' +
      '<div class="popup-row"><strong>ID:</strong> ' + d.id + '</div>' +
      '<div class="popup-row"><strong>Confidence:</strong> ' + d.confidence + '%</div>' +
      '<div class="popup-row"><strong>Area:</strong> ' + d.area_km2 + ' km²</div>' +
      '<div class="popup-row"><strong>Pixels:</strong> ' + (d.spill_pixels ? d.spill_pixels.toLocaleString() : '—') + '</div>' +
      '<div class="popup-row"><strong>Detected:</strong> ' + new Date(d.detected_at).toLocaleString() + '</div>' +
      lookalikeLine + windLine + opticalLine + aisLine +
      '<div class="popup-row"><strong>Alert sent:</strong> ' + (d.alert_sent ? 'Yes' : 'No') + '</div>'
    ).addTo(spillLayerGroup);

  map.flyTo(ll[0], 10, { duration: 1.5 });
  L.circleMarker(ll[0], { radius: 16, color: colorHex, fillColor: colorHex, fillOpacity: 0.07, weight: 1, opacity: 0.5 }).addTo(spillLayerGroup);
  if (d.drift_vector) addDriftArrow(d);
  if (layers.heatmap) buildHeatmap();
}
