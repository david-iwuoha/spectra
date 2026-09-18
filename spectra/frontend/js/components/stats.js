export function updateStats(d) {
  const count = parseInt(document.getElementById('statDetections').textContent, 10) + 1;
  const area = parseFloat(document.getElementById('statArea').textContent) + (d.area_km2 || 0);
  document.getElementById('statDetections').textContent = count;
  document.getElementById('statArea').textContent = area.toFixed(2);
}
