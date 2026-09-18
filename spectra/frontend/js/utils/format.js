const CARDINALS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];

export function bearingToCardinal(deg) {
  return CARDINALS[Math.round(deg / 22.5) % 16];
}

const VALIDITY_ICON = { valid: '✓', borderline: '◑', too_low: '↓', too_high: '↑', error: '✕', unavailable: '—' };
export function validityIcon(s) { return VALIDITY_ICON[s] || '?'; }

const VALIDITY_CHIP_CLASS = {
  valid: 'wv-valid', borderline: 'wv-borderline', too_low: 'wv-too-low',
  too_high: 'wv-too-high', error: 'wv-error', unavailable: 'wv-unavailable',
};
export function validityChipClass(s) { return VALIDITY_CHIP_CLASS[s] || 'wv-unavailable'; }

const RISK_CHIP_CLASS = { low: 'wr-low', medium: 'wr-medium', high: 'wr-high', unknown: 'wr-unknown' };
export function riskChipClass(l) { return RISK_CHIP_CLASS[l] || 'wr-unknown'; }

const VERDICT_ICON = { confirmed: '✓', unconfirmed: '✕', inconclusive: '△', unavailable: '—' };
export function verdictIcon(v) { return VERDICT_ICON[v] || '?'; }

const VERDICT_CHIP_CLASS = {
  confirmed: 'ov-confirmed', unconfirmed: 'ov-unconfirmed',
  inconclusive: 'ov-inconclusive', unavailable: 'ov-unavailable',
};
export function verdictChipClass(v) { return VERDICT_CHIP_CLASS[v] || 'ov-unavailable'; }

export function indexClass(val, thresholds) {
  if (val == null) return 'oi-unknown';
  const inRange = (v, lo, hi) => (lo == null || v >= lo) && (hi == null || v <= hi);
  if (inRange(val, thresholds.good[0], thresholds.good[1])) return 'oi-good';
  if (inRange(val, thresholds.warn[0], thresholds.warn[1])) return 'oi-warn';
  return 'oi-neutral';
}

export function compassSVG(direction) {
  const dw = ((direction || 0) + 180) % 360;
  return `<svg viewBox="0 0 42 42" width="42" height="42" style="flex-shrink:0;">
    <circle cx="21" cy="21" r="19" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
    <text x="21" y="7" text-anchor="middle" font-family="DM Mono,monospace" font-size="6" fill="rgba(255,255,255,0.2)">N</text>
    <text x="21" y="40" text-anchor="middle" font-family="DM Mono,monospace" font-size="6" fill="rgba(255,255,255,0.2)">S</text>
    <text x="4" y="24" text-anchor="middle" font-family="DM Mono,monospace" font-size="6" fill="rgba(255,255,255,0.2)">W</text>
    <text x="38" y="24" text-anchor="middle" font-family="DM Mono,monospace" font-size="6" fill="rgba(255,255,255,0.2)">E</text>
    <g transform="rotate(${dw},21,21)">
    <line x1="21" y1="21" x2="21" y2="8" stroke="var(--warn-2)" stroke-width="1.5"/>
    <polygon points="21,5 18.5,10 23.5,10" fill="var(--warn-2)"/>
    </g></svg>`;
}
