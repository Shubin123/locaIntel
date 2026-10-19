/* Motion display uses image coordinates, never inferred compass directions or speed. */
function validVector(d) {
  const m = d.motion;
  return m?.state === 'moving' && Array.isArray(m.vector) && m.vector.length === 2 &&
    m.vector.every(Number.isFinite) && Math.hypot(...m.vector) > 0 &&
    Number.isFinite(m.gap_seconds) && m.gap_seconds > 0 && (m.gap_seconds <= 2 || (m.evidence_kind === 'three_frame_appearance' && m.evidence?.frames >= 3 && m.evidence?.background_stable && m.gap_seconds <= 15));
}
function vectorGeometry(box, vector, width, height, gain) {
  if (![...box, ...vector, width, height, gain].every(Number.isFinite) || width <= 0 || height <= 0 || gain <= 0) return null;
  const end = [(box[0] + box[2]) * width / 2, (box[1] + box[3]) * height / 2];
  const delta = [vector[0] * width * gain, vector[1] * height * gain];
  const magnitude = Math.hypot(...delta);
  if (magnitude < .01) return null;
  let scale = Math.min(1, Math.min(width, height) * .3 / magnitude);
  for (let axis = 0; axis < 2; axis++) {
    const bound = axis === 0 ? width : height;
    if (delta[axis] > 0) scale = Math.min(scale, Math.max(0, end[axis] - 3) / delta[axis]);
    if (delta[axis] < 0) scale = Math.min(scale, Math.max(0, bound - 3 - end[axis]) / -delta[axis]);
  }
  const tail = [end[0] - delta[0] * scale, end[1] - delta[1] * scale];
  const length = Math.hypot(end[0] - tail[0], end[1] - tail[1]);
  if (length < .5) return null;
  const unit = [(end[0] - tail[0]) / length, (end[1] - tail[1]) / length];
  const head = Math.min(12, Math.max(2, length * .35));
  const base = [end[0] - unit[0] * head, end[1] - unit[1] * head];
  const wing = head * .5;
  return { end, tail, head: [end, [base[0] - unit[1] * wing, base[1] + unit[0] * wing], [base[0] + unit[1] * wing, base[1] - unit[0] * wing]], clipped: scale < .999 };
}
function screenDirection(vector, width = 640, height = 400) {
  const angle = (Math.atan2(vector[1] * height, vector[0] * width) * 180 / Math.PI + 360) % 360;
  return ['→ right', '↘ down-right', '↓ down', '↙ down-left', '← left', '↖ up-left', '↑ up', '↗ up-right'][Math.floor((angle + 22.5) / 45) % 8];
}
function motionDescription(d) {
  if (!validVector(d)) return d.motion?.state === 'stationary' ? (d.motion?.evidence_kind === 'three_frame_appearance' ? 'Persistent position across three distinct frames' : 'Stationary in measured pair') : 'Unknown · no reliable vector';
  const m = d.motion, displacement = m.evidence?.displacement_px;
  const label = m.direction ? ({right:'→ right','down-right':'↘ down-right',down:'↓ down','down-left':'↙ down-left',left:'← left','up-left':'↖ up-left',up:'↑ up','up-right':'↗ up-right'}[m.direction] || m.direction) : screenDirection(m.vector);
  return `${label} on image · ${Number.isFinite(displacement) ? displacement.toFixed(1) + ' analysis px / ' : ''}${m.gap_seconds.toFixed(2)}s${m.heading != null ? ' · ' + m.heading + '° calibrated' : ''}`;
}
function motionExplanation(c, measured) {
  const motion = c.analysis?.motion, gap = motion?.gap_seconds;
  if (measured) return `${measured} measured motion ${measured === 1 ? 'vector' : 'vectors'}. Arrowheads point to the vehicle’s current position; dots mark the enlarged tail. Display scale: ${document.getElementById('vector-scale').value}×.${document.getElementById('vector-overlay').checked ? '' : ' Arrows are hidden; enable Motion arrows to see them.'}`;
  if (!(c.detections || []).some(d => ['car', 'truck', 'bus', 'motorcycle', 'bicycle'].includes(d.label))) return 'No vehicles detected in this frame. No motion vectors to display.';
  const reasons = {
    camera_moved: 'The camera moved between frames.',
    camera_geometry_changed: 'The camera view or image dimensions changed.',
    duplicate_frame: 'Both frames are identical; a repeated image cannot prove a vehicle is stationary.',
    insufficient_background_features: 'Too few background features to validate motion.',
    unstable_background: 'Background motion could not be separated reliably.',
    no_vehicle_pair: 'No matching vehicle pair was available.'
  };
  if (motion?.status === 'analyzed') return 'No reliable moving vectors in this frame pair. Blue boxes have stationary evidence; dashed amber boxes remain unknown.';
  const reason = reasons[motion?.status] || (Number.isFinite(gap) && gap > 2 ? `These frames are ${gap.toFixed(1)} seconds apart; movement needs a gap of 2 seconds or less.` : 'DriveBC stills do not provide a trustworthy short-gap frame pair.');
  return `Motion unavailable. ${reason} Vehicle boxes and heat are detections and sampled positions, not motion.`;
}
function renderVectorLayer(picture, c, detail, width, height, focusedVehicle = null) {
  picture.querySelector('.motion-vector-layer')?.remove();
  if (!(detail || document.getElementById('boxes').checked) || (detail && !document.getElementById('vector-overlay').checked)) return;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('motion-vector-layer'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Measured image-space vehicle displacement arrows');
  const add = (parent, tag, attrs) => { const element = document.createElementNS(svg.namespaceURI, tag); for (const [k, v] of Object.entries(attrs)) element.setAttribute(k, v); parent.append(element); return element; };
  for (const [i, d] of (c.detections || []).entries()) {
    if (!validVector(d) || (detail && focusedVehicle != null && focusedVehicle !== i)) continue;
    const geometry = vectorGeometry(d.box, d.motion.vector, width, height, detail ? Number(document.getElementById('vector-scale').value) : 8);
    if (!geometry) continue;
    const group = add(svg, 'g', { class: 'measured-vector', 'data-vehicle': i + 1 });
    const title = add(group, 'title', {}); title.textContent = `Vehicle #${i + 1}: ${motionDescription(d)}${geometry.clipped ? '; arrow shortened to fit image' : ''}`;
    const line = { x1: geometry.tail[0], y1: geometry.tail[1], x2: geometry.end[0], y2: geometry.end[1], 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' };
    add(group, 'line', { ...line, stroke: '#061012', 'stroke-width': 8 });
    add(group, 'line', { ...line, stroke: '#c5ff72', 'stroke-width': 4 });
    add(group, 'polygon', { points: geometry.head.map(p => p.join(',')).join(' '), fill: '#c5ff72', stroke: '#061012', 'stroke-width': 1.5, 'vector-effect': 'non-scaling-stroke' });
    add(group, 'circle', { cx: geometry.tail[0], cy: geometry.tail[1], r: Math.max(2, width / 200), fill: '#fff', stroke: '#061012', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke' });
  }
  picture.append(svg);
}
