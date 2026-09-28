// Copy the selected facing edge chain of one region onto another region.
// Coordinates are copied verbatim in source-image pixels; the target arc is replaced.
window.HTH_REFERENCE_BOUNDARY = (() => {
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];

  function path(points, start, end, direction) {
    const indices = [start];
    while (indices.at(-1) !== end && indices.length <= points.length) {
      indices.push((indices.at(-1) + direction + points.length) % points.length);
    }
    return indices.at(-1) === end ? indices : null;
  }

  function containsEdge(indices, edge, direction, count) {
    const first = direction === 1 ? edge : (edge + 1) % count;
    return indices.slice(0, -1).includes(first);
  }

  function facingArcs(points, selectedEdge, transverseAxis, oppositeCenter) {
    const values = points.map(point => point[transverseAxis]);
    const minimum = Math.min(...values), maximum = Math.max(...values);
    if (minimum === maximum) throw Error('The selected region has no usable boundary span.');
    // A hand-traced side may be a few pixels inside the absolute extrema. Restrict
    // candidates to the outer band, then prefer the corners facing the other region.
    const tolerance = Math.max(2, (maximum - minimum) * 0.03);
    const starts = values.flatMap((value, index) => value <= minimum + tolerance ? [index] : []);
    const ends = values.flatMap((value, index) => value >= maximum - tolerance ? [index] : []);
    const facingAxis = 1 - transverseAxis, candidates = [];
    for (const start of starts) for (const end of ends) for (const direction of [1, -1]) {
      const indices = path(points, start, end, direction);
      if (!indices || !containsEdge(indices, selectedEdge, direction, points.length)) continue;
      const length = indices.slice(1).reduce((sum, index, step) => sum + distance(points[indices[step]], points[index]), 0);
      const facingDistance = Math.abs(points[start][facingAxis] - oppositeCenter[facingAxis]) + Math.abs(points[end][facingAxis] - oppositeCenter[facingAxis]);
      candidates.push({indices, start, end, direction, score:facingDistance + length * 0.01});
    }
    if (!candidates.length) throw Error('Could not identify a facing boundary from that edge. Select another edge.');
    return candidates.sort((a, b) => a.score - b.score);
  }

  const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const within = (a, b, p) => Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0]) && Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1]);
  function segmentsIntersect(a, b, c, d) {
    const abC = cross(a, b, c), abD = cross(a, b, d), cdA = cross(c, d, a), cdB = cross(c, d, b);
    if (abC === 0 && within(a, b, c) || abD === 0 && within(a, b, d) || cdA === 0 && within(c, d, a) || cdB === 0 && within(c, d, b)) return true;
    return (abC > 0) !== (abD > 0) && (cdA > 0) !== (cdB > 0);
  }

  function simplePolygon(points) {
    if (points.length < 3) return false;
    for (let i = 0; i < points.length; i++) {
      const a = points[i], b = points[(i + 1) % points.length];
      if (same(a, b)) return false;
      for (let j = i + 1; j < points.length; j++) {
        if (j === i + 1 || i === 0 && j === points.length - 1) continue;
        if (segmentsIntersect(a, b, points[j], points[(j + 1) % points.length])) return false;
      }
    }
    return true;
  }

  function mirror(source, sourceEdge, target, targetEdge) {
    if (!Array.isArray(source) || !Array.isArray(target) || source.length < 3 || target.length < 3) throw Error('Both regions need polygon boundaries.');
    if (!Number.isInteger(sourceEdge) || sourceEdge < 0 || sourceEdge >= source.length || !Number.isInteger(targetEdge) || targetEdge < 0 || targetEdge >= target.length) throw Error('Select a valid source edge and a target edge.');
    const center = points => [0, 1].map(axis => (Math.min(...points.map(point => point[axis])) + Math.max(...points.map(point => point[axis]))) / 2);
    const sourceCenter = center(source), targetCenter = center(target);
    const transverseAxis = Math.abs(targetCenter[1] - sourceCenter[1]) >= Math.abs(targetCenter[0] - sourceCenter[0]) ? 0 : 1;
    const sourceArcs = facingArcs(source, sourceEdge, transverseAxis, targetCenter);
    const targetArcs = facingArcs(target, targetEdge, transverseAxis, sourceCenter);
    let best = null;
    for (const sourceArc of sourceArcs) for (const targetArc of targetArcs) {
      const copied = sourceArc.indices.map(index => [...source[index]]);
      const targetStart = target[targetArc.start], targetEnd = target[targetArc.end];
      const forwardCost = distance(copied[0], targetStart) + distance(copied.at(-1), targetEnd);
      const reverseCost = distance(copied.at(-1), targetStart) + distance(copied[0], targetEnd);
      if (reverseCost < forwardCost) copied.reverse();
      const rest = path(target, targetArc.end, targetArc.start, targetArc.direction);
      const boundary = [...copied, ...rest.slice(1, -1).map(index => [...target[index]])];
      if (!simplePolygon(boundary)) continue;
      const score = sourceArc.score + targetArc.score + Math.min(forwardCost, reverseCost) * 0.01;
      if (!best || score < best.score) best = {boundary, copiedVertices:copied.length, score};
    }
    if (!best) throw Error('Mirroring would create a crossed or collapsed polygon. Choose another source or target edge.');
    return {boundary:best.boundary, copiedVertices:best.copiedVertices};
  }

  return {mirror};
})();
