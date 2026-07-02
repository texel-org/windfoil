// geometry — split quadratic Béziers into xy-monotone pieces.
//
// The per-piece closed form (docs/ALGORITHM.md) needs each piece monotone in both x and y, so that each box
// edge is crossed at most once and the y-window solve has a single branch. A quadratic has at most one
// interior extremum per axis, so splitting there yields 1–3 monotone pieces; bands.js then files them.

// Parameter of the interior extremum of the quadratic component (p0, p1, p2), if strictly inside (0, 1).
// Strict compares, no epsilon: an extremum exactly on an endpoint needs no split; a near-endpoint one still
// splits (and a floating-point-collapsed split simply yields a zero-length piece the shader skips).
function extremumT(p0, p1, p2) {
  const a = p0 - 2 * p1 + p2;
  if (a === 0) return null; // a straight component (incl. every line, whose control is the midpoint)
  const t = (p0 - p1) / a;
  return t > 0 && t < 1 ? t : null;
}

// de Casteljau subdivision of a 6-float quad [x0,y0,cx,cy,x1,y1] at t → [left, right]. The shared midpoint
// is the identical float on both sides, so consecutive pieces chain with no gap.
function subdivide(q, t) {
  const lerp = (a, b) => a + (b - a) * t;
  const x01 = lerp(q[0], q[2]), y01 = lerp(q[1], q[3]);
  const x12 = lerp(q[2], q[4]), y12 = lerp(q[3], q[5]);
  const xm = lerp(x01, x12), ym = lerp(y01, y12);
  return [
    [q[0], q[1], x01, y01, xm, ym],
    [xm, ym, x12, y12, q[4], q[5]],
  ];
}

// Append the xy-monotone pieces of one quad (1–3 of them, in curve order) to `out` as flat 6-float runs.
export function pushMonotonePieces(q, out) {
  const tx = extremumT(q[0], q[2], q[4]);
  const ty = extremumT(q[1], q[3], q[5]);
  let first = null, second = null;
  if (tx !== null && ty !== null) {
    first = Math.min(tx, ty);
    second = Math.max(tx, ty);
  } else {
    first = tx !== null ? tx : ty;
  }
  let rest = q;
  let consumed = 0;
  for (const t of [first, second]) {
    if (t === null) continue;
    // re-map the global split param into the remaining piece; skip one that collapses onto an endpoint
    const denom = 1 - consumed;
    const local = denom > 0 ? (t - consumed) / denom : 1;
    if (!(local > 0 && local < 1)) continue;
    const [l, r] = subdivide(rest, local);
    out.push(...l);
    rest = r;
    consumed = t;
  }
  out.push(...rest);
}
