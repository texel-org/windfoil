// kernel-coverage.js — analytic soft-shadow coverage on the CPU: the disc (sun) kernel convolved with a vector
// silhouette at a PER-PIXEL-VARIABLE radius. This is the twin of the pluggable-kernel gather (see the
// kernels/pluggable-filters branch, docs/KERNELS.md) specialised to the sun's disc and to a spatially-varying
// radius — i.e. an ideal penumbra, sharp where the radius is small (contact) and soft where it grows (a high
// canopy). No GPU required, so it renders the preview and is checkable against an independent point sampler.
//
// A uniform disc of radius r has unit mass 1/(πr²). Working in kernel units u = (x−cx)/r, v = (y−cy)/r (unit
// disc), the horizontal cumulative is Φ(u,v) = (clamp(u,−w,w)+w)/π with w = √(1−v²); the disc-filtered winding
// number is the Green's-theorem boundary integral F = Σ_pieces ∫ Φ(u(t),v(t))·v′(t) dt (ALGORITHM.md §2 with the
// box clamp replaced by Φ). coverage = min(|F|, 1) under the nonzero rule. This is exactly the light a receiver
// point loses to the silhouette under a disc sun — the soft shadow.

const GL8 = {
  x: [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
    0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
  w: [0.10122853629037626, 0.22238103445337448, 0.31370664587788727, 0.36268378337836198,
    0.36268378337836198, 0.31370664587788727, 0.22238103445337448, 0.10122853629037626],
};

const INV_PI = 1 / Math.PI;

// Φ for the unit disc: fraction of the disc's mass to the left of u at height v.
function discPhi(u, v) {
  const w = Math.sqrt(Math.max(1 - v * v, 0));
  return (Math.min(Math.max(u, -w), w) + w) * INV_PI;
}
// Marginal CDF (the "fully to the right" weight): mass of the disc below height v.
function discYcdf(v) {
  const a = Math.min(Math.max(v, -1), 1);
  const w = Math.sqrt(Math.max(1 - a * a, 0));
  return 0.5 + (a * w + Math.asin(a)) * INV_PI;
}

// Solve the monotone quadratic y(t) = y0 + a1·t + a2·t² = target on [0,1], saturating to the endpoints.
function tAtY(a2, a1, y0, y1, target) {
  const rising = y1 >= y0;
  if (rising ? target <= y0 : target >= y0) return 0;
  if (rising ? target >= y1 : target <= y1) return 1;
  const c = y0 - target;
  if (Math.abs(a2) < 1e-12 * Math.max(Math.abs(a1), 1)) return Math.min(Math.max(-c / a1, 0), 1);
  const disc = Math.max(a1 * a1 - 4 * a2 * c, 0);
  const sq = Math.sqrt(disc);
  const q = -0.5 * (a1 + (a1 >= 0 ? sq : -sq));
  const r1 = q / a2, r2 = q !== 0 ? c / q : 0;
  // pick the root on the monotone branch (the one in [0,1])
  const cand = [r1, r2].filter((t) => t >= -1e-6 && t <= 1 + 1e-6);
  if (cand.length === 1) return Math.min(Math.max(cand[0], 0), 1);
  // both in range (rare near-degenerate) — pick by matching the rise direction
  return Math.min(Math.max((a1 < 0) === rising ? r1 : r2, 0), 1);
}

const N_SUB = 6; // composite GL slices — the disc rim's √ needs resolution; offline, so be generous

// One monotone piece [x0,y0,cx,cy,x1,y1]'s contribution to F, elliptical disc radii (rx,ry) about (cx0,cy0).
// (rx==ry is the round sun; rx<ry stretches the penumbra vertically for a grazing view.)
function piecePhiIntegral(p, cx0, cy0, rx, ry) {
  const x0 = p[0], y0 = p[1], cxp = p[2], cyp = p[3], x1 = p[4], y1 = p[5];
  const a2x = x0 - 2 * cxp + x1, a2y = y0 - 2 * cyp + y1;
  const a1x = 2 * (cxp - x0), a1y = 2 * (cyp - y0);
  // t-range where y(t) ∈ [cy0 − ry, cy0 + ry] (the disc's v-support), on the monotone piece
  const tA = tAtY(a2y, a1y, y0, y1, cy0 - ry);
  const tB = tAtY(a2y, a1y, y0, y1, cy0 + ry);
  let tlo = Math.min(tA, tB), thi = Math.max(tA, tB);
  if (thi - tlo < 1e-12) return 0;
  const invrx = 1 / rx, invry = 1 / ry;
  const xAt = (t) => (a2x * t + a1x) * t + x0;
  let xmin = Math.min(xAt(tlo), xAt(thi)), xmax = Math.max(xAt(tlo), xAt(thi));
  if (Math.abs(a2x) > 1e-20) {
    const te = -a1x / (2 * a2x);
    if (te > tlo && te < thi) { const xe = xAt(te); xmin = Math.min(xmin, xe); xmax = Math.max(xmax, xe); }
  }
  const yEval = (t) => (a2y * t + a1y) * t + y0;
  const dyEval = (t) => 2 * a2y * t + a1y;
  if (xmax <= cx0 - rx) return 0; // fully left → Φ = 0
  if (xmin >= cx0 + rx) {          // fully right → Φ = row mass → marginal CDF telescoped
    // Fast path: a piece fully spanning the disc's y-support contributes exactly its winding sign (±1) —
    // the interior backdrop — with no asin. This is the common case for foliage far to the right.
    if (y0 <= cy0 - ry && y1 >= cy0 + ry) return 1;
    if (y1 <= cy0 - ry && y0 >= cy0 + ry) return -1;
    return discYcdf((yEval(thi) - cy0) * invry) - discYcdf((yEval(tlo) - cy0) * invry);
  }
  let acc = 0;
  const seg = (thi - tlo) / N_SUB;
  for (let s = 0; s < N_SUB; s++) {
    const sa = tlo + s * seg, hm = 0.5 * seg, mid = sa + hm;
    for (let g = 0; g < 8; g++) {
      const t = mid + hm * GL8.x[g];
      const u = (xAt(t) - cx0) * invrx;
      const v = (yEval(t) - cy0) * invry;
      const vp = dyEval(t) * invry; // v′(t)
      acc += GL8.w[g] * hm * discPhi(u, v) * vp;
    }
  }
  return acc;
}

/**
 * A flat monotone-piece soup with a uniform grid for near-pixel culling. `pieces` is a Float64Array/Array of
 * 6-float runs [x0,y0,cx,cy,x1,y1]; the grid buckets piece indices by bounding box.
 */
export function buildPieceGrid(pieces, cell) {
  const n = pieces.length / 6;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const bb = new Float64Array(n * 4);
  for (let k = 0; k < n; k++) {
    const b = k * 6;
    const x0 = Math.min(pieces[b], pieces[b + 2], pieces[b + 4]);
    const x1 = Math.max(pieces[b], pieces[b + 2], pieces[b + 4]);
    const y0 = Math.min(pieces[b + 1], pieces[b + 3], pieces[b + 5]);
    const y1 = Math.max(pieces[b + 1], pieces[b + 3], pieces[b + 5]);
    bb[k * 4] = x0; bb[k * 4 + 1] = y0; bb[k * 4 + 2] = x1; bb[k * 4 + 3] = y1;
    minX = Math.min(minX, x0); minY = Math.min(minY, y0); maxX = Math.max(maxX, x1); maxY = Math.max(maxY, y1);
  }
  const cols = Math.max(1, Math.ceil((maxX - minX) / cell));
  const rows = Math.max(1, Math.ceil((maxY - minY) / cell));
  const cells = Array.from({ length: cols * rows }, () => []);
  const cx = (x) => Math.min(cols - 1, Math.max(0, Math.floor((x - minX) / cell)));
  const cy = (y) => Math.min(rows - 1, Math.max(0, Math.floor((y - minY) / cell)));
  for (let k = 0; k < n; k++) {
    for (let j = cy(bb[k * 4 + 1]); j <= cy(bb[k * 4 + 3]); j++) {
      for (let i = cx(bb[k * 4]); i <= cx(bb[k * 4 + 2]); i++) cells[j * cols + i].push(k);
    }
  }
  return { pieces, bb, minX, minY, cell, cols, rows, cells, cx, cy };
}

/**
 * Analytic disc-filtered coverage at (px,py) with disc radius r (shape units), gathering only pieces whose
 * bbox is within r of the point via the grid. Nonzero fill → min(|F|,1).
 */
export function discCoverage(grid, px, py, rx, ry = rx) {
  const { pieces, bb, cols, cells, cx, cy } = grid;
  // Scan the disc's y-band across the WHOLE row to the right: pieces fully right of the disc carry the
  // interior winding (the "everything to the right" dependency, ALGORITHM.md §6), so the x-query can't stop
  // at px+rx. Fully-left pieces (x < px−rx) contribute 0 and are skipped by starting at i0.
  const i0 = cx(px - rx), j0 = cy(py - ry), j1 = cy(py + ry);
  let F = 0;
  const seen = grid._seen || (grid._seen = new Int32Array(pieces.length / 6).fill(-1));
  const tag = (grid._tag = (grid._tag || 0) + 1);
  const scratch = [0, 0, 0, 0, 0, 0];
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i < cols; i++) {
      const bucket = cells[j * cols + i];
      for (let m = 0; m < bucket.length; m++) {
        const k = bucket[m];
        if (seen[k] === tag) continue;
        seen[k] = tag;
        if (bb[k * 4 + 1] > py + ry || bb[k * 4 + 3] < py - ry || bb[k * 4 + 2] < px - rx) continue;
        const b = k * 6;
        for (let q = 0; q < 6; q++) scratch[q] = pieces[b + q];
        F += piecePhiIntegral(scratch, px, py, rx, ry);
      }
    }
  }
  return Math.min(Math.abs(F), 1);
}

// ── Independent ground truth: stratified point sampling of the disc ─────────────────────────────────────────
// Winding at (x,y) by casting a +x ray and counting signed crossings of the monotone pieces.
function windingAt(pieces, x, y) {
  let w = 0;
  const n = pieces.length / 6;
  for (let k = 0; k < n; k++) {
    const b = k * 6;
    const y0 = pieces[b + 1], cy = pieces[b + 3], y1 = pieces[b + 5];
    if ((y0 > y) === (y1 > y) && (cy > y) === (y0 > y)) continue; // quick y reject (hull)
    const a2 = y0 - 2 * cy + y1, a1 = 2 * (cy - y0), a0 = y0 - y;
    let roots;
    if (Math.abs(a2) < 1e-12) {
      roots = Math.abs(a1) < 1e-20 ? [] : [-a0 / a1];
    } else {
      const d = a1 * a1 - 4 * a2 * a0;
      if (d < 0) continue;
      const s = Math.sqrt(d);
      roots = [(-a1 + s) / (2 * a2), (-a1 - s) / (2 * a2)];
    }
    const x0 = pieces[b], cx = pieces[b + 2], x1 = pieces[b + 4];
    for (const t of roots) {
      if (t < 0 || t > 1) continue;
      const xt = (x0 - 2 * cx + x1) * t * t + 2 * (cx - x0) * t + x0;
      if (xt <= x) continue;
      const dy = 2 * a2 * t + a1;
      if (dy > 0) w += 1; else if (dy < 0) w -= 1;
    }
  }
  return w;
}

/** Disc coverage by stratified sampling (independent of the boundary integral) — the validation ground truth.
 * Samples an ellipse of radii (rx,ry) so it matches the analytic elliptical disc. */
export function discCoverageSampled(pieces, px, py, rx, ry = rx, rings = 12, spokes = 24) {
  let inside = 0, total = 0;
  for (let ir = 0; ir < rings; ir++) {
    const rr = Math.sqrt((ir + 0.5) / rings); // equal-area rings in the unit disc, then scaled to (rx,ry)
    for (let is = 0; is < spokes; is++) {
      const ang = (is + 0.5) / spokes * Math.PI * 2 + ir * 0.61803;
      const x = px + rx * rr * Math.cos(ang), y = py + ry * rr * Math.sin(ang);
      if (windingAt(pieces, x, y) !== 0) inside++;
      total++;
    }
  }
  return inside / total;
}
