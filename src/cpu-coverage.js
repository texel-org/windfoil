// cpu-coverage.js — a faithful JS port of the windfoil fragment integral (src/windfoil.wgsl), evaluated on the
// CPU. It exists so the algorithm — including the soft-shadow footprint widening — can be exercised and a
// preview rendered on a machine with no WebGPU adapter (e.g. CI), and so the blur math can be unit-tested
// against a known closed form. It mirrors integrate_piece / integrate_band / integrate_face and the fs blur
// exactly; it is NOT the fast path (the GPU shader is), just an independent, checkable reference.

// mono_root — solve the monotone quadratic a2·t²+a1·t+a0 = v on [0,1], saturating to the endpoints.
function monoRoot(a2, a1, a0, e1, v, rising) {
  if (rising) {
    if (a0 >= v) return 0;
    if (e1 <= v) return 1;
  } else {
    if (a0 <= v) return 0;
    if (e1 >= v) return 1;
  }
  const c = a0 - v;
  if (Math.abs(a2) < 1e-12 * Math.max(Math.abs(a1), 1)) return Math.min(Math.max(-c / a1, 0), 1);
  const disc = Math.max(a1 * a1 - 4 * a2 * c, 0);
  const sq = Math.sqrt(disc);
  const qq = -0.5 * (a1 + (a1 >= 0 ? sq : -sq));
  const r1 = qq / a2;
  const r2 = qq !== 0 ? c / qq : 0;
  const t = ((a1 < 0) === rising) ? r1 : r2;
  return Math.min(Math.max(t, 0), 1);
}

// INSIDE zone: exact midpoint rule for ∫ (x(t)+hx)·y′(t) dt over [ta,tb].
function integrateInside(a2x, a2y, a1x, a1y, x0, ta, tb, hx) {
  if (tb <= ta) return 0;
  const tm = 0.5 * (ta + tb);
  const d = 0.5 * (tb - ta);
  const xMid = (a2x * tm + a1x) * tm + x0 + hx;
  const dx = 2 * a2x * tm + a1x;
  const dy = 2 * a2y * tm + a1y;
  return 2 * d * xMid * dy + (2 * d * d * d / 3) * (a2x * dy + 2 * a2y * dx);
}

// One xy-monotone piece over the y-window [lo,hi]; LEFT(0) / INSIDE(exact) / RIGHT(full width) split.
function integratePiece(q1, q2, q3, lo, hi, hx) {
  const a2x = q1[0] - 2 * q2[0] + q3[0], a2y = q1[1] - 2 * q2[1] + q3[1];
  const a1x = 2 * (q2[0] - q1[0]), a1y = 2 * (q2[1] - q1[1]);
  const yRising = q3[1] >= q1[1];
  // WGSL: t_lo solves for select(hi,lo,rising) = rising ? lo : hi; t_hi for the other edge.
  const tLo = monoRoot(a2y, a1y, q1[1], q3[1], yRising ? lo : hi, yRising);
  const tHi = monoRoot(a2y, a1y, q1[1], q3[1], yRising ? hi : lo, yRising);
  if (tHi <= tLo) return 0;
  const xRising = q3[0] >= q1[0];
  const tLeft = Math.min(Math.max(monoRoot(a2x, a1x, q1[0], q3[0], -hx, xRising), tLo), tHi);
  const tRight = Math.min(Math.max(monoRoot(a2x, a1x, q1[0], q3[0], hx, xRising), tLo), tHi);
  const t1 = xRising ? tLeft : tRight;
  const t2 = Math.max(xRising ? tRight : tLeft, t1);
  let acc = integrateInside(a2x, a2y, a1x, a1y, q1[0], t1, t2, hx);
  const ra = xRising ? t2 : tLo;
  const rb = xRising ? tHi : t1;
  if (rb > ra) {
    const tm = 0.5 * (ra + rb);
    acc += (rb - ra) * (2 * a2y * tm + a1y) * (2 * hx);
  }
  return acc;
}

function clippedDy(y1, y3, wlo, whi) {
  return Math.min(Math.max(y3, wlo), whi) - Math.min(Math.max(y1, wlo), whi);
}

const SORT_MIN = 4;

// Accumulate one row band's pieces over the rc-relative y-window [wlo,whi] (curves are absolute; subtract rc).
function integrateBand(curves, start, count, rcx, rcy, wlo, whi, sx) {
  let acc = 0;
  const hx = sx * 0.5;
  const sorted = count > SORT_MIN;
  const coordUlp = Math.max(Math.abs(rcx), Math.abs(rcy)) * 1.2e-7;
  for (let i = 0; i < count; i++) {
    const b = (start + i) * 6;
    const q1 = [curves[b] - rcx, curves[b + 1] - rcy];
    const q2 = [curves[b + 2] - rcx, curves[b + 3] - rcy];
    const q3 = [curves[b + 4] - rcx, curves[b + 5] - rcy];
    const xHullMax = Math.max(q1[0], q2[0], q3[0]);
    if (xHullMax <= -hx) { if (sorted) break; else continue; }
    const pyLo = Math.min(q1[1], q3[1]);
    const pyHi = Math.max(q1[1], q3[1]);
    const lo = Math.max(wlo, pyLo);
    const hi = Math.min(whi, pyHi);
    if (hi <= lo) continue;
    const xHullMin = Math.min(q1[0], q2[0], q3[0]);
    if (xHullMin >= hx) { acc += sx * clippedDy(q1[1], q3[1], wlo, whi); continue; }
    if (xHullMax - xHullMin + (pyHi - pyLo) <= coordUlp * 16) {
      const xm = Math.min(Math.max((q1[0] + q3[0]) * 0.5, -hx), hx) + hx;
      acc += xm * clippedDy(q1[1], q3[1], wlo, whi);
      continue;
    }
    acc += integratePiece(q1, q2, q3, lo, hi, hx);
  }
  return acc;
}

const ROW_STRIDE = 5;

// One shape's winding integral over the pixel box (rc ± s/2), gathered through the row bands its slab touches.
function integrateFace(curves, rows, band, rcx, rcy, sx, sy) {
  const rowBase = band.rowBase, R = band.bandCount, invH = band.invH;
  const sy2 = sy * 0.5;
  const dy0 = band.y0 - rcy;
  const bandIndex = (dy) => Math.min(Math.max(Math.floor(dy * invH), 0), R - 1);
  let ri0 = 0, ri1 = 0;
  if (invH > 0) {
    ri0 = bandIndex(-dy0 - sy2);
    ri1 = bandIndex(-dy0 + sy2);
  }
  let f = 0;
  for (let ri = ri0; ri <= ri1; ri++) {
    let wlo = -sy2, whi = sy2;
    if (invH > 0) {
      wlo = Math.max(wlo, dy0 + ri / invH);
      whi = Math.min(whi, dy0 + (ri + 1) / invH);
    }
    if (whi <= wlo) continue;
    const rIdx = (rowBase + ri) * ROW_STRIDE;
    f += integrateBand(curves, rows[rIdx], rows[rIdx + 1], rcx, rcy, wlo, whi, sx);
  }
  return f;
}

/**
 * Coverage at a point for one shape, with the same variable box-blur widening the shader applies. `s` is the
 * footprint (units per device px, [sx,sy]); `blur` = [basePx, gradX, gradY, maxPx] in the instance's units,
 * and `bboxMin` = [x0,y0] the gradient is measured from. Nonzero fill (the shadow demo's rule).
 *
 * @returns {number} coverage in [0,1]
 */
export function coverageAt(curves, rows, band, bboxMin, rcx, rcy, s, blur) {
  const [basePx, gradX, gradY, maxBlur] = blur;
  const blurPx = Math.min(Math.max(basePx + gradX * (rcx - bboxMin[0]) + gradY * (rcy - bboxMin[1]), 0), maxBlur);
  const sx = s[0] * (1 + blurPx);
  const sy = s[1] * (1 + blurPx);
  const f = integrateFace(curves, rows, band, rcx, rcy, sx, sy);
  return Math.min(Math.abs(f / (sx * sy)), 1);
}
