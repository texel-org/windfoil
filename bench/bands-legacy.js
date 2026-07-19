// bands-legacy.js — FROZEN copy of src/bands.js's bandPieces as of the windfoil atlas-v3 change, used ONLY by
// the Slug benchmark atlas (bench/slug.js). Slug's shader reads the original [start, count, density, xMin,
// xMax] stride-5 row layout and whole-piece banding; freezing it here keeps the Slug side of the comparison
// bit-identical while windfoil's own atlas format evolves.

// Aim for ~this many pieces per band (before y-overlap duplication inflates it). Coarser than you'd expect for
// a gather because the per-fragment cost of an extra piece is nearly free here: a piece fully left of the pixel
// ends the scan (early break), and one fully right adds a clamp/subtract with no root solve — so a longer band
// mostly adds cheap compares, while FEWER bands means a pixel's y-footprint spans fewer of them (fewer band
// setups) and each piece duplicates into fewer bands (smaller atlas). Raised from 6 → 10 after benchmarking
// (bench/): ~8–19% faster at small/medium sizes (where windfoil is fill-heavy and it matters most) with no
// regression at large sizes and a ~15% smaller atlas. Tuning is windfoil-specific — a dual-ray method (Slug)
// wants finer bands, since it root-solves every curve in the band with no cheap far path.
const TARGET_PER_BAND = 10;
const MAX_BANDS = 64;
// Bands with MORE than this many pieces are x-sorted (by hull x-max, descending) so the shader can stop at the
// first piece fully left of the pixel; shorter bands stay in curve order and take the plain linear scan.
// MUST equal SORT_MIN in windfoil.wgsl — the shader only breaks early on bands it assumes are sorted, so a mismatch
// is either a correctness bug (breaks an unsorted band) or lost perf (never breaks a sorted one).
// Tuning: was 8 (the median Lato band occupancy) on the theory that small bands don't repay a sort; re-measured
// on the zoom-ladder bench after the 1px pad + minification guard, 4 is mildly better (tiger −4–5%, glyphs
// neutral) — the early break pays for itself on nearly any band, and the sort itself is build-time-only.
export const BAND_SORT_MIN = 4;

function chooseBands(pieceCount, targetPerBand) {
  if (pieceCount <= targetPerBand) return 1;
  return Math.min(Math.ceil(pieceCount / targetPerBand), MAX_BANDS);
}

// The band index a y-value maps to: floor((y − y0)·invH) clamped to [0, R−1] (invH = 0 ⇒ the single band 0).
function bandIndex(y, y0, invH, R) {
  if (invH <= 0) return 0;
  return Math.min(Math.max(Math.floor((y - y0) * invH), 0), R - 1);
}

// Solve the monotone quadratic y-component a·t² + b·t + y0 = v on [0, 1] (f64 twin of the shader's
// mono_root): saturate to the endpoint if the piece starts past / never reaches v, else take the branch
// whose derivative sign matches `rising`.
function monoRootT(a, b, e0, e1, v, rising) {
  if (rising ? e0 >= v : e0 <= v) return 0;
  if (rising ? e1 <= v : e1 >= v) return 1;
  const c = e0 - v;
  if (Math.abs(a) < 1e-12 * Math.max(Math.abs(b), 1)) return Math.min(Math.max(-c / b, 0), 1);
  const disc = Math.max(b * b - 4 * a * c, 0);
  const q = -0.5 * (b + Math.sign(b || 1) * Math.sqrt(disc));
  const r1 = q / a, r2 = q !== 0 ? c / q : 0;
  const want = rising ? 1 : -1;
  const t = (2 * a * r1 + b) * want >= 0 ? r1 : r2;
  return Math.min(Math.max(t, 0), 1);
}

// Exact winding integral ∫∫_strip w dA of one band's pieces over its y-strip [b0, b1]. Each piece contributes
// ∫ (x(t) − x0)·y′(t) dt over the t-range where y(t) ∈ [b0, b1] — a quartic antiderivative, exact in f64.
// (The x reference x0 is immaterial for a closed contour — its net flux through the strip is zero — it only
// keeps magnitudes small.) Windows tile across bands, so duplicated pieces never double-count (same argument
// as the shader gather).
function bandWindingArea(pieces, bucket, x0, b0, b1) {
  let area = 0;
  for (const k of bucket) {
    const p = k * 6;
    const X0 = pieces[p],
      Y0 = pieces[p + 1],
      CX = pieces[p + 2],
      CY = pieces[p + 3],
      X1 = pieces[p + 4],
      Y1 = pieces[p + 5];
    const lo = Math.max(b0, Math.min(Y0, Y1));
    const hi = Math.min(b1, Math.max(Y0, Y1));
    if (hi <= lo) continue;
    const rising = Y1 >= Y0;
    const ay = Y0 - 2 * CY + Y1, by = 2 * (CY - Y0);
    const tA = monoRootT(ay, by, Y0, Y1, rising ? lo : hi, rising);
    const tB = monoRootT(ay, by, Y0, Y1, rising ? hi : lo, rising);
    if (tB <= tA) continue;
    const ax = X0 - 2 * CX + X1, bx = 2 * (CX - X0), cx = X0 - x0;
    // (ax·t² + bx·t + cx)·(2·ay·t + by), integrated term-by-term
    const c3 = 2 * ax * ay, c2 = ax * by + 2 * bx * ay, c1 = bx * by + 2 * cx * ay, c0 = cx * by;
    const F = (t) => ((c3 / 4 * t + c2 / 3) * t + c1 / 2) * t * t + c0 * t;
    area += F(tB) - F(tA);
  }
  return area;
}

// Bit-pun an f32 into a u32 so profile data rides in the (integer) row table.
const punBuf = new DataView(new ArrayBuffer(4));
function f32bits(v) {
  punBuf.setFloat32(0, Math.fround(v), true);
  return punBuf.getUint32(0, true);
}

/**
 * File a shape's monotone `pieces` (flat 6-float runs) into row bands over [y0, y1], appending the
 * band-duplicated pieces to `curveOut` and each band's [start, count, densityBits, xMinBits, xMaxBits] to
 * `rowOut`. Beyond start/count, each band carries three f32s (bit-punned into the integer table):
 *   • density — the band's exact winding integral divided by its height and x-hull width; the minification
 *     guard integrates this constant density over the pixel overlap instead of gathering curves.
 *   • xMin/xMax — the hull used to normalize and clip that density, preserving per-band letterform hints
 *     instead of smearing each strip across the whole ink box.
 * Returns { rowBase, bandCount, bandH, invH }; callers already know y0, so the shader header uses its slot for
 * bandH and turns repeated edge divisions into multiplications (rowBase is in row-quintuple units).
 */
export function bandPieces(pieces, y0, y1, curveOut, rowOut, targetPerBand = TARGET_PER_BAND) {
  const n = pieces.length / 6;
  const R = chooseBands(n, targetPerBand);
  const invH = R > 1 && y1 > y0 ? R / (y1 - y0) : 0;

  const buckets = Array.from({ length: R }, () => []);
  let xLeft = Infinity; // x reference for the winding areas (any constant works for closed contours;
  for (let k = 0; k < n; k++) { //     the leftmost point keeps the f32 magnitudes small)
    const yLo = Math.min(pieces[k * 6 + 1], pieces[k * 6 + 3], pieces[k * 6 + 5]);
    const yHi = Math.max(pieces[k * 6 + 1], pieces[k * 6 + 3], pieces[k * 6 + 5]);
    const lo = bandIndex(yLo, y0, invH, R);
    const hi = bandIndex(yHi, y0, invH, R);
    for (let b = lo; b <= hi; b++) buckets[b].push(k);
    xLeft = Math.min(xLeft, pieces[k * 6], pieces[k * 6 + 2], pieces[k * 6 + 4]);
  }

  const rowBase = rowOut.length / 5;
  const bandH = R > 1 ? (y1 - y0) / R : y1 - y0;
  // Pair the uploaded f32 scales after rounding, so index and edge math share the same band grid.
  const headerBandH = R > 1 ? Math.fround(1 / Math.fround(invH)) : Math.fround(bandH);
  const xMax = (k) => Math.max(pieces[k * 6], pieces[k * 6 + 2], pieces[k * 6 + 4]);
  const xMin = (k) => Math.min(pieces[k * 6], pieces[k * 6 + 2], pieces[k * 6 + 4]);
  for (let b = 0; b < R; b++) {
    const bucket = buckets[b];
    if (bucket.length > BAND_SORT_MIN) bucket.sort((a, c) => xMax(c) - xMax(a));
    const start = curveOut.length / 6;
    let bxMin = 3e38, bxMax = -3e38; // empty band ⇒ inverted far sentinels — the skip tests see "no ink"
    //                                  (finite, since WGSL implementations may assume floats are finite)
    for (const k of bucket) {
      for (let j = 0; j < 6; j++) curveOut.push(pieces[k * 6 + j]);
      bxMin = Math.min(bxMin, xMin(k));
      bxMax = Math.max(bxMax, xMax(k));
    }
    const area = n ? bandWindingArea(pieces, bucket, xLeft, y0 + b * bandH, y0 + (b + 1) * bandH) : 0;
    const density = area / Math.max(headerBandH * (bxMax - bxMin), 1e-30);
    rowOut.push(start, bucket.length, f32bits(density), f32bits(bxMin), f32bits(bxMax));
  }
  return { rowBase, bandCount: R, bandH: headerBandH, invH };
}
