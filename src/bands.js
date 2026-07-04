// bands.js — file each glyph's monotone pieces into horizontal row bands, the acceleration structure the
// shader gathers over. See docs/ALGORITHM.md §6 for why one band axis suffices and why duplicated pieces
// never double-count.
//
// Each unique glyph is split into xy-monotone pieces once and filed into row bands once, so repeated
// letters share one banded copy. A piece is duplicated into every band its y-extent touches, under the same
// `floor((y − y0)·invH)` mapping the shader selects with. Long bands are sorted by hull x-max so the shader
// can stop at the first piece fully left of the pixel.

import { pushMonotonePieces } from './geometry.js';
import { glyphQuads } from './font.js';

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
// Tuning: measured over the Lato glyph set (a–z + ",."), post-duplication band occupancy is min 2 / median 8 /
// avg 8.3 / max 21, clustered at 6–9. 8 sits at the median — small common bands stay on the cheap scan, and
// only the heavier tail (~40% of bands, where the linear scan wastes the most) pays for the sort + early break.
export const BAND_SORT_MIN = 8;

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

// Exact winding integral ∫∫_strip w dA of one band's pieces over its y-strip [b0, b1], with x measured from
// the shape's left edge x0 (so the shader can spread it back over the bbox width). Each piece contributes
// ∫ (x(t) − x0)·y′(t) dt over the t-range where y(t) ∈ [b0, b1] — a quartic antiderivative, exact in f64.
// Windows tile across bands, so duplicated pieces never double-count (same argument as the shader gather).
function bandWindingArea(pieces, bucket, x0, b0, b1) {
  let area = 0;
  for (const k of bucket) {
    const p = k * 6;
    const X0 = pieces[p], Y0 = pieces[p + 1], CX = pieces[p + 2], CY = pieces[p + 3], X1 = pieces[p + 4], Y1 = pieces[p + 5];
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

// Bit-pun an f32 into a u32 so the area rides in the (integer) row table.
const punBuf = new DataView(new ArrayBuffer(4));
function f32bits(v) {
  punBuf.setFloat32(0, Math.fround(v), true);
  return punBuf.getUint32(0, true);
}

/**
 * File a shape's monotone `pieces` (flat 6-float runs) into row bands over [y0, y1], appending the
 * band-duplicated pieces to `curveOut` and each band's [start, count, areaBits, xMinBits, xMaxBits] to
 * `rowOut`. Beyond start/count, each band carries three f32s (bit-punned into the integer table):
 *   • area — the band strip's EXACT winding integral ∫∫_strip w dA, for the shader's minification guard
 *     (tiny glyphs render from this banded ink profile instead of gathering curves).
 *   • xMin/xMax — the hull of the band's pieces in x, for band-level skips: a pixel fully right of a band's
 *     ink adds nothing; one fully left of it adds nothing either when its slab covers the whole strip (a
 *     closed contour's net flux through a full strip is zero). One compare instead of a piece scan.
 * Returns the band header { rowBase, bandCount, y0, invH } the shader reads (rowBase in row-quintuple units).
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
    rowOut.push(start, bucket.length, f32bits(area), f32bits(bxMin), f32bits(bxMax));
  }
  return { rowBase, bandCount: R, y0, invH };
}

/**
 * Build the banded glyph atlas for the unique (non-space) characters of `text`: extract each glyph's
 * outline, split it into monotone pieces, and file it into row bands. Returns the curve buffer + row table
 * the shader reads, a per-glyph lookup table, and a few counts for reporting.
 */
export function buildGlyphAtlas(font, text) {
  const chars = [...new Set([...text])].filter((ch) => ch !== ' ');
  const curves = [];
  const rows = [];
  const table = {};
  let monotoneTotal = 0;
  for (const ch of chars) {
    const g = glyphQuads(font, ch);
    if (!g) continue; // blank glyph
    const pieces = [];
    for (let i = 0; i < g.quads.length; i += 6) pushMonotonePieces(g.quads.slice(i, i + 6), pieces);
    monotoneTotal += pieces.length / 6;
    const [, y0, , y1] = g.bbox;
    const header = bandPieces(pieces, y0, y1, curves, rows);
    table[ch] = { ...header, advance: g.advance, bbox: g.bbox };
  }
  const bandedPieces = curves.length / 6;
  return {
    curves: new Float32Array(curves),
    rows: new Uint32Array(rows),
    table,
    stats: {
      uniqueGlyphs: Object.keys(table).length,
      monotonePieces: monotoneTotal,
      bandCount: rows.length / 5,
      bandedPieces,
      duplication: monotoneTotal ? bandedPieces / monotoneTotal : 1,
    },
  };
}
