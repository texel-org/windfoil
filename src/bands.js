// bands.js — file each shape's monotone pieces into horizontal row bands, the acceleration structure the
// shader gathers over. See docs/ALGORITHM.md §6 for why one band axis suffices and why band-clipped pieces
// never double-count.
//
// Each unique glyph is split into xy-monotone pieces once and filed into row bands once, so repeated
// letters share one banded copy. A piece is CLIPPED into every band its y-extent touches (the same curve
// restricted to that band, so per-band hulls stay tight), under the same `floor((y − y0)·invH)` mapping the
// shader selects with. Each band stores two segments, both sorted by hull x-min descending: F pieces (span
// the whole band — their fully-right prefix aggregates to a signed count) and E pieces (endpoint inside the
// band — their fully-right prefix aggregates to a span sum when the window covers the band), each with one
// packed conservative-f16 cull word per piece. Layout per band in the curve buffer:
//   metaF[fCount+1] · metaE[2·(eCount+1)] · piece data (3 vec2 per piece, F then E)
// with two vec4<u32> row-table headers per band: (start, fCount, eCount, wF|wE) and the guard profile
// (density, xMin, xMax, 0). MUST match windfoil.wgsl.

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
// Tuning: was 8 (the median Lato band occupancy) on the theory that small bands don't repay a sort; re-measured
// on the zoom-ladder bench after the 1px pad + minification guard, 4 is mildly better (tiger −4–5%, glyphs
// neutral) — the early break pays for itself on nearly any band, and the sort itself is build-time-only.
export const BAND_SORT_MIN = 4;

function chooseBands(pieceCount, targetPerBand) {
  if (pieceCount <= targetPerBand) return 1;
  // Small shapes (glyph-sized and below — the only ones the minification guard realistically fires on)
  // keep the fixed pieces-per-band rule, so the guard's banded-ink profile is unchanged.
  if (pieceCount <= 6 * targetPerBand) return Math.min(Math.ceil(pieceCount / targetPerBand), MAX_BANDS);
  // Large shapes: R ~ √(n/2). Per-fragment gather cost is (bands in the window)·setup + (pieces per
  // band)·scan; the window's band count grows with R while occupancy shrinks as n/R, and the square root
  // balances them (dense-art n≈360 → R 13, benchmarked best across the zoom ladder — bench/README.md). The
  // floor keeps mid-size shapes from dropping below their old granularity too abruptly.
  const sqrtR = Math.round(Math.sqrt(pieceCount / 2));
  const floor = Math.min(Math.ceil(pieceCount / targetPerBand), 12);
  return Math.max(1, Math.min(Math.max(sqrtR, floor), MAX_BANDS));
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

// ── f16 cull words ────────────────────────────────────────────────────────────────────────────────────
// Per-piece cull metadata is packed as two f16s in one 32-bit word (the shader unpacks with
// unpack2x16float). Culling only needs a CONSERVATIVE hull — lows round toward −∞, highs toward +∞ — so
// quantization can never reject a piece that matters; a piece inside the widened hull just falls through to
// the exact integral. Clamping to the finite f16 range keeps the packed word's f32 exponent field below
// all-ones, so no bit pattern ever forms an f32 NaN/Inf (which a JS-number round-trip would canonicalize).
if (typeof Float16Array === 'undefined') {
  // Every WebGPU-capable runtime this repo targets ships Float16Array (ES2025); fail loudly otherwise
  // rather than silently mis-packing cull words.
  throw new Error('windfoil atlas build needs Float16Array (ES2025)');
}
const f16buf = new Float16Array(1);
const f16u16 = new Uint16Array(f16buf.buffer);

// f32 → f16 bits, rounded toward +∞ (`up`) or −∞, clamped to ±65504.
function f16BitsDir(v, up) {
  const x = Math.max(-65504, Math.min(65504, v));
  f16buf[0] = x; // round-to-nearest first…
  const back = f16buf[0];
  if ((up && back < x) || (!up && back > x)) { // …then step one f16 ulp in the required direction
    const b = f16u16[0], s = b & 0x8000, m = b & 0x7fff;
    if (up) f16u16[0] = s ? (m === 0 ? 0x0001 : b - 1) : b + 1;
    else f16u16[0] = s ? b + 1 : (m === 0 ? 0x8001 : b - 1);
  }
  return f16u16[0];
}

// Two f16 bit-halves → one u32 (lo in the low half), or the f32 whose bits those are, as a JS number
// (exact for all finite f32 — the finite-f16 clamp above rules out NaN/Inf patterns).
function packBits(loBits, hiBits) {
  return ((hiBits << 16) | loBits) >>> 0;
}
function packHalves(loBits, hiBits) {
  punBuf.setUint32(0, packBits(loBits, hiBits), true);
  return punBuf.getFloat32(0, true);
}

// f64 evaluation of one quadratic Bézier component at t (numerically stable de Casteljau form).
function evalComp(p0, p1, p2, t) {
  const s = 1 - t;
  return s * (s * p0 + t * p1) + t * (s * p1 + t * p2);
}

// Restrict piece q (flat 6 floats) to t ∈ [u, v]: endpoints from evaluation, control from the polar form
// f(u, v). Computed in f64 and rounded to f32 on output — the exact same curve on a sub-interval, up to one
// rounding of the new points. The control is clamped into the endpoint box so the shader's monotone-hull
// invariant (q2 within the endpoint span) survives that rounding. t = 0 / t = 1 reuse the original endpoint
// values bitwise, so chains with neighboring pieces stay gap-free.
function clipPiece(q, u, v) {
  const fx0 = Math.fround(u === 0 ? q[0] : evalComp(q[0], q[2], q[4], u));
  const fy0 = Math.fround(u === 0 ? q[1] : evalComp(q[1], q[3], q[5], u));
  const fx1 = Math.fround(v === 1 ? q[4] : evalComp(q[0], q[2], q[4], v));
  const fy1 = Math.fround(v === 1 ? q[5] : evalComp(q[1], q[3], q[5], v));
  const cu = (1 - u) * (1 - v), cm = u * (1 - v) + v * (1 - u), cv = u * v;
  const cx = Math.min(
    Math.max(Math.fround(cu * q[0] + cm * q[2] + cv * q[4]), Math.min(fx0, fx1)),
    Math.max(fx0, fx1),
  );
  const cy = Math.min(
    Math.max(Math.fround(cu * q[1] + cm * q[3] + cv * q[5]), Math.min(fy0, fy1)),
    Math.max(fy0, fy1),
  );
  return [fx0, fy0, cx, cy, fx1, fy1];
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

  const buckets = Array.from({ length: R }, () => []); // original piece indices (guard density + hull)
  const subs = Array.from({ length: R }, () => []); // band-CLIPPED sub-pieces (what the shader gathers)
  // Sub-pieces are the original curve restricted to the band's y-range, extended OUTWARD past each interior
  // split by `pad` (a few f32 ulps at coordinate scale). The shader clamps every window to its band's edges,
  // whose rc-relative f32 arithmetic can wobble by ~2 ulps against the build-time f64 grid — the pad
  // guarantees each sub-piece still spans its whole window, so windows tile across bands with no sliver lost
  // and no double-count, exactly as with whole-piece duplication. The padded overhang lies outside every
  // window, so pad size never reaches the rendered output.
  const pad = (Math.abs(y0) + Math.abs(y1) + (y1 - y0) + 1) * 2e-6;
  const bandH64 = R > 1 ? (y1 - y0) / R : y1 - y0;
  let xLeft = Infinity; // x reference for the winding areas (any constant works for closed contours;
  for (let k = 0; k < n; k++) { //     the leftmost point keeps the f32 magnitudes small)
    const q = pieces.slice(k * 6, k * 6 + 6);
    const yLo = Math.min(q[1], q[3], q[5]);
    const yHi = Math.max(q[1], q[3], q[5]);
    const lo = bandIndex(yLo, y0, invH, R);
    const hi = bandIndex(yHi, y0, invH, R);
    for (let b = lo; b <= hi; b++) buckets[b].push(k);
    xLeft = Math.min(xLeft, q[0], q[2], q[4]);
    if (Math.fround(q[1]) === Math.fround(q[5])) continue; // zero y-span ⇒ zero area for every window: drop
    if (lo === hi) {
      subs[lo].push([q[0], q[1], q[2], q[3], q[4], q[5]]);
      continue;
    }
    const rising = q[5] >= q[1];
    const ay = q[1] - 2 * q[3] + q[5], by = 2 * (q[3] - q[1]);
    for (let b = lo; b <= hi; b++) {
      const eLo = y0 + b * bandH64 - pad;
      const eHi = y0 + (b + 1) * bandH64 + pad;
      const u = monoRootT(ay, by, q[1], q[5], rising ? eLo : eHi, rising);
      const v = monoRootT(ay, by, q[1], q[5], rising ? eHi : eLo, rising);
      if (v <= u) continue; // endpoint sits exactly on the band edge — nothing of the piece in here
      const s = clipPiece(q, u, v);
      if (s[1] !== s[5]) subs[b].push(s); // rounding-collapsed slivers contribute nothing: drop
    }
  }

  const rowBase = rowOut.length / 4; // in vec4<u32> units — each band takes TWO vec4 headers (see below)
  const bandH = bandH64;
  // Pair the uploaded f32 scales after rounding, so index and edge math share the same band grid.
  const headerBandH = R > 1 ? Math.fround(1 / Math.fround(invH)) : Math.fround(bandH);
  const xMax = (k) => Math.max(pieces[k * 6], pieces[k * 6 + 2], pieces[k * 6 + 4]);
  const xMin = (k) => Math.min(pieces[k * 6], pieces[k * 6 + 2], pieces[k * 6 + 4]);
  const hullMin = (s) => Math.min(s[0], s[4]);
  const hullMax = (s) => Math.max(s[0], s[4]);
  const padSafe = pad * 0.25; // > the shader's band-edge f32 wobble, < the clip pad — a safe F margin
  for (let b = 0; b < R; b++) {
    const bucket = buckets[b];
    const bLo = y0 + b * bandH64, bHi = y0 + (b + 1) * bandH64;
    // Split the band's clipped pieces into segments (docs: "F/E" = full-span / endpoint):
    //   F — spans the entire band (plus clip pad), so under ANY window ⊆ band it either straddles the box
    //       in x or contributes exactly ±(window height): far-right F pieces aggregate to a signed COUNT.
    //   E — has a real endpoint inside the band; keeps the general per-piece handling.
    const fList = [], eList = [];
    for (const s of subs[b]) {
      const sLo = Math.min(s[1], s[5]), sHi = Math.max(s[1], s[5]);
      if (sLo <= bLo - padSafe && sHi >= bHi + padSafe) fList.push(s);
      else eList.push(s);
    }
    fList.sort((a, c) => hullMin(c) - hullMin(a)); // x-min desc: the fully-right set is a PREFIX
    eList.sort((a, c) => hullMin(c) - hullMin(a)); // x-min desc for E too — same prefix trick, span sums
    let wF = 0, wE = 0;
    for (const s of fList) wF = Math.max(wF, hullMax(s) - hullMin(s));
    for (const s of eList) wE = Math.max(wE, hullMax(s) - hullMin(s));
    const start = curveOut.length / 2; // vec2 units: band record = [metaF | metaE | prefixE | piece data]
    // metaF: (packed f16 x-hull, signed fully-right count BEFORE this piece) + a terminator with the total,
    // so the shader's prefix jump lands on a valid count at every stop position.
    let count = 0;
    for (const s of fList) {
      curveOut.push(packHalves(f16BitsDir(hullMin(s), false), f16BitsDir(hullMax(s), true)), count);
      count += s[5] > s[1] ? 1 : -1;
    }
    curveOut.push(0, count);
    // metaE: TWO vec2s per piece — (packed f16 x-hull, P_i) and the exact stored endpoint y's (y1, y3).
    // The second word makes the fully-right cheap path one 8-byte cache-adjacent load, gives the y-cull
    // exact bounds, and carries P_i = Σ_{j<i} signed span inline: the fully-right E prefix of a band whose
    // window covers it contributes exactly sx·P_k (each such span lies inside the band, so no clamp binds;
    // the clip pad's overhang is ~1e-6 of a band and vanishes below the f32 floor). A terminator pair
    // carries P_total so the jump lands on a valid sum at every stop position.
    {
      let P = 0;
      for (const s of eList) {
        curveOut.push(
          packHalves(f16BitsDir(hullMin(s), false), f16BitsDir(hullMax(s), true)),
          Math.fround(P),
        );
        curveOut.push(s[1], s[5]);
        P += s[5] - s[1];
      }
      curveOut.push(0, Math.fround(P), 0, 0);
    }
    for (const s of fList) curveOut.push(s[0], s[1], s[2], s[3], s[4], s[5]);
    for (const s of eList) curveOut.push(s[0], s[1], s[2], s[3], s[4], s[5]);
    // Guard data (density + hull) stays derived from the ORIGINAL pieces so the approximate minification
    // profile is bit-identical to the pre-clipping atlas.
    let bxMin = 3e38, bxMax = -3e38; // empty band ⇒ inverted far sentinels — the skip tests see "no ink"
    //                                  (finite, since WGSL implementations may assume floats are finite)
    for (const k of bucket) {
      bxMin = Math.min(bxMin, xMin(k));
      bxMax = Math.max(bxMax, xMax(k));
    }
    const area = n ? bandWindingArea(pieces, bucket, xLeft, y0 + b * bandH, y0 + (b + 1) * bandH) : 0;
    const density = area / Math.max(headerBandH * (bxMax - bxMin), 1e-30);
    // Two vec4<u32> headers: H0 = (start, fCount, eCount, packed wF|wE) for the gather, H1 = guard profile.
    rowOut.push(start, fList.length, eList.length, packBits(f16BitsDir(wF, true), f16BitsDir(wE, true)));
    rowOut.push(f32bits(density), f32bits(bxMin), f32bits(bxMax), 0);
  }
  return { rowBase, bandCount: R, bandH: headerBandH, invH };
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
  let bandedPieces = 0; // stored (band-clipped) pieces: Σ per-band fCount + eCount
  for (let i = 0; i < rows.length; i += 8) bandedPieces += rows[i + 1] + rows[i + 2];
  return {
    curves: new Float32Array(curves),
    rows: new Uint32Array(rows),
    table,
    stats: {
      uniqueGlyphs: Object.keys(table).length,
      monotonePieces: monotoneTotal,
      bandCount: rows.length / 8,
      bandedPieces,
      duplication: monotoneTotal ? bandedPieces / monotoneTotal : 1,
    },
  };
}
