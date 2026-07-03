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

const TARGET_PER_BAND = 6; // aim for ~this many pieces per band (before y-overlap duplication inflates it)
const MAX_BANDS = 64;
// Bands with MORE than this many pieces are x-sorted (by hull x-max, descending) so the shader can stop at the
// first piece fully left of the pixel; shorter bands stay in curve order and take the plain linear scan.
// MUST equal SORT_MIN in windfoil.wgsl — the shader only breaks early on bands it assumes are sorted, so a mismatch
// is either a correctness bug (breaks an unsorted band) or lost perf (never breaks a sorted one).
// Tuning: measured over the Lato glyph set (a–z + ",."), post-duplication band occupancy is min 2 / median 8 /
// avg 8.3 / max 21, clustered at 6–9. 8 sits at the median — small common bands stay on the cheap scan, and
// only the heavier tail (~40% of bands, where the linear scan wastes the most) pays for the sort + early break.
export const BAND_SORT_MIN = 8;

function chooseBands(pieceCount) {
  if (pieceCount <= TARGET_PER_BAND) return 1;
  return Math.min(Math.ceil(pieceCount / TARGET_PER_BAND), MAX_BANDS);
}

// The band index a y-value maps to: floor((y − y0)·invH) clamped to [0, R−1] (invH = 0 ⇒ the single band 0).
function bandIndex(y, y0, invH, R) {
  if (invH <= 0) return 0;
  return Math.min(Math.max(Math.floor((y - y0) * invH), 0), R - 1);
}

/**
 * File a shape's monotone `pieces` (flat 6-float runs) into row bands over [y0, y1], appending the
 * band-duplicated pieces to `curveOut` and each band's [start, count] to `rowOut`. Returns the band header
 * { rowBase, bandCount, y0, invH } the shader reads (rowBase in [start,count]-pair units).
 */
export function bandPieces(pieces, y0, y1, curveOut, rowOut) {
  const n = pieces.length / 6;
  const R = chooseBands(n);
  const invH = R > 1 && y1 > y0 ? R / (y1 - y0) : 0;

  const buckets = Array.from({ length: R }, () => []);
  for (let k = 0; k < n; k++) {
    const yLo = Math.min(pieces[k * 6 + 1], pieces[k * 6 + 3], pieces[k * 6 + 5]);
    const yHi = Math.max(pieces[k * 6 + 1], pieces[k * 6 + 3], pieces[k * 6 + 5]);
    const lo = bandIndex(yLo, y0, invH, R);
    const hi = bandIndex(yHi, y0, invH, R);
    for (let b = lo; b <= hi; b++) buckets[b].push(k);
  }

  const rowBase = rowOut.length / 2;
  const xMax = (k) => Math.max(pieces[k * 6], pieces[k * 6 + 2], pieces[k * 6 + 4]);
  for (let b = 0; b < R; b++) {
    const bucket = buckets[b];
    if (bucket.length > BAND_SORT_MIN) bucket.sort((a, c) => xMax(c) - xMax(a));
    const start = curveOut.length / 6;
    for (const k of bucket) {
      for (let j = 0; j < 6; j++) curveOut.push(pieces[k * 6 + j]);
    }
    rowOut.push(start, bucket.length);
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
      bandCount: rows.length / 2,
      bandedPieces,
      duplication: monotoneTotal ? bandedPieces / monotoneTotal : 1,
    },
  };
}
