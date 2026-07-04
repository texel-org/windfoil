// slug.js — build the Slug dual-band glyph atlas, for benchmarking against windfoil (bench/main.js).
//
// Slug casts two rays (horizontal + vertical), so a glyph needs two band sets: a HORIZONTAL set filed by y
// (for the +x ray) and a VERTICAL set filed by x (for the +y ray). We reuse the exact same row-band filer as
// windfoil (`bandPieces` from ../src/bands.js) for both — the horizontal set is the plain quads, and the
// vertical set is the quads rotated 90° into (y, −x), which turns "file/sort by x" back into bandPieces's
// "file by the 2nd coord, sort by the 1st". Storing the vertical curves pre-rotated lets slug.wgsl run one
// identical gather per ray. Unlike windfoil, curves are NOT split into monotone pieces — Slug's two-root
// solver handles a whole quadratic, which is part of what the benchmark is comparing.
//
// The atlas packs both band sets into ONE curve buffer + ONE row table (bandPieces appends to shared arrays,
// so all indices stay global), which means the Slug scene reuses windfoil's 4-binding pipeline
// (`createGlyphRenderer`) unchanged — only the instance stride (20 floats) and the shader differ.

import { bandPieces } from '../src/bands.js';
import { glyphQuads } from '../src/font.js';
import { loadShaderCode } from '../src/gpu.js';

const WGSL_URL = new URL('./slug.wgsl', import.meta.url);

// Slug bands its curves finer than windfoil's default: a dual-ray method root-solves every curve in the band
// (no cheap "far curve" path), so it prefers shorter bands. Pinning this keeps Slug's numbers comparable across
// windfoil's band-tuning changes — the two algorithms each band at their own optimum.
const SLUG_TARGET_PER_BAND = 6;

/** Load the Slug WGSL (shares the environment-agnostic loader from gpu.js). */
export function loadSlugShaderCode() {
  return loadShaderCode(WGSL_URL);
}

// Rotate a flat run of whole quads [x0,y0, cx,cy, x1,y1, ...] by −90° into (y, −x). A vertical +y ray on the
// original becomes a horizontal +x ray on the rotated curve, with winding orientation preserved (a rotation,
// not a reflection), so the two rays' signed coverages share a winding sign and combine directly in the shader.
function rotateQuads(quads) {
  const out = new Array(quads.length);
  for (let i = 0; i < quads.length; i += 2) {
    out[i] = quads[i + 1]; // x' = y
    out[i + 1] = -quads[i]; // y' = −x
  }
  return out;
}

/**
 * File one shape's whole quads into BOTH Slug band sets (appended to the shared `curves`/`rows`), returning the
 * two band headers the instance carries. Shared by the glyph atlas and the complex-shape scene.
 *
 * @returns { hRowBase, hBandCount, y0, invH, vRowBase, vBandCount, rotY0, invW, bbox }
 */
export function bandSlugShape(quads, bbox, curves, rows) {
  const [x0, y0, x1, y1] = bbox;
  // Horizontal bands: plain quads, filed by y over [y0, y1].
  const h = bandPieces(quads, y0, y1, curves, rows, SLUG_TARGET_PER_BAND);
  // Vertical bands: quads rotated to (y, −x), filed by the rotated y (= −x) over [−x1, −x0]. The returned
  // header's y0/invH are therefore the rotated y-origin (−hiX) and bands-per-x-unit.
  const v = bandPieces(rotateQuads(quads), -x1, -x0, curves, rows, SLUG_TARGET_PER_BAND);
  return {
    hRowBase: h.rowBase, hBandCount: h.bandCount, y0: h.y0, invH: h.invH,
    vRowBase: v.rowBase, vBandCount: v.bandCount, rotY0: v.y0, invW: v.invH,
    bbox,
  };
}

/**
 * Build the Slug dual-band atlas for the unique (non-space) characters of `text`. Returns the shared curve
 * buffer + row table the shader reads, a per-glyph lookup with BOTH band headers, and a few counts.
 */
export function buildSlugAtlas(font, text) {
  const chars = [...new Set([...text])].filter((ch) => ch !== ' ');
  const curves = [];
  const rows = [];
  const table = {};
  let curveTotal = 0;
  for (const ch of chars) {
    const g = glyphQuads(font, ch);
    if (!g) continue; // blank glyph
    curveTotal += g.quads.length / 6;
    table[ch] = { advance: g.advance, ...bandSlugShape(g.quads, g.bbox, curves, rows) };
  }
  const bandedPieces = curves.length / 6;
  return {
    curves: new Float32Array(curves),
    rows: new Uint32Array(rows),
    table,
    stats: {
      uniqueGlyphs: Object.keys(table).length,
      curves: curveTotal, // whole quads per glyph, summed over unique glyphs
      bandCount: rows.length / 5,
      bandedPieces, // after y- and x-duplication across both band sets
      duplication: curveTotal ? bandedPieces / (curveTotal * 2) : 1, // vs 2× (both band sets)
    },
  };
}

export const FLOATS_PER_SLUG_INSTANCE = 20;
