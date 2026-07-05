// shapes.js — procedural vector shapes + a generic (font-free) atlas builder, for demos that aren't text.
//
// windfoil fills any 2D shape built from quadratic-Bézier contours, not just glyphs. bands.js already files
// arbitrary monotone pieces into row bands; buildGlyphAtlas just happens to source its outlines from a font.
// This module provides the same path for hand-built shapes (the soft-shadow canopy uses leaves), plus a small
// deterministic RNG and a leaf generator.

import { pushMonotonePieces } from './geometry.js';
import { bandPieces } from './bands.js';

// mulberry32 — a tiny deterministic PRNG so scenes are screenshot-stable (same seed → same canopy every run).
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Rotate a flat quad list [x0,y0,cx,cy,x1,y1, …] about the origin by `angle` (radians), in place-safe copy.
export function rotateQuads(quads, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const out = new Array(quads.length);
  for (let i = 0; i < quads.length; i += 2) {
    const x = quads[i], y = quads[i + 1];
    out[i] = x * c - y * s;
    out[i + 1] = x * s + y * c;
  }
  return out;
}

/**
 * A stylised leaf as a closed quadratic contour, pointing +y from base (0,0) to tip (0, ~length): four quads
 * (two per side) give a rounded base and a pointed tip. `rng` (0..1) adds per-leaf variety. Returned as a flat
 * quad list in the leaf's local space, already rotated by `angle`.
 */
export function makeLeaf({ rng = Math.random, length = 1, width = 0.42, angle = 0 } = {}) {
  const L = length * (0.85 + 0.3 * rng());
  const W = width * length * (0.8 + 0.4 * rng());
  const belly = 0.42 + 0.12 * rng();       // where the leaf is widest, along its length
  const baseBulge = 0.06 + 0.05 * rng();    // control height near the base (rounds the base)
  const tipTaper = 0.9 + 0.06 * rng();      // control height near the tip (sharpens the point)
  const skew = (rng() - 0.5) * 0.25;        // slight left/right asymmetry, like a real leaf

  const B = [0, 0];
  const T = [0, L];
  const Rm = [W, belly * L];
  const Lm = [-W * (1 - skew), belly * L];
  // right-lower, right-upper, left-upper, left-lower — control points bulge out then taper to the tip
  const quads = [
    ...B, 0.78 * W, baseBulge * L, ...Rm,
    ...Rm, 0.6 * W, tipTaper * L, ...T,
    ...T, -0.6 * W * (1 - skew), tipTaper * L, ...Lm,
    ...Lm, -0.78 * W * (1 - skew), baseBulge * L, ...B,
  ];
  return angle ? rotateQuads(quads, angle) : quads;
}

/** The xy bbox [x0,y0,x1,y1] of a flat quad/point list. */
export function quadsBounds(quads) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]); x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]); y1 = Math.max(y1, quads[i + 1]);
  }
  return [x0, y0, x1, y1];
}

/**
 * Build a banded atlas for a list of shapes (each a flat quad list), the font-free twin of buildGlyphAtlas.
 * Splits each shape into xy-monotone pieces, files it into row bands, and returns the shared curve + row
 * buffers the shader reads plus a per-shape table of band headers and bboxes.
 *
 * @param {number[][]} shapes  each element a flat quad list [x0,y0,cx,cy,x1,y1, …]
 * @returns {{ curves: Float32Array, rows: Uint32Array, table: {rowBase,bandCount,y0,invH,bbox:number[]}[] }}
 */
export function buildShapeAtlas(shapes) {
  const curves = [];
  const rows = [];
  const table = [];
  for (const quads of shapes) {
    const pieces = [];
    for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
    const [x0, y0, x1, y1] = quadsBounds(pieces);
    const header = bandPieces(pieces, y0, y1, curves, rows);
    table.push({ ...header, bbox: [x0, y0, x1, y1] });
  }
  return { curves: new Float32Array(curves), rows: new Uint32Array(rows), table };
}
