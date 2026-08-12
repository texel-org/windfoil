// shapes.js — the shared shape library for the tools: flat quadratics in pixel coordinates.
//
// Every tool here speaks one geometry format — a flat array [x0,y0, cx,cy, x1,y1, ...] of quadratic Béziers,
// with a straight segment written as a quad whose control point is its midpoint. That is exactly what the
// shader consumes (src/geometry.js), what the box-filter reference ray-casts, what a 2D canvas replays as
// quadraticCurveTo, and what an SVG `d` string writes out — so one list of numbers feeds every renderer with
// no conversion anywhere, and the comparisons are of rasterisers, not of geometry pipelines.
//
// `buildShapes` is the validation dataset (tools/validate/) and doubles as the scene library for the
// comparison demo (tools/comparison.js). It is authored in a fixed CELL×CELL cell; `scaleQuads` takes it to
// any render size, which keeps the validation numbers stable no matter what size a demo asks for.

import { glyphQuads } from '../../src/font.js';

/** The cell `buildShapes` authors its coordinates in. */
export const CELL = 128;

export const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';

// ── primitives ──────────────────────────────────────────────────────────────────────────────────────────
/** A straight segment as a quad (control point at the midpoint — exact, not an approximation). */
export function line(x0, y0, x1, y1) {
  return [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1];
}

/** A closed contour through [x, y] points. */
export function polygon(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) out.push(...line(...pts[i], ...pts[(i + 1) % pts.length]));
  return out;
}

/** Axis-aligned rectangle as a closed contour. dir = +1 or −1 flips the traversal, flipping its winding sign. */
export function rect(x0, y0, x1, y1, dir = 1) {
  const cs = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return polygon(dir >= 0 ? cs : cs.slice().reverse());
}

export function rotate(pts, deg, cx = CELL / 2, cy = CELL / 2) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]);
}

/** A circle as `n` exact-tangent quadratic arcs. */
export function circle(cx, cy, r, n = 8) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI, am = (a0 + a1) / 2;
    const k = 1 / Math.cos((a1 - a0) / 2);
    out.push(
      cx + r * Math.cos(a0), cy + r * Math.sin(a0),
      cx + r * k * Math.cos(am), cy + r * k * Math.sin(am),
      cx + r * Math.cos(a1), cy + r * Math.sin(a1),
    );
  }
  return out;
}

/** The vertices of a {points/step} star polygon (step > 1 self-intersects). */
export function starPts(cx, cy, r, points, step) {
  const p = [];
  for (let k = 0; k < points; k++) {
    const a = -Math.PI / 2 + ((k * step) % points) * (2 * Math.PI / points);
    p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return p;
}

/** Uniformly scale a flat quad list about the origin (cell coordinates → render pixels). */
export const scaleQuads = (quads, k) => (k === 1 ? quads.slice() : quads.map((v) => v * k));

/** Translate a flat quad list. Used to set a shape's sub-pixel phase against the sample grid. */
export const translateQuads = (quads, dx, dy = dx) =>
  (dx === 0 && dy === 0 ? quads.slice() : quads.map((v, i) => v + (i % 2 === 0 ? dx : dy)));

/** Ink bounds [x0, y0, x1, y1] of a flat list of interleaved x, y coordinates. */
export function bboxOf(flat) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < flat.length; i += 2) {
    if (flat[i] < x0) x0 = flat[i];
    if (flat[i] > x1) x1 = flat[i];
    if (flat[i + 1] < y0) y0 = flat[i + 1];
    if (flat[i + 1] > y1) y1 = flat[i + 1];
  }
  return [x0, y0, x1, y1];
}

// ── composite stress shapes ─────────────────────────────────────────────────────────────────────────────
// The line-based shapes come in two forms of the SAME geometry: `quads` (the outline rectangle of each
// line, which ours and the box reference fill) and `segments` (flat [x0,y0,x1,y1,w] centerlines, which the
// stroked variants hand to the canvas as stroke() + lineWidth). A butt-capped stroked segment IS the
// rectangle, mathematically — so a stroked variant measures the host's stroke pipeline (including any thin-
// stroke/hairline special case) against the identical exact shape.

// A ladder of vertical bars: 4px wide down to a barely-visible hairline, each half the width of the last.
// Each bar gets a different sub-pixel phase (the i·0.37 term) so edges straddle pixel boundaries instead of
// snapping to the grid, where every rasterizer is trivially exact.
export function hairlines(n = 6) {
  const quads = [], segments = [];
  for (let i = 0, w = 4; i < n; i++, w /= 2) {
    const x = 16 + i * 18 + i * 0.37;
    quads.push(...polygon([[x, 14], [x + w, 14], [x + w, 114], [x, 114]]));
    segments.push(x + w / 2, 14, x + w / 2, 114, w);
  }
  return { quads, segments };
}

// n thin rectangles radiating from a hub, one per angle — a bicycle wheel. The spokes start at r0 so they
// stay disjoint (nonzero winding stays 1 everywhere; no fold-limit noise in the comparison).
export function spokes(cx, cy, r0, r1, n, w) {
  const quads = [], segments = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI;
    const dx = Math.cos(a), dy = Math.sin(a), px = -dy * (w / 2), py = dx * (w / 2);
    quads.push(...polygon([
      [cx + dx * r0 + px, cy + dy * r0 + py],
      [cx + dx * r1 + px, cy + dy * r1 + py],
      [cx + dx * r1 - px, cy + dy * r1 - py],
      [cx + dx * r0 - px, cy + dy * r0 - py],
    ]));
    segments.push(cx + dx * r0, cy + dy * r0, cx + dx * r1, cy + dy * r1, w);
  }
  return { quads, segments };
}

// A +1/−1 picket fence of k sub-pixel bars (all dyadic widths, so the geometry is float-exact). Every bar is
// filled under nonzero (true coverage ≈ 1 across the block), but a 1px footprint spans many opposite-sign
// bars, so the averaged winding → 0 and the fold fades toward black — the minification regime of
// tools/failure.js at native scale.
export function fence(k = 256) {
  const x0 = 20, y0 = 20, x1 = 108, y1 = 108, bw = (x1 - x0) / k;
  const out = [];
  for (let i = 0; i < k; i++) out.push(...rect(x0 + i * bw, y0, x0 + (i + 1) * bw, y1, i % 2 ? -1 : 1));
  return out;
}

/**
 * One glyph's outline, scaled to fit centred in a `cell`×`cell` box with `pad` px of margin. Font units are
 * Y-down with the baseline at 0 (see src/font.js), which is already the image convention, so the fit is a
 * uniform scale plus a translate — no flip.
 */
export function glyphShape(font, ch, cell = CELL, pad = cell * (14 / CELL)) {
  const g = glyphQuads(font, ch);
  if (!g) throw new Error(`glyph '${ch}' has no outline in this font (blank or missing)`);
  const [x0, y0, x1, y1] = g.bbox;
  const gw = x1 - x0, gh = y1 - y0, box = cell - 2 * pad;
  const k = Math.min(box / gw, box / gh);
  const ox = pad + (box - gw * k) / 2 - x0 * k, oy = pad + (box - gh * k) / 2 - y0 * k;
  return g.quads.map((v, i) => (i % 2 === 0 ? ox + v * k : oy + v * k));
}

/**
 * The shape dataset, authored in the CELL×CELL cell: the synthetic stress shapes, the winding-fold failure
 * cases, then every lowercase letter of the given font. Entries are { label, quads, evenodd?, segments?,
 * fold? }:
 *   segments — the canvas reference STROKES these [x0,y0,x1,y1,w] centerlines instead of filling the quads
 *              (ours and box still fill the quads, the identical shape);
 *   fold     — a documented winding-fold limit (tools/failure.js, docs/ALGORITHM.md §4/§8): 'ours vs box'
 *              is EXPECTED to deviate here, so boots report these separately from the common shapes. The
 *              self-intersecting stars stay in the common set — their sliver deviation is shared by every
 *              single-sample renderer, not a true failure.
 */
export function buildShapes(font) {
  const hl = hairlines(), thin = spokes(64, 64, 12, 58, 24, 0.75), thick = spokes(64, 64, 14, 58, 24, 2.5);
  return [
    { label: 'rotated square 30°', quads: polygon(rotate([[28, 28], [100, 28], [100, 100], [28, 100]], 30)) },
    { label: 'thin diagonal sliver', quads: polygon(rotate([[12, 63.5], [116, 63.5], [116, 64.5], [12, 64.5]], 27)) },
    { label: 'hairlines 4..0.125px', quads: hl.quads }, // vertical bars, each half the width of the last
    { label: 'hairlines (stroked)', quads: hl.quads, segments: hl.segments },
    { label: 'spokes 24 x 0.75px', quads: thin.quads }, // sub-pixel widths at 15° steps
    { label: 'spokes 0.75px (stroked)', quads: thin.quads, segments: thin.segments }, // sub-1px stroke → hairline special case
    { label: 'spokes 24 x 2.5px', quads: thick.quads }, // same wheel, multi-pixel widths
    { label: 'spokes 2.5px (stroked)', quads: thick.quads, segments: thick.segments },
    { label: 'circle r=44', quads: circle(64, 64, 44, 64) }, // 64 arcs: smooth enough that curve flattening is negligible
    { label: 'star {5/2} nonzero', quads: polygon(starPts(64, 64, 52, 5, 2)) }, // self-intersecting → winding 2 core
    { label: 'star {5/2} even-odd', quads: polygon(starPts(64, 64, 52, 5, 2)), evenodd: true }, // hollow core
    // winding-fold failure mechanisms, straight from tools/failure.js (same coordinates):
    { label: 'fold A ±1 cancellation', fold: true, // +1 half abuts −1 half: true 1, fold 0 → black seam
      quads: [...rect(16, 16, 64.5, 112, +1), ...rect(64.5, 16, 112, 112, -1)] },
    { label: 'fold B winding ×2', fold: true, // doubled contour → +2: edge AA saturates, edge fattens ~½px
      quads: [...rect(16, 16, 64.5, 112, +1), ...rect(16, 16, 64.5, 112, +1)] },
    { label: 'fold C overlap {0,1,2}', fold: true, // overlap corner sees three winding levels → over-counts
      quads: [...rect(16, 16, 80.5, 112, +1), ...rect(48, 40.5, 128, 88.5, +1)] },
    { label: 'fold D even-odd halo', fold: true, evenodd: true, // doubled contour: empty interior, false halo
      quads: [...rect(24.5, 24.5, 96.5, 96.5, +1), ...rect(24.5, 24.5, 96.5, 96.5, +1)] },
    { label: 'fold E1 w=1 (control)', fold: true, // same averaged winding as E2, different true coverage:
      quads: rect(16, 16, 64.5, 112, +1) }, // single edge at 50% of the column — the fold is exact here…
    { label: 'fold E2 w=2 doubled', fold: true, // …and 2× too high here; ours renders E1 and E2 identically
      quads: [...rect(16, 16, 64.25, 112, +1), ...rect(16, 16, 64.25, 112, +1)] },
    { label: 'fold F minified fence', fold: true, quads: fence() }, // ±1 bars: true ≈ 1, fold fades to black
    ...[...ALPHABET].map((ch) => ({ label: `glyph '${ch}'`, quads: glyphShape(font, ch) })),
  ];
}
