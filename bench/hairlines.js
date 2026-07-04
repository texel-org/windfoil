// hairlines.js — the AA-quality scene: geometry chosen so the difference between Slug's point-sampled
// coverage model and windfoil's exact area integral is visible to the naked eye, not just in diff heatmaps.
// On plain text the two models nearly coincide (see README "check" numbers); each of this scene's three
// concentric elements isolates a case where they don't:
//
//   • hairline fan (r 100–460) — 48 sub-pixel-width strokes through every angle. A point-sampled ray
//     either slices a near-parallel thin stroke for a long run (coverage → 1) or misses it, so Slug renders
//     the fan beaded, with tone varying by angle (worst near axis-aligned); the area integral gives every
//     stroke the same smooth sub-pixel gray regardless of angle.
//   • taper ring (r 500–700) — 36 needle spikes converging to near-point tips, the corner case where Slug
//     leaves thin needles / isolated specks (its min(|xcov|,|ycov|) floor meets ill-conditioned crossings).
//   • zone plate (r 720–980) — concentric rings whose period shrinks outward through the pixel grid. Slug's
//     per-ray coverage is a sum of saturating ramps assuming ~1 edge per pixel; several crossings per pixel
//     deviate from the true average, where the integral degrades to the correct gray.
//   • crossing (r < 95, dead center, where the deep-zoom levels land) — three strokes overlapping IN ONE
//     SHAPE, two of them at a shallow ±4° angle so their edges run within a pixel of each other for a long
//     stretch. This is raw self-intersecting input with no union sweep: the case a Slug-based renderer
//     needs a CPU prepass (flattening to single edges at a device tolerance) to draw cleanly, and the area
//     integral handles as-is at any zoom.
//
// One closed-contour shape per tile (like the complex-shape scene), tiled to fill the viewport at every zoom.

import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { bandSlugShape } from './slug.js';
import { INK } from './scene.js';

// A line as a midpoint-control quad, the repo-wide convention.
function pushLine(out, x0, y0, x1, y1) {
  out.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1);
}

// Closed rectangle of width `w` running radially from r0 to r1 at angle `a` — one hairline stroke.
// r0 may be negative, giving a full chord through the center (the crossing element).
function strokeQuads(out, a, r0, r1, w) {
  const ca = Math.cos(a), sa = Math.sin(a);
  const nx = -sa * (w / 2), ny = ca * (w / 2); // unit normal × half-width
  const ax = ca * r0, ay = sa * r0, bx = ca * r1, by = sa * r1;
  pushLine(out, ax - nx, ay - ny, bx - nx, by - ny);
  pushLine(out, bx - nx, by - ny, bx + nx, by + ny);
  pushLine(out, bx + nx, by + ny, ax + nx, ay + ny);
  pushLine(out, ax + nx, ay + ny, ax - nx, ay - ny);
}

// Closed triangle from a base of width `wBase` at r0 to a point tip at r1 — one needle spike.
function spikeQuads(out, a, r0, r1, wBase) {
  const ca = Math.cos(a), sa = Math.sin(a);
  const nx = -sa * (wBase / 2), ny = ca * (wBase / 2);
  const ax = ca * r0, ay = sa * r0, tx = ca * r1, ty = sa * r1;
  pushLine(out, ax - nx, ay - ny, tx, ty);
  pushLine(out, tx, ty, ax + nx, ay + ny);
  pushLine(out, ax + nx, ay + ny, ax - nx, ay - ny);
}

// Circle of radius r as `arcs` quadratic arcs (control at the tangent miter — same construction as the
// complex-shape scene). `reverse` flips the orientation, so ring = outer circle + reversed inner circle.
function circleQuads(out, r, arcs, reverse) {
  for (let k = 0; k < arcs; k++) {
    const i = reverse ? arcs - 1 - k : k;
    let a0 = (i / arcs) * 2 * Math.PI, a1 = ((i + 1) / arcs) * 2 * Math.PI;
    if (reverse) [a0, a1] = [a1, a0];
    const am = (a0 + a1) / 2, kf = 1 / Math.cos((a1 - a0) / 2);
    out.push(r * Math.cos(a0), r * Math.sin(a0), r * kf * Math.cos(am), r * kf * Math.sin(am), r * Math.cos(a1), r * Math.sin(a1));
  }
}

/** The hairline test pattern as flat whole quads + its point bbox. */
export function hairlineQuads({ strokes = 48, spikes = 36, arcs = 16 } = {}) {
  const quads = [];
  // crossing: raw overlapping strokes — a shallow ±4° pair (edges share pixels for a long run) + a steep one
  const deg = Math.PI / 180;
  strokeQuads(quads, 4 * deg, -95, 95, 14);
  strokeQuads(quads, -4 * deg, -95, 95, 14);
  strokeQuads(quads, 78 * deg, -95, 95, 14);
  // fan: sub-pixel strokes through every angle (includes exactly-horizontal and exactly-vertical)
  for (let i = 0; i < strokes; i++) strokeQuads(quads, (i / strokes) * 2 * Math.PI, 100, 460, 2.5);
  // taper ring: needle tips pointing outward
  for (let i = 0; i < spikes; i++) spikeQuads(quads, (i / spikes) * 2 * Math.PI, 500, 700, 8);
  // zone plate: 50%-duty rings, period shrinking 16 → 4 units toward the rim
  for (let r = 720; ; ) {
    const period = 16 - (12 * (r - 720)) / 260;
    if (r + period / 2 > 980) break;
    circleQuads(quads, r + period / 2, arcs, false); // outer edge
    circleQuads(quads, r, arcs, true); // inner edge, reversed → a ring under nonzero fill
    r += period;
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]); x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]); y1 = Math.max(y1, quads[i + 1]);
  }
  return { quads, bbox: [x0, y0, x1, y1] };
}

/**
 * Build the hairline scene: both atlases plus a grid of instances tiling [−extent, extent]², one pattern per
 * tile (same framework as the complex-shape scene). Returns the shape the harness consumes.
 *
 * @param {object} o { emWorld, extent, color }
 */
export function buildHairlineScene({ emWorld, extent, color = INK }) {
  const { quads, bbox } = hairlineQuads();
  const span = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]);
  const scale = emWorld / span;

  // windfoil atlas: split into xy-monotone pieces, then band by y.
  const wCurves = [], wRows = [];
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  const wH = bandPieces(pieces, bbox[1], bbox[3], wCurves, wRows);

  // Slug atlas: whole quads into both band sets.
  const sCurves = [], sRows = [];
  const sH = bandSlugShape(quads, bbox, sCurves, sRows);

  const [r, g, b, a = 1] = color;
  const spacing = emWorld * 1.06;
  const start = -Math.ceil(extent / spacing) * spacing;
  const w = [], s = [];
  let count = 0;
  for (let gy = start; gy <= extent; gy += spacing) {
    for (let gx = start; gx <= extent; gx += spacing) {
      w.push(
        gx, gy, scale, 0, bbox[0], bbox[1], bbox[2], bbox[3], r, g, b, a,
        wH.rowBase, wH.bandCount, wH.y0, wH.invH,
      );
      s.push(
        gx, gy, scale, 0, bbox[0], bbox[1], bbox[2], bbox[3], r, g, b, a,
        sH.hRowBase, sH.hBandCount, sH.y0, sH.invH,
        sH.vRowBase, sH.vBandCount, sH.rotY0, sH.invW,
      );
      count++;
    }
  }

  return {
    wCurves: new Float32Array(wCurves), wRows: new Uint32Array(wRows),
    sCurves: new Float32Array(sCurves), sRows: new Uint32Array(sRows),
    wInstances: new Float32Array(w), sInstances: new Float32Array(s),
    count, center: { x: 0, y: 0 }, worldSpan: 2 * extent, fillRule: 0,
    stats: {
      quads: quads.length / 6,
      wBands: wRows.length / 5, wBanded: wCurves.length / 6,
      sBands: sRows.length / 5, sBanded: sCurves.length / 6,
    },
  };
}
