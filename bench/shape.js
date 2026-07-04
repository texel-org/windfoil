// shape.js — a complex, self-crossing curve scene, the counterpoint to the glyph grid. Glyphs are sparse (a
// few curves per band, ~1 edge per pixel) which is Slug's sweet spot; this scene is the opposite: one shape
// built from ~240 whole quadratics that all span the shape's extent and overlap into a high-winding core.
//
// It stresses exactly where windfoil should pull ahead of Slug:
//   • Self-crossing / overlap → many edges per pixel, so Slug's ~1-edge-per-pixel coverage model degrades
//     (visible in --check), while windfoil's area integral stays exact up to its own fold limit.
//   • Curves span every band → a pixel's band is packed with curves, most of them far from the pixel. windfoil
//     handles a far curve with a compare + a clamp/subtract of its endpoints (NO root solve), and reads ONE
//     band axis; Slug root-solves every non-left curve, for BOTH the horizontal and the vertical ray. So on a
//     dense shape windfoil does far fewer solves and ~half the band reads.
//
// The shape is a starburst of N narrow ellipses rotated about the center. Ellipses are the natural "big curve"
// (each arc spans much of the shape), rotating them piles a deep winding stack at the middle, and it's cleanly
// parametric so the curve count is easy to dial. Reuses the same primitives as the rest of the repo:
// pushMonotonePieces (windfoil bands) and bandSlugShape (Slug dual bands).

import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { bandSlugShape } from './slug.js';
import { INK } from './scene.js';

// One ellipse (center c, radii rx/ry, rotated by `rot`) as `arcs` quadratic-Bézier arcs, appended to `out` as
// flat [x0,y0, cx,cy, x1,y1] runs. Control point at the tangent-intersection (miter) of each arc — the same
// circle-as-quadratics construction the validation harness uses.
function ellipseQuads(out, cx, cy, rx, ry, rot, arcs) {
  const cr = Math.cos(rot), sr = Math.sin(rot);
  const at = (ex, ey) => [cx + ex * cr - ey * sr, cy + ex * sr + ey * cr]; // ellipse-local → world
  for (let k = 0; k < arcs; k++) {
    const a0 = (k / arcs) * 2 * Math.PI, a1 = ((k + 1) / arcs) * 2 * Math.PI, am = (a0 + a1) / 2;
    const kf = 1 / Math.cos((a1 - a0) / 2); // push the control out to the tangent intersection
    const [x0, y0] = at(rx * Math.cos(a0), ry * Math.sin(a0));
    const [px, py] = at(rx * Math.cos(am) * kf, ry * Math.sin(am) * kf);
    const [x1, y1] = at(rx * Math.cos(a1), ry * Math.sin(a1));
    out.push(x0, y0, px, py, x1, y1);
  }
}

/** The complex shape as flat whole quads + its point bbox. N narrow ellipses rotated over [0, π). */
export function complexShapeQuads({ n = 30, arcs = 8, rx = 1000, ry = 280 } = {}) {
  const quads = [];
  for (let i = 0; i < n; i++) ellipseQuads(quads, 0, 0, rx, ry, (i / n) * Math.PI, arcs);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]); x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]); y1 = Math.max(y1, quads[i + 1]);
  }
  return { quads, bbox: [x0, y0, x1, y1] };
}

/**
 * Build the complex-shape scene: the shape's windfoil and Slug atlases, plus a grid of instances tiling
 * [−extent, extent]² so the viewport is glyph-… shape-filled at every zoom (same framework as the glyph grid).
 * All tiles point at the one shared atlas (the 240 curves are stored once). Returns the same shape the harness
 * consumes for the glyph scene.
 *
 * @param {object} o { emWorld, extent, fillRule (0 nonzero / 1 even-odd), color }
 */
export function buildShapeScene({ emWorld, extent, fillRule = 0, color = INK }) {
  const { quads, bbox } = complexShapeQuads();
  const span = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]);
  const scale = emWorld / span; // shape units → world units (the shape occupies ~emWorld world units)

  // windfoil atlas: split into xy-monotone pieces, then band by y.
  const wCurves = [], wRows = [];
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  const wH = bandPieces(pieces, bbox[1], bbox[3], wCurves, wRows);

  // Slug atlas: whole quads into both band sets.
  const sCurves = [], sRows = [];
  const sH = bandSlugShape(quads, bbox, sCurves, sRows);

  // Tile the shape on a regular grid, one shape centered on the origin so the magnified view lands in a core.
  const [r, g, b, a = 1] = color;
  const spacing = emWorld * 1.06; // shapes nearly touching
  const start = -Math.ceil(extent / spacing) * spacing;
  const w = [], s = [];
  let count = 0;
  for (let gy = start; gy <= extent; gy += spacing) {
    for (let gx = start; gx <= extent; gx += spacing) {
      w.push(
        gx, gy, scale, fillRule, bbox[0], bbox[1], bbox[2], bbox[3], r, g, b, a,
        wH.rowBase, wH.bandCount, wH.y0, wH.invH,
      );
      s.push(
        gx, gy, scale, fillRule, bbox[0], bbox[1], bbox[2], bbox[3], r, g, b, a,
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
    count, center: { x: 0, y: 0 }, worldSpan: 2 * extent, fillRule,
    stats: {
      quads: quads.length / 6,
      wBands: wRows.length / 5, wBanded: wCurves.length / 6,
      sBands: sRows.length / 5, sBanded: sCurves.length / 6,
    },
  };
}
