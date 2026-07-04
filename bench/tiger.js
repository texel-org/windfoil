// tiger.js — the third scene: an actual SVG drawing (the Ghostscript tiger), 304 overlapping shapes / ~14k
// quadratics from bench/fixtures/tiger-quadratics.json. Unlike the glyph grid (non-overlapping) and the single
// complex shape, this has real painter's-order OVERDRAW — many shapes stack over the same pixels, so a pixel
// runs one fragment per covering shape, each gathering that shape's own bands. It's the scene windfoil's
// smaller bands buffer was designed around (see the repo README's tiger figures).
//
// Each shape becomes one instance pointing at its own band header in the shared atlas; the whole drawing is one
// tileable unit (like a glyph), tiled to fill the viewport so every zoom renders a full screen. Fill rule is
// nonzero for all; "stroke" shapes are filled as-is (we're measuring coverage cost, not stroking — colors and
// stroke semantics are ignored per the task).

import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { bandSlugShape } from './slug.js';

const FIXTURE = new URL('./fixtures/tiger-quadratics.json', import.meta.url);

// bbox over every point (endpoints + control) of a flat quad list, for conservative banding.
function quadsBbox(quads) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]); x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]); y1 = Math.max(y1, quads[i + 1]);
  }
  return [x0, y0, x1, y1];
}

/**
 * Build the tiger scene: parse the fixture, band every shape into both atlases, and tile the whole drawing to
 * fill [−extent, extent]². Returns the normalized scene object the harness consumes (identical shape to the
 * glyph and complex-shape scenes).
 *
 * @param {object} o { emWorld, extent, color }  color: [r,g,b] override, or null to use each shape's own color
 */
export async function buildTigerScene({ emWorld, extent, color = null }) {
  const doc = JSON.parse(await Deno.readTextFile(FIXTURE));
  const [bx0, by0, bx1, by1] = doc.bbox;
  const cx = (bx0 + bx1) / 2, cy = (by0 + by1) / 2; // center the drawing on the origin
  const span = Math.max(bx1 - bx0, by1 - by0);
  const scale = emWorld / span; // drawing units → world units (whole drawing ≈ emWorld world units)

  const wCurves = [], wRows = [], sCurves = [], sRows = [];
  const shapes = []; // { bbox, color, w:{rowBase,bandCount,y0,invH}, s:{...} }
  let rawCurves = 0, monoPieces = 0;

  for (const shape of doc.data) {
    // Flatten this shape's curves to [x0,y0,cx,cy,x1,y1,...], centered on the origin. SVG fill semantics
    // implicitly close every subpath, but the fixture stores only the curves that were drawn — two shapes
    // (152, 160) end a subpath away from its start. An unclosed contour has no well-defined winding number
    // (the two algorithms would disagree along the band swept by the missing edge — windfoil showed a phantom
    // hairline, Slug a phantom speck), so append the implicit closing edge as a straight quad (control at the
    // midpoint) wherever a subpath's end ≠ its start. Both atlases consume `quads`, so both stay identical.
    const cs = shape.curves;
    const quads = [];
    let subX = 0, subY = 0; // current subpath's start point (centered)
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      const x0 = c[0] - cx, y0 = c[1] - cy, x2 = c[4] - cx, y2 = c[5] - cy;
      if (i === 0 || quads[quads.length - 2] !== x0 || quads[quads.length - 1] !== y0) {
        subX = x0; subY = y0; // a pen-up move → new subpath
      }
      quads.push(x0, y0, c[2] - cx, c[3] - cy, x2, y2);
      const last = i === cs.length - 1;
      const breaks = last || cs[i + 1][0] - cx !== x2 || cs[i + 1][1] - cy !== y2;
      if (breaks && (x2 !== subX || y2 !== subY)) {
        quads.push(x2, y2, (x2 + subX) / 2, (y2 + subY) / 2, subX, subY);
      }
    }
    rawCurves += quads.length / 6;
    const bbox = quadsBbox(quads);

    // windfoil: split into xy-monotone pieces, band by y.
    const pieces = [];
    for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
    monoPieces += pieces.length / 6;
    const wH = bandPieces(pieces, bbox[1], bbox[3], wCurves, wRows);
    // slug: whole quads into both band sets.
    const sH = bandSlugShape(quads, bbox, sCurves, sRows);

    shapes.push({ bbox, color: shape.color || [0, 0, 0], w: wH, s: sH });
  }

  // Tile: one full copy of the drawing centered on each grid point.
  const spacing = emWorld * 1.06;
  const start = -Math.ceil(extent / spacing) * spacing;
  const w = [], s = [];
  let count = 0;
  for (let gy = start; gy <= extent; gy += spacing) {
    for (let gx = start; gx <= extent; gx += spacing) {
      for (const sh of shapes) {
        const [r, g, b] = color || sh.color;
        const bb = sh.bbox;
        w.push(
          gx, gy, scale, 0, bb[0], bb[1], bb[2], bb[3], r, g, b, 1,
          sh.w.rowBase, sh.w.bandCount, sh.w.y0, sh.w.invH,
        );
        s.push(
          gx, gy, scale, 0, bb[0], bb[1], bb[2], bb[3], r, g, b, 1,
          sh.s.hRowBase, sh.s.hBandCount, sh.s.y0, sh.s.invH,
          sh.s.vRowBase, sh.s.vBandCount, sh.s.rotY0, sh.s.invW,
        );
        count++;
      }
    }
  }

  return {
    wCurves: new Float32Array(wCurves), wRows: new Uint32Array(wRows),
    sCurves: new Float32Array(sCurves), sRows: new Uint32Array(sRows),
    wInstances: new Float32Array(w), sInstances: new Float32Array(s),
    count, center: { x: 0, y: 0 }, worldSpan: 2 * extent,
    stats: {
      shapes: shapes.length, rawCurves, monoPieces,
      wBands: wRows.length / 5, wBanded: wCurves.length / 6,
      sBands: sRows.length / 5, sBanded: sCurves.length / 6,
      tiles: count / shapes.length,
    },
  };
}
