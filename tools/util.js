// util.js — shared, environment-agnostic building blocks for the coverage tools: validate.js (Deno, using
// @napi-rs/canvas) and tools/chrome/ (browser, using a <canvas> element). Everything here is pure geometry +
// rasterization glue that runs the same under Deno and Chrome. The two environment-specific pieces — loading
// the font (off disk vs. over fetch) and which Canvas 2D backs `canvasCoverage` — are passed in by the caller,
// so this file never imports @napi-rs/canvas or touches Deno/DOM globals directly.

import { renderToRGBA } from '../src/gpu.js';
import { pushMonotonePieces } from '../src/geometry.js';
import { bandPieces } from '../src/bands.js';
import { glyphQuads } from '../src/font.js';

export const S = 128; // cell size in px
export const F = 24; // point-sample grid per pixel for the box-filter reference

// ── shapes: flat quads [x0,y0,cx,cy,x1,y1,...] in cell coordinates (0..S), a line = a midpoint quad ──────
export function line(x0, y0, x1, y1) {
  return [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1];
}
export function polygon(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) out.push(...line(...pts[i], ...pts[(i + 1) % pts.length]));
  return out;
}
export function rotate(pts, deg, cx = S / 2, cy = S / 2) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]);
}
export function circle(cx, cy, r, n = 8) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI, am = (a0 + a1) / 2;
    const k = 1 / Math.cos((a1 - a0) / 2);
    out.push(cx + r * Math.cos(a0), cy + r * Math.sin(a0), cx + r * k * Math.cos(am), cy + r * k * Math.sin(am),
      cx + r * Math.cos(a1), cy + r * Math.sin(a1));
  }
  return out;
}
export function starPts(cx, cy, r, points, step) {
  const p = [];
  for (let k = 0; k < points; k++) {
    const a = -Math.PI / 2 + ((k * step) % points) * (2 * Math.PI / points);
    p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return p;
}
export function glyphShape(font, ch) {
  const g = glyphQuads(font, ch);
  const [x0, y0, x1, y1] = g.bbox;
  const gw = x1 - x0, gh = y1 - y0, pad = 14, box = S - 2 * pad;
  const k = Math.min(box / gw, box / gh);
  const ox = pad + (box - gw * k) / 2 - x0 * k, oy = pad + (box - gh * k) / 2 - y0 * k;
  return g.quads.map((v, i) => (i % 2 === 0 ? ox + v * k : oy + v * k));
}

// The validation shape set. Font-dependent (glyphs), so a factory. Each entry: [label, quads, evenodd].
export function makeShapes(font) {
  return [
    ['rotated square 30°', polygon(rotate([[28, 28], [100, 28], [100, 100], [28, 100]], 30)), false],
    ['thin diagonal sliver', polygon(rotate([[12, 63.5], [116, 63.5], [116, 64.5], [12, 64.5]], 27)), false],
    ['circle r=44', circle(64, 64, 44, 64), false], // 64 arcs: smooth enough that curve flattening is negligible
    ["glyph 'o' (with hole)", glyphShape(font, 'o'), false],
    ["glyph 'e' (aperture)", glyphShape(font, 'e'), false],
    ['star {5/2} nonzero', polygon(starPts(64, 64, 52, 5, 2)), false], // self-intersecting → winding 2 core
    ['star {5/2} even-odd', polygon(starPts(64, 64, 52, 5, 2)), true], // hollow core
  ];
}

// ── our shader (WebGPU; runs under both Deno --unstable-webgpu and the browser) ─────────────────────────
export function buildScene(quads, evenodd, scale) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]); x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]); y1 = Math.max(y1, pieces[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, y0: by0, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([0, 0, scale, rule, x0, y0, x1, y1, 1, 1, 1, 1, rowBase, bandCount, by0, invH]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

export async function ourCoverage(quads, evenodd) {
  const { curves, rows, instances } = buildScene(quads, evenodd, 1);
  const rgba = await renderToRGBA({
    width: S, height: S, background: [0, 0, 0, 1], curves, rows, instances, instanceCount: 1,
  });
  const out = new Float64Array(S * S);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4] / 255;
  return out;
}

// ── Canvas 2D coverage — the DRY core of the old skiaCoverage. `ctx` is any Canvas 2D context backed by an
// S×S canvas: @napi-rs/canvas under Deno, a <canvas> element in the browser (both are Skia, possibly different
// builds — the whole point of the comparison). Returns per-pixel coverage 0..1 (255·coverage read back as a byte).
export function canvasCoverage(quads, evenodd, ctx) {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, S, S);
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  let px = null, py = null;
  for (let i = 0; i < quads.length; i += 6) {
    const [x0, y0, cx, cy, x1, y1] = quads.slice(i, i + 6);
    if (px === null || Math.abs(x0 - px) > 1e-4 || Math.abs(y0 - py) > 1e-4) {
      if (px !== null) ctx.closePath();
      ctx.moveTo(x0, y0);
    }
    ctx.quadraticCurveTo(cx, cy, x1, y1);
    px = x1; py = y1;
  }
  ctx.closePath();
  ctx.fill(evenodd ? 'evenodd' : 'nonzero');
  const d = ctx.getImageData(0, 0, S, S).data;
  const out = new Float64Array(S * S);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255;
  return out;
}

// ── point-sampled box filter (independent: winding by ray casting the raw curves) ───────────────────────
// Signed winding W and crossing count K of a rightward ray from (px,py) against the raw quads.
export function windingAt(px, py, quads) {
  let W = 0, K = 0;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3], x1 = quads[i + 4], y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue; // hull y-reject
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) { const sq = Math.sqrt(disc); t0 = (-b + sq) / (2 * a); t1 = (-b - sq) / (2 * a); }
    }
    for (const t of [t0, t1]) {
      if (t < 0 || t > 1) continue;
      const xt = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1;
      if (xt > px) { K++; const dy = 2 * a * t + b; W += dy >= 0 ? 1 : -1; }
    }
  }
  return { W, K };
}

export function boxCoverage(quads, evenodd) {
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let inside = 0;
      for (let j = 0; j < F; j++) {
        for (let i = 0; i < F; i++) {
          const { W, K } = windingAt(x + (i + 0.5) / F, y + (j + 0.5) / F, quads);
          if (evenodd ? (K & 1) === 1 : W !== 0) inside++;
        }
      }
      out[y * S + x] = inside / (F * F);
    }
  }
  return out;
}

// ── stats + naming ──────────────────────────────────────────────────────────────────────────────────────
export function stats(a, b) {
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    sum += e;
    if (e > max) max = e;
  }
  return { mean: sum / a.length, max };
}

export const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
