// harness.js — the environment-agnostic core of the validation suite: the shapes, the three coverage
// sources, the stats and the error-map images. The same code runs under Deno (`deno task validate`, via
// ../validate.js) and in a browser (`deno task serve`, then /tools/validate/); each boot supplies only what
// differs between the two hosts — a 2D-canvas context factory and the WebGPU device.
//
// The four coverage sources per shape (all white-on-black, so the stored byte is 255·coverage, linear):
//   • ours   — the windfoil shader via renderToRGBA (8-bit readback).
//   • slug   — the benchmark's Slug port (bench/slug.wgsl), the other analytic AA model, same GPU pipeline.
//   • canvas — the host's 2D canvas rasterizer: @napi-rs/canvas (Skia) under Deno; whatever the engine
//              uses in a browser (Skia in Chrome, CoreGraphics in Safari, WebRender in Firefox).
//   • box    — the mathematical box filter, estimated by a zero-AA point sample: for each pixel, the
//              fraction of an F×F grid of sub-sample points that fall inside the shape (winding by ray
//              casting against the raw curves — no shared AA model). This is the true filter, but its own
//              per-pixel noise is ~1/F, so it validates the shape of the error and any bias, not sub-1e-3
//              precision.
//
// We report mean |Δcoverage| for each renderer vs the box reference. If "ours vs box" and "canvas vs box"
// are close, both track the box filter equally (the residual is the point-sample noise, not our error).

import { renderToRGBA } from '../../src/gpu.js';
import { pushMonotonePieces } from '../../src/geometry.js';
import { bandPieces } from '../../src/bands.js';
import { glyphQuads } from '../../src/font.js';
import { bandSlugShape, loadSlugShaderCode } from '../../bench/slug.js';

export const S = 128; // cell size in px
export const F = 24; // point-sample grid per pixel for the box-filter reference
export const AMP = 15; // error-map gain: |Δ|·AMP, so 1/AMP coverage reads full-bright
export const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';

// ── shapes: flat quads [x0,y0,cx,cy,x1,y1,...] in cell coordinates (0..S), a line = a midpoint quad ──────
function line(x0, y0, x1, y1) {
  return [x0, y0, (x0 + x1) / 2, (y0 + y1) / 2, x1, y1];
}
function polygon(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) out.push(...line(...pts[i], ...pts[(i + 1) % pts.length]));
  return out;
}
function rotate(pts, deg, cx = S / 2, cy = S / 2) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]);
}
function circle(cx, cy, r, n = 8) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI, am = (a0 + a1) / 2;
    const k = 1 / Math.cos((a1 - a0) / 2);
    out.push(cx + r * Math.cos(a0), cy + r * Math.sin(a0), cx + r * k * Math.cos(am), cy + r * k * Math.sin(am),
      cx + r * Math.cos(a1), cy + r * Math.sin(a1));
  }
  return out;
}
function starPts(cx, cy, r, points, step) {
  const p = [];
  for (let k = 0; k < points; k++) {
    const a = -Math.PI / 2 + ((k * step) % points) * (2 * Math.PI / points);
    p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return p;
}
// The line-based shapes come in two forms of the SAME geometry: `quads` (the outline rectangle of each
// line, which ours and the box reference fill) and `segments` (flat [x0,y0,x1,y1,w] centerlines, which the
// stroked variants hand to the canvas as stroke() + lineWidth). A butt-capped stroked segment IS the
// rectangle, mathematically — so a stroked variant measures the host's stroke pipeline (including any thin-
// stroke/hairline special case) against the identical exact shape.

// A ladder of vertical bars: 4px wide down to a barely-visible hairline, each half the width of the last.
// Each bar gets a different sub-pixel phase (the i·0.37 term) so edges straddle pixel boundaries instead of
// snapping to the grid, where every rasterizer is trivially exact.
function hairlines(n = 6) {
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
function spokes(cx, cy, r0, r1, n, w) {
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

// Axis-aligned rectangle as a closed contour. dir = +1 or −1 flips the traversal, flipping its winding sign.
function rect(x0, y0, x1, y1, dir = 1) {
  const cs = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return polygon(dir >= 0 ? cs : cs.slice().reverse());
}

// A +1/−1 picket fence of k sub-pixel bars (all dyadic widths, so the geometry is float-exact). Every bar is
// filled under nonzero (true coverage ≈ 1 across the block), but a 1px footprint spans many opposite-sign
// bars, so the averaged winding → 0 and the fold fades toward black — the minification regime of
// tools/failure.js at native scale.
function fence(k = 256) {
  const x0 = 20, y0 = 20, x1 = 108, y1 = 108, bw = (x1 - x0) / k;
  const out = [];
  for (let i = 0; i < k; i++) out.push(...rect(x0 + i * bw, y0, x0 + (i + 1) * bw, y1, i % 2 ? -1 : 1));
  return out;
}

function glyphShape(font, ch) {
  const g = glyphQuads(font, ch);
  const [x0, y0, x1, y1] = g.bbox;
  const gw = x1 - x0, gh = y1 - y0, pad = 14, box = S - 2 * pad;
  const k = Math.min(box / gw, box / gh);
  const ox = pad + (box - gw * k) / 2 - x0 * k, oy = pad + (box - gh * k) / 2 - y0 * k;
  return g.quads.map((v, i) => (i % 2 === 0 ? ox + v * k : oy + v * k));
}

/**
 * The full suite: the synthetic stress shapes, the winding-fold failure cases, then every lowercase letter
 * of the given font. Entries are { label, quads, evenodd?, segments?, fold? }:
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

// ── 1. our shader ───────────────────────────────────────────────────────────────────────────────────
function buildScene(quads, evenodd, scale) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]); x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]); y1 = Math.max(y1, pieces[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, bandH, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([0, 0, scale, rule, x0, y0, x1, y1, 1, 1, 1, 1, rowBase, bandCount, bandH, invH]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

// ss > 1 is the "exact mode" knob: render the shader at ss× resolution and box-average back down to S×S.
// The winding fold then applies per SUB-pixel (whose footprint is 1/ss of a pixel), so the documented fold
// failures shrink ~1/ss and the result converges to the exact box filter as ss grows — at ss× the cost.
// Common (fold-exact) shapes are unchanged apart from less 8-bit readback noise. Not a shader mode — the
// fold itself stays lossy; this just shrinks what it loses.
//
// Exact mode also specializes the pipeline with MINIFICATION_GUARD off (a WGSL override constant, set via
// the standard pipeline `constants` map), so every sub-pixel is the true integral: the guard's ink-profile
// approximation must never stand in for an "exact" measurement. (It gates on the whole ink box shrinking
// below ~3.7 device px, so it can't fire on this suite's ~90px shapes — and supersampling only grows the
// device-space ink box — but tiny shapes added later shouldn't silently leak the approximation into
// exact-mode numbers.)
export async function ourCoverage(device, quads, evenodd, ss = 1) {
  const { curves, rows, instances } = buildScene(quads, evenodd, ss);
  const W = S * ss;
  const constants = ss > 1 ? { MINIFICATION_GUARD: 0 } : undefined; // 0 = false
  const rgba = await renderToRGBA({
    device, constants, width: W, height: W, background: [0, 0, 0, 1], curves, rows, instances, instanceCount: 1,
  });
  const out = new Float64Array(S * S);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) out[((y / ss) | 0) * S + ((x / ss) | 0)] += rgba[(y * W + x) * 4] / 255;
  }
  for (let i = 0; i < out.length; i++) out[i] /= ss * ss;
  return out;
}

// ── 2. Slug (bench/slug.wgsl) — the second analytic AA model, on the same GPU device ───────────────────
// Whole quads into Slug's dual band sets (bench/slug.js); the instance carries both band headers (20 floats).
// Same 4-binding pipeline as ours, different shader. Always rendered at 1× — like the canvas and the box
// filter, it is a reference, so exact mode's supersampling applies only to ours.
function buildSlugScene(quads, evenodd) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]); x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]); y1 = Math.max(y1, quads[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const sH = bandSlugShape(quads, [x0, y0, x1, y1], curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([
    0, 0, 1, rule, x0, y0, x1, y1, 1, 1, 1, 1,
    sH.hRowBase, sH.hBandCount, sH.y0, sH.invH,
    sH.vRowBase, sH.vBandCount, sH.rotY0, sH.invW,
  ]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

export async function slugCoverage(device, quads, evenodd) {
  const { curves, rows, instances } = buildSlugScene(quads, evenodd);
  const rgba = await renderToRGBA({
    device, code: await loadSlugShaderCode(), width: S, height: S, background: [0, 0, 0, 1],
    curves, rows, instances, instanceCount: 1,
  });
  const out = new Float64Array(S * S);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4] / 255;
  return out;
}

// ── 3. the host's 2D canvas (Skia under Deno, the engine's rasterizer in a browser) ────────────────────
// With `segments` ([x0,y0,x1,y1,w] centerlines), the canvas strokes them (butt caps — the same rectangles
// the quads describe) instead of filling the path, exercising the host's stroke pipeline.
export function canvasCoverage(createContext2D, quads, evenodd, segments) {
  const ctx = createContext2D(S, S);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, S, S);
  if (segments) {
    ctx.strokeStyle = '#fff';
    ctx.lineCap = 'butt';
    for (let i = 0; i < segments.length; i += 5) {
      ctx.lineWidth = segments[i + 4];
      ctx.beginPath();
      ctx.moveTo(segments[i], segments[i + 1]);
      ctx.lineTo(segments[i + 2], segments[i + 3]);
      ctx.stroke();
    }
    return readCoverage(ctx);
  }
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
  return readCoverage(ctx);
}

function readCoverage(ctx) {
  const d = ctx.getImageData(0, 0, S, S).data;
  const out = new Float64Array(S * S);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4] / 255;
  return out;
}

// ── 4. point-sampled box filter (independent: winding by ray casting the raw curves) ───────────────────
// Every crossing of a rightward ray at height py against the raw quads: its x position and winding sign.
function crossingsAt(py, quads) {
  const cross = [];
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
      const dy = 2 * a * t + b;
      cross.push([xt, dy >= 0 ? 1 : -1]);
    }
  }
  return cross;
}

// For each pixel, the fraction of an F×F sub-sample grid inside the shape. One ray per sub-sample row
// serves every sample column on it: walk the columns right-to-left past the crossings sorted rightmost-
// first, keeping the running signed winding W and crossing count K. Each sample sees exactly the crossings
// with xt > px — the same winding number a per-point ray cast computes, without re-solving the quads per
// column.
export function boxCoverage(quads, evenodd) {
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let j = 0; j < F; j++) {
      const cross = crossingsAt(y + (j + 0.5) / F, quads).sort((p, q) => q[0] - p[0]);
      let ptr = 0, W = 0, K = 0;
      for (let x = S - 1; x >= 0; x--) {
        for (let i = F - 1; i >= 0; i--) {
          const px = x + (i + 0.5) / F;
          while (ptr < cross.length && cross[ptr][0] > px) { W += cross[ptr][1]; K++; ptr++; }
          if (evenodd ? (K & 1) === 1 : W !== 0) out[y * S + x]++;
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) out[i] /= F * F;
  return out;
}

// ── compare ─────────────────────────────────────────────────────────────────────────────────────────
export function stats(a, b) {
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    sum += e;
    if (e > max) max = e;
  }
  return { mean: sum / a.length, max };
}

// ── report images: S×S RGBA8 (boots upscale for PNGs or let CSS scale a canvas) ────────────────────────
const mapRGBA = (rgbAt) => {
  const d = new Uint8Array(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    const [r, g, b] = rgbAt(i);
    const o = i * 4;
    d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
  }
  return d;
};
/** Coverage as grayscale (white = covered). */
export const grayRGBA = (cov) => mapRGBA((i) => { const v = Math.round(cov[i] * 255); return [v, v, v]; });
/** |Δcoverage| amplified ×AMP so faint differences show (hot orange). */
export const diffRGBA = (a, b) =>
  mapRGBA((i) => {
    const v = Math.round(Math.min(Math.abs(a[i] - b[i]) * AMP, 1) * 255);
    return [v, Math.round(v * 0.28), Math.round(v * 0.12)];
  });

export const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');

/**
 * Run the whole suite, yielding one result per shape as it completes — so a boot can stream rows into a
 * console table (Deno) or the page (browser) while later shapes are still rendering.
 *
 * @param {object} o
 * @param {object} o.font              a parsed font (see font.js loadFont/parseFont)
 * @param {Function} o.createContext2D (w, h) → a 2D canvas context in the host environment
 * @param {GPUDevice} o.device         a shared WebGPU device (see gpu.js requestDevice)
 * @param {number} [o.supersample]     render ours at this factor and box-average down (see ourCoverage);
 *                                     the canvas and box references stay at 1× — they're the yardstick
 */
export async function* validateShapes({ font, createContext2D, device, supersample = 1 }) {
  for (const { label, quads, evenodd = false, segments, fold = false } of buildShapes(font)) {
    const ours = await ourCoverage(device, quads, evenodd, supersample);
    const slug = await slugCoverage(device, quads, evenodd);
    const canvas = canvasCoverage(createContext2D, quads, evenodd, segments);
    const box = boxCoverage(quads, evenodd);
    yield {
      label, fold, ours, slug, canvas, box,
      oursVsBox: stats(ours, box), slugVsBox: stats(slug, box), canvasVsBox: stats(canvas, box),
    };
  }
}
