// harness.js — the environment-agnostic core of the validation suite: the shapes, the coverage sources
// (including the box-filter oracle), the stats and the error-map images. The same code runs under Deno (`deno task validate`, via
// ../validate.js) and in a browser (`deno task serve`, then /tools/validate/); each boot supplies only what
// differs between the two hosts — a 2D-canvas context factory and the WebGPU device.
//
// The four coverage sources per shape (all white-on-black fills):
//   • ours   — the windfoil shader via renderToCoverage (r32float target, so the comparison sees the
//              fragment stage's exact f32 coverage — no 8-bit quantization floor).
//   • slug   — the benchmark's Slug port (bench/slug.wgsl), the other analytic AA model, same GPU pipeline
//              and float readback.
//   • canvas — the host's 2D canvas rasterizer: @napi-rs/canvas (Skia) under Deno; whatever the engine
//              uses in a browser (Skia in Chrome, CoreGraphics in Safari, WebRender in Firefox). Its API is
//              8-bit (getImageData), so this column carries a ~1/(4·255) ≈ 1e-3 quantization floor of its
//              own — negligible against its actual distance from the box-filter target.
//   • oracle — the box filter itself, computed to high precision by a geometric oracle independent of the
//              shader's area integral: along each scanline the covered length per pixel column is EXACT
//              (interval arithmetic over ray-cast crossings of the raw curves), and the remaining
//              one-dimensional integral over y uses adaptive Gauss–Legendre between the analytic
//              breakpoints (curve endpoint / extremum heights), in f64. Each shape reports the oracle's own
//              convergence residual (`refErr`, typically ≤ 1e-9), so reference noise is quantified, not
//              assumed away. `boxCoverage(quads, evenodd, f)` keeps the old F×F zero-AA point sample —
//              per-pixel quantization 1/f² — for the convergence study that shows it approaching the
//              oracle as f grows.
//
// Stats per renderer-vs-oracle: mean and worst-pixel |Δcoverage| over the whole cell, plus mean / median /
// p99 restricted to the "edge" pixels a curve actually affects (edgeMaskFor) — interior/exterior pixels
// agree trivially, so cell-wide means understate edge behaviour; the edge-restricted stats are the honest
// ones.

import { renderToCoverage } from '../../src/gpu.js';
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
    out.push(
      cx + r * Math.cos(a0),
      cy + r * Math.sin(a0),
      cx + r * k * Math.cos(am),
      cy + r * k * Math.sin(am),
      cx + r * Math.cos(a1),
      cy + r * Math.sin(a1),
    );
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
    {
      label: 'thin diagonal sliver',
      quads: polygon(rotate([[12, 63.5], [116, 63.5], [116, 64.5], [12, 64.5]], 27)),
    },
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
    {
      label: 'fold A ±1 cancellation',
      fold: true, // +1 half abuts −1 half: true 1, fold 0 → black seam
      quads: [...rect(16, 16, 64.5, 112, +1), ...rect(64.5, 16, 112, 112, -1)],
    },
    {
      label: 'fold B winding ×2',
      fold: true, // doubled contour → +2: edge AA saturates, edge fattens ~½px
      quads: [...rect(16, 16, 64.5, 112, +1), ...rect(16, 16, 64.5, 112, +1)],
    },
    {
      label: 'fold C overlap {0,1,2}',
      fold: true, // overlap corner sees three winding levels → over-counts
      quads: [...rect(16, 16, 80.5, 112, +1), ...rect(48, 40.5, 128, 88.5, +1)],
    },
    {
      label: 'fold D even-odd halo',
      fold: true,
      evenodd: true, // doubled contour: empty interior, false halo
      quads: [...rect(24.5, 24.5, 96.5, 96.5, +1), ...rect(24.5, 24.5, 96.5, 96.5, +1)],
    },
    {
      label: 'fold E1 w=1 (control)',
      fold: true, // same averaged winding as E2, different true coverage:
      quads: rect(16, 16, 64.5, 112, +1),
    }, // single edge at 50% of the column — the fold is exact here…
    {
      label: 'fold E2 w=2 doubled',
      fold: true, // …and 2× too high here; ours renders E1 and E2 identically
      quads: [...rect(16, 16, 64.25, 112, +1), ...rect(16, 16, 64.25, 112, +1)],
    },
    { label: 'fold F minified fence', fold: true, quads: fence() }, // ±1 bars: true ≈ 1, fold fades to black
    ...[...ALPHABET].map((ch) => ({ label: `glyph '${ch}'`, quads: glyphShape(font, ch) })),
  ];
}

// ── 1. our shader ───────────────────────────────────────────────────────────────────────────────────
// snap = true rounds the instance bbox outward to integers. Validation-only: the rasterizer quantizes the
// instance quad's vertex positions to a sub-pixel grid (~2⁻⁸..2⁻⁹ px), which shears the interpolated
// pixel-center rc by up to that much and shifts every edge's coverage accordingly (~1e-3 worst-case on a
// unit-gradient edge) — an artifact of drawing ANY analytic AA through a vertex quad, bit-identical for
// ours and slug, and invisible for shapes whose bbox is already integer (circle, spokes). Snapping the
// bbox puts the vertices on the grid, so the comparison isolates the integral's own arithmetic.
function buildScene(quads, evenodd, scale, snap = false) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pieces.length; i += 2) {
    x0 = Math.min(x0, pieces[i]);
    x1 = Math.max(x1, pieces[i]);
    y0 = Math.min(y0, pieces[i + 1]);
    y1 = Math.max(y1, pieces[i + 1]);
  }
  if (snap) {
    x0 = Math.floor(x0);
    y0 = Math.floor(y0);
    x1 = Math.ceil(x1);
    y1 = Math.ceil(y1);
  }
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, bandH, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([
    0,
    0,
    scale,
    rule,
    x0,
    y0,
    x1,
    y1,
    1,
    1,
    1,
    1,
    rowBase,
    bandCount,
    bandH,
    invH,
  ]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

// exact = true is the "exact mode" knob: specialize the pipeline with the shader's EXACT_MODE override on
// (set via the standard pipeline `constants` map, like MINIFICATION_GUARD), replacing the scalar winding
// fold with in-shader sampling of the TRUE fill rule on an EXACT_GRID×EXACT_GRID grid per pixel
// (windfoil.wgsl exact_coverage — it also bypasses the minification guard). Correct on the winding-fold
// failure cases; ordinary AA edges pick up the grid's sub-sample quantisation (~1/64 coverage steps at the
// default 8×8), so it is an offline/print correctness mode, not a quality upgrade for common fills. The
// override is compiled out of the normal pipeline, so the fast path costs nothing when it is off.
export async function ourCoverage(device, quads, evenodd, exact = false, snapBBox = false) {
  const { curves, rows, instances } = buildScene(quads, evenodd, 1, snapBBox);
  const constants = exact ? { EXACT_MODE: 1 } : undefined;
  const cov = await renderToCoverage({
    device,
    constants,
    width: S,
    height: S,
    background: [0, 0, 0, 1],
    curves,
    rows,
    instances,
    instanceCount: 1,
  });
  return Float64Array.from(cov);
}

// ── 2. Slug (bench/slug.wgsl) — the second analytic AA model, on the same GPU device ───────────────────
// Whole quads into Slug's dual band sets (bench/slug.js); the instance carries both band headers (20 floats).
// Same 4-binding pipeline as ours, different shader. Like the canvas and the box filter it is a reference,
// so exact mode applies only to ours.
function buildSlugScene(quads, evenodd) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < quads.length; i += 2) {
    x0 = Math.min(x0, quads[i]);
    x1 = Math.max(x1, quads[i]);
    y0 = Math.min(y0, quads[i + 1]);
    y1 = Math.max(y1, quads[i + 1]);
  }
  const curveOut = [], rowOut = [];
  const sH = bandSlugShape(quads, [x0, y0, x1, y1], curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([
    0,
    0,
    1,
    rule,
    x0,
    y0,
    x1,
    y1,
    1,
    1,
    1,
    1,
    sH.hRowBase,
    sH.hBandCount,
    sH.y0,
    sH.invH,
    sH.vRowBase,
    sH.vBandCount,
    sH.rotY0,
    sH.invW,
  ]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

export async function slugCoverage(device, quads, evenodd) {
  const { curves, rows, instances } = buildSlugScene(quads, evenodd);
  const cov = await renderToCoverage({
    device,
    code: await loadSlugShaderCode(),
    width: S,
    height: S,
    background: [0, 0, 0, 1],
    curves,
    rows,
    instances,
    instanceCount: 1,
  });
  return Float64Array.from(cov);
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
    px = x1;
    py = y1;
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

// ── 4. the geometric references: ray-cast crossings shared by the point sample and the oracle ──────────
// Every crossing of a rightward ray at height py against the raw quads: its x position and winding sign.
// The quadratic solve uses the cancellation-free form (q = −(b + sign(b)·√disc)/2; roots q/a and c/q) so
// near-tangent scanlines — which the oracle's Gauss–Legendre nodes probe close to curve extrema — keep
// full precision.
function crossingsAt(py, quads) {
  const cross = [];
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i],
      y0 = quads[i + 1],
      cx = quads[i + 2],
      cy = quads[i + 3],
      x1 = quads[i + 4],
      y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue; // hull y-reject
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const q = -0.5 * (b + (b >= 0 ? 1 : -1) * Math.sqrt(disc));
        t0 = q / a;
        t1 = q !== 0 ? c / q : t0;
      }
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

// For each pixel, the fraction of an f×f sub-sample grid inside the shape — the zero-AA point sample the
// suite used as its reference before the oracle existed. Per-pixel quantization is 1/f² (coverage can only
// change in those increments), so it cannot certify errors below that; it stays here, parameterized, for
// the convergence study that shows it approaching the oracle as f grows. One ray per sub-sample row serves
// every sample column on it: walk the columns right-to-left past the crossings sorted rightmost-first,
// keeping the running signed winding W and crossing count K.
export function boxCoverage(quads, evenodd, f = F) {
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let j = 0; j < f; j++) {
      const cross = crossingsAt(y + (j + 0.5) / f, quads).sort((p, q) => q[0] - p[0]);
      let ptr = 0, W = 0, K = 0;
      for (let x = S - 1; x >= 0; x--) {
        for (let i = f - 1; i >= 0; i--) {
          const px = x + (i + 0.5) / f;
          while (ptr < cross.length && cross[ptr][0] > px) {
            W += cross[ptr][1];
            K++;
            ptr++;
          }
          if (evenodd ? (K & 1) === 1 : W !== 0) out[y * S + x]++;
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) out[i] /= f * f;
  return out;
}

// ── 5. the box-filter oracle: exact in x, adaptive Gauss–Legendre in y, f64 ────────────────────────────
// The box-filtered coverage of pixel (c, r) is ∫∫ [fill] dx dy over [c,c+1]×[r,r+1]. The inner x-integral
// needs no quadrature at all: at any height y the fill is a union of intervals bounded by the ray-cast
// crossings, so the covered length inside each pixel column is exact interval arithmetic. Only the outer
// 1-D y-integral is numeric. Per pixel row it is split at the analytic breakpoints — the y-values where a
// curve starts, ends, or turns around (endpoint / extremum heights), the only places the covered length can
// jump — and each smooth piece is integrated by 8-point Gauss–Legendre, comparing the whole-interval
// estimate against its two halves and bisecting until they agree to OTOL (or ODEPTH halvings). The accepted
// disagreements accumulate into a per-row error bound; a shape's `refErr` is the worst row's, so the
// reference's own residual is measured and reported, not assumed. Kinks the breakpoints cannot know about
// (crossings swapping order, covered intervals merging, a crossing sliding over a pixel boundary) are C⁰
// and the bisection localizes onto them.

// 8-point Gauss–Legendre nodes / weights on [−1, 1].
const GL_X = [
  -0.9602898564975363,
  -0.7966664774136267,
  -0.5255324099163290,
  -0.1834346424956498,
  0.1834346424956498,
  0.5255324099163290,
  0.7966664774136267,
  0.9602898564975363,
];
const GL_W = [
  0.1012285362903763,
  0.2223810344533745,
  0.3137066458778873,
  0.3626837833783620,
  0.3626837833783620,
  0.3137066458778873,
  0.2223810344533745,
  0.1012285362903763,
];
const OTOL = 1e-10, ODEPTH = 24;

// Add w · (covered length inside each pixel column of row-height scanline y) into out[0..S).
function accumulateRow(quads, evenodd, y, w, out) {
  const cross = crossingsAt(y, quads).sort((p, q) => q[0] - p[0]);
  let W = 0, K = 0, hi = Infinity;
  for (let i = 0; i <= cross.length; i++) {
    const lo = i < cross.length ? cross[i][0] : -Infinity;
    if (evenodd ? (K & 1) === 1 : W !== 0) {
      const a = Math.max(lo, 0), b = Math.min(hi, S);
      for (let c = Math.max(0, Math.floor(a)); c < b; c++) {
        out[c] += w * (Math.min(b, c + 1) - Math.max(a, c));
      }
    }
    if (i < cross.length) {
      W += cross[i][1];
      K++;
      hi = lo;
    }
  }
}

// 8-point Gauss–Legendre estimate of ∫[ya,yb] covered-length dy, added into out.
function glRow(quads, evenodd, ya, yb, out) {
  const h = (yb - ya) / 2, m = (ya + yb) / 2;
  for (let g = 0; g < 8; g++) accumulateRow(quads, evenodd, m + h * GL_X[g], h * GL_W[g], out);
}

// Adaptive wrapper: accept the two-half estimate when it agrees with the whole to OTOL (per column), else
// bisect. The top level always bisects once — a single whole-vs-halves comparison can be fooled at a kink
// (both estimates landing on the same wrong value), and one mandatory split re-tests it at shifted node
// positions. Returns the sum of accepted disagreements — a bound on this y-range's quadrature error.
function adaptRow(quads, evenodd, ya, yb, depth, out) {
  const whole = new Float64Array(S), halves = new Float64Array(S), ym = (ya + yb) / 2;
  glRow(quads, evenodd, ya, yb, whole);
  glRow(quads, evenodd, ya, ym, halves);
  glRow(quads, evenodd, ym, yb, halves);
  let d = 0;
  for (let c = 0; c < S; c++) d = Math.max(d, Math.abs(whole[c] - halves[c]));
  if ((depth === 0 || d > OTOL) && depth < ODEPTH) {
    return adaptRow(quads, evenodd, ya, ym, depth + 1, out) +
      adaptRow(quads, evenodd, ym, yb, depth + 1, out);
  }
  for (let c = 0; c < S; c++) out[c] += halves[c];
  return d;
}

/**
 * The box filter, computed to high precision (see the section comment). Returns the coverage and the
 * oracle's own convergence residual: `refErr` bounds the worst pixel's quadrature error, so downstream
 * claims can state the reference's noise instead of assuming it negligible.
 *
 * @returns {{ cov: Float64Array, refErr: number }}
 */
export function oracleCoverage(quads, evenodd) {
  // Analytic breakpoints, two families. (1) Heights where the covered length can JUMP: a curve starts,
  // ends, or turns around (endpoint / y-extremum heights). (2) Heights where a per-COLUMN covered length
  // merely kinks: a crossing slides over an integer x (a pixel boundary) — solvable per quad per integer,
  // so the quadrature between breakpoints sees an analytic integrand instead of chasing kinks by bisection.
  const ys = [];
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i],
      y0 = quads[i + 1],
      cx = quads[i + 2],
      cy = quads[i + 3],
      x1 = quads[i + 4],
      y1 = quads[i + 5];
    ys.push(y0, y1);
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0);
    if (Math.abs(a) > 1e-12) {
      const t = -b / (2 * a);
      if (t > 0 && t < 1) ys.push((1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1);
    }
    const yAt = (t) => (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1;
    const ax = x0 - 2 * cx + x1, bx = 2 * (cx - x0);
    if (Math.abs(ax) < 1e-12 && Math.abs(bx) < 1e-12) continue; // x(t) constant: its crossing never moves
    const lo = Math.max(0, Math.ceil(Math.min(x0, cx, x1))),
      hi = Math.min(S, Math.floor(Math.max(x0, cx, x1)));
    for (let c = lo; c <= hi; c++) {
      const cc = x0 - c;
      if (Math.abs(ax) < 1e-12) {
        const t = -cc / bx;
        if (t > 0 && t < 1) ys.push(yAt(t));
      } else {
        const disc = bx * bx - 4 * ax * cc;
        if (disc < 0) continue;
        const q = -0.5 * (bx + (bx >= 0 ? 1 : -1) * Math.sqrt(disc));
        for (const t of [q / ax, q !== 0 ? cc / q : q / ax]) {
          if (t > 0 && t < 1) ys.push(yAt(t));
        }
      }
    }
  }
  ys.sort((p, q) => p - q);

  const cov = new Float64Array(S * S), rowAcc = new Float64Array(S);
  let refErr = 0;
  for (let r = 0; r < S; r++) {
    const breaks = [r];
    for (const y of ys) {
      if (y > r && y < r + 1 && y - breaks[breaks.length - 1] > 1e-12) breaks.push(y);
    }
    breaks.push(r + 1);
    rowAcc.fill(0);
    let rowErr = 0;
    for (let s = 0; s + 1 < breaks.length; s++) {
      rowErr += adaptRow(quads, evenodd, breaks[s], breaks[s + 1], 0, rowAcc);
    }
    refErr = Math.max(refErr, rowErr);
    cov.set(rowAcc, r * S);
  }
  return { cov, refErr };
}

// ── compare ─────────────────────────────────────────────────────────────────────────────────────────
/**
 * The pixels a curve actually affects: walk every quad densely, mark each visited pixel and its 3×3
 * neighbourhood. Interior and exterior pixels — where every correct renderer scores an exact 0 or 1 and a
 * cell-wide mean is padded toward zero — stay unmarked; error statistics restricted to this mask describe
 * the partially-covered edge pixels the reviewers actually care about. Geometric (not derived from any
 * renderer's output), so it also covers boundaries whose true coverage is 0 or 1 — a duplicated even-odd
 * contour, a pixel-aligned edge — where a renderer can still err.
 */
export function edgeMaskFor(quads) {
  const mask = new Uint8Array(S * S);
  const mark = (x, y) => {
    const c = Math.floor(x), r = Math.floor(y);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cc = c + dx, rr = r + dy;
        if (cc >= 0 && cc < S && rr >= 0 && rr < S) mask[rr * S + cc] = 1;
      }
    }
  };
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i],
      y0 = quads[i + 1],
      cx = quads[i + 2],
      cy = quads[i + 3],
      x1 = quads[i + 4],
      y1 = quads[i + 5];
    const ext = Math.abs(x0 - cx) + Math.abs(cx - x1) + Math.abs(y0 - cy) + Math.abs(cy - y1);
    const n = Math.min(4096, Math.max(2, Math.ceil(ext * 2))); // a sample at least every half-pixel of hull extent
    for (let k = 0; k <= n; k++) {
      const t = k / n, u = 1 - t;
      mark(u * u * x0 + 2 * u * t * cx + t * t * x1, u * u * y0 + 2 * u * t * cy + t * t * y1);
    }
  }
  return mask;
}

/**
 * |Δcoverage| statistics of a renderer against the oracle: cell-wide mean and max, then mean / median / p99
 * restricted to the edge mask. `edgeErrs` (the raw masked errors, unsorted) rides along so boots can pool
 * shapes into aggregate quantiles.
 */
export function statsVs(cov, oracle, mask) {
  let sum = 0, max = 0, bad = 0, edgeSum = 0;
  const edge = [];
  for (let i = 0; i < cov.length; i++) {
    const e = Math.abs(cov[i] - oracle[i]);
    // A float readback can very occasionally return a non-finite value (transient GPU fault). The old
    // 8-bit path silently clamped these; here they are excluded from the stats but COUNTED, so a run that
    // hit one says so instead of poisoning every aggregate with NaN.
    if (!Number.isFinite(e)) {
      bad++;
      continue;
    }
    sum += e;
    if (e > max) max = e;
    if (mask[i]) {
      edge.push(e);
      edgeSum += e;
    }
  }
  const edgeErrs = Float64Array.from(edge);
  const sorted = Float64Array.from(edgeErrs).sort();
  const edgeN = edgeErrs.length;
  const q = (p) => (edgeN ? sorted[Math.floor(p * (edgeN - 1))] : 0);
  return {
    mean: sum / (cov.length - bad || 1),
    max,
    bad,
    edgeN,
    edgeMean: edgeN ? edgeSum / edgeN : 0,
    p50: q(0.5),
    p99: q(0.99),
    edgeErrs,
  };
}

// ── report images: S×S RGBA8 (boots upscale for PNGs or let CSS scale a canvas) ────────────────────────
const mapRGBA = (rgbAt) => {
  const d = new Uint8Array(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    const [r, g, b] = rgbAt(i);
    const o = i * 4;
    d[o] = r;
    d[o + 1] = g;
    d[o + 2] = b;
    d[o + 3] = 255;
  }
  return d;
};
/** Coverage as grayscale (white = covered). */
export const grayRGBA = (cov) =>
  mapRGBA((i) => {
    const v = Math.round(cov[i] * 255);
    return [v, v, v];
  });
/** |Δcoverage| amplified ×AMP so faint differences show (hot orange). */
export const diffRGBA = (a, b) =>
  mapRGBA((i) => {
    const v = Math.round(Math.min(Math.abs(a[i] - b[i]) * AMP, 1) * 255);
    return [v, Math.round(v * 0.28), Math.round(v * 0.12)];
  });

export const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');

/**
 * Run the whole suite, yielding one result per shape as it completes — so a boot can stream rows into a
 * console table (Deno) or the page (browser) while later shapes are still rendering. Each result carries
 * the four coverage fields, the oracle's own residual (`refErr`), and per-renderer statsVs against the
 * oracle (see statsVs for the fields).
 *
 * @param {object} o
 * @param {object} o.font              a parsed font (see font.js loadFont/parseFont)
 * @param {Function} o.createContext2D (w, h) → a 2D canvas context in the host environment
 * @param {GPUDevice} o.device         a shared WebGPU device (see gpu.js requestDevice)
 * @param {boolean} [o.exact]          render ours with the shader's EXACT_MODE override (see ourCoverage);
 *                                     slug, canvas and the oracle are unaffected — they're the yardstick
 */
export async function* validateShapes({ font, createContext2D, device, exact = false }) {
  for (const { label, quads, evenodd = false, segments, fold = false } of buildShapes(font)) {
    const ours = await ourCoverage(device, quads, evenodd, exact);
    const slug = await slugCoverage(device, quads, evenodd);
    const canvas = canvasCoverage(createContext2D, quads, evenodd, segments);
    const { cov: oracle, refErr } = oracleCoverage(quads, evenodd);
    const mask = edgeMaskFor(quads);
    yield {
      label,
      fold,
      ours,
      slug,
      canvas,
      oracle,
      refErr,
      oursVsOracle: statsVs(ours, oracle, mask),
      slugVsOracle: statsVs(slug, oracle, mask),
      canvasVsOracle: statsVs(canvas, oracle, mask),
    };
  }
}
