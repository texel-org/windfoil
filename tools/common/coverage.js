// coverage.js — the ways this repo can turn one shape into per-pixel coverage, one function each.
//
// All four take the same flat quads (./shapes.js) and return the same thing: a Float64Array of size*size
// values in 0..1, linear, 1 = fully covered. Because the input geometry and the output format are identical,
// any two of them subtract directly — which is the whole basis of tools/validate.js, tools/failure.js and
// tools/comparison.js.
//
//   windfoilCoverage — the windfoil shader itself (src/windfoil.wgsl via renderToRGBA), 8-bit readback.
//   slugCoverage     — the benchmark's Slug port (bench/slug.wgsl): the other analytic AA model, same pipeline.
//   canvasCoverage   — a host 2D canvas: @napi-rs/canvas (Skia, i.e. Chrome's rasteriser) under Deno, or the
//                      engine's own rasteriser in a browser.
//   pointCoverage    — the mathematical box filter, estimated by point-sampling an N×N grid per pixel with
//                      winding computed by ray casting the RAW curves. It shares no code and no model with
//                      the shader, so it is a genuine independent reference rather than a self-comparison,
//                      and it is right on the cases the winding fold gets wrong. Its own noise is ~1/N, so
//                      it settles the shape of an error and any bias; at N=1 it degenerates to the classic
//                      binary "is the pixel centre inside?" fill, which is a useful renderer in its own right.

import { renderToRGBA } from '../../src/gpu.js';
import { pushMonotonePieces } from '../../src/geometry.js';
import { bandPieces } from '../../src/bands.js';
import { bandSlugShape, loadSlugShaderCode } from '../../bench/slug.js';
import { bboxOf } from './shapes.js';

// A coverage buffer read out of an RGBA8 render: white-on-black, so the red channel IS the coverage.
function readRGBA(rgba, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = rgba[i * 4] / 255;
  return out;
}

// ── 1. the windfoil shader ──────────────────────────────────────────────────────────────────────────────
/** Split a shape into monotone pieces and file them into one instance's row bands (the atlas of one shape). */
export function buildWindfoilScene(quads, evenodd = false, scale = 1) {
  const pieces = [];
  for (let i = 0; i < quads.length; i += 6) pushMonotonePieces(quads.slice(i, i + 6), pieces);
  const [x0, y0, x1, y1] = bboxOf(pieces);
  const curveOut = [], rowOut = [];
  const { rowBase, bandCount, bandH, invH } = bandPieces(pieces, y0, y1, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array(
    [0, 0, scale, rule, x0, y0, x1, y1, 1, 1, 1, 1, rowBase, bandCount, bandH, invH],
  );
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

/**
 * @param {number[]} quads
 * @param {object} o
 * @param {number} o.size            render size in px (square)
 * @param {boolean} [o.evenodd]
 * @param {GPUDevice} [o.device]     reuse a device across many renders (one is requested per call otherwise)
 * @param {boolean} [o.exact]        compile the shader's EXACT_MODE override on: in-shader sampling of the
 *   TRUE fill rule on an EXACT_GRID×EXACT_GRID grid per pixel instead of the scalar winding fold (and it
 *   bypasses the minification guard). Correct on the winding-fold failure cases; ordinary AA edges pick up
 *   the grid's sub-sample quantisation, so it is an offline correctness mode, not a quality upgrade. The
 *   override is compiled out of the normal pipeline, so the fast path costs nothing when it is off.
 * @param {[number, number]} [o.style] coverage-style (gamma, sharp); [1, 1] = exact (identity)
 */
export async function windfoilCoverage(quads, { size, evenodd = false, device, exact = false, style } = {}) {
  const { curves, rows, instances } = buildWindfoilScene(quads, evenodd);
  const rgba = await renderToRGBA({
    device,
    constants: exact ? { EXACT_MODE: 1 } : undefined,
    width: size, height: size, background: [0, 0, 0, 1],
    curves, rows, instances, instanceCount: 1, style,
  });
  return readRGBA(rgba, size * size);
}

// ── 2. Slug (bench/slug.wgsl) — the other analytic AA model, on the same GPU pipeline ────────────────────
// Whole quads into Slug's dual band sets (bench/slug.js); the instance carries both band headers (20 floats).
export function buildSlugScene(quads, evenodd = false) {
  const bbox = bboxOf(quads);
  const curveOut = [], rowOut = [];
  const s = bandSlugShape(quads, bbox, curveOut, rowOut);
  const rule = evenodd ? 1 : 0;
  const instances = new Float32Array([
    0, 0, 1, rule, ...bbox, 1, 1, 1, 1,
    s.hRowBase, s.hBandCount, s.y0, s.invH,
    s.vRowBase, s.vBandCount, s.rotY0, s.invW,
  ]);
  return { curves: new Float32Array(curveOut), rows: new Uint32Array(rowOut), instances };
}

export async function slugCoverage(quads, { size, evenodd = false, device } = {}) {
  const { curves, rows, instances } = buildSlugScene(quads, evenodd);
  const rgba = await renderToRGBA({
    device, code: await loadSlugShaderCode(),
    width: size, height: size, background: [0, 0, 0, 1],
    curves, rows, instances, instanceCount: 1,
  });
  return readRGBA(rgba, size * size);
}

// ── 3. a host 2D canvas (Skia under Deno via @napi-rs/canvas; the engine's own in a browser) ─────────────
/**
 * @param {(w: number, h: number) => CanvasRenderingContext2D} createContext2D
 * @param {number[]} quads
 * @param {object} o
 * @param {number} o.size
 * @param {boolean} [o.evenodd]
 * @param {number[]} [o.segments] flat [x0,y0,x1,y1,w] centerlines to STROKE (butt caps) instead of filling
 *   the quads. A butt-capped stroke IS the rectangle the quads describe, so this measures the host's stroke
 *   pipeline — including any thin-stroke/hairline special case — against the identical exact shape.
 */
export function canvasCoverage(createContext2D, quads, { size, evenodd = false, segments } = {}) {
  const ctx = createContext2D(size, size);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
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
  } else {
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    let px = null, py = null;
    for (let i = 0; i < quads.length; i += 6) {
      const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3];
      const x1 = quads[i + 4], y1 = quads[i + 5];
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
  }
  return readRGBA(ctx.getImageData(0, 0, size, size).data, size * size);
}

// ── 4. the point-sampled box filter (independent: winding by ray casting the raw curves) ─────────────────
// Every crossing of a rightward ray at height py against the raw quads: its x position and winding sign.
// Fills the caller's scratch arrays and returns the count, so the per-scanline work allocates nothing.
function crossingsAt(py, quads, xs, sg) {
  let n = 0;
  for (let i = 0; i < quads.length; i += 6) {
    const x0 = quads[i], y0 = quads[i + 1], cx = quads[i + 2], cy = quads[i + 3];
    const x1 = quads[i + 4], y1 = quads[i + 5];
    if ((y0 < py && cy < py && y1 < py) || (y0 > py && cy > py && y1 > py)) continue; // hull y-reject
    const a = y0 - 2 * cy + y1, b = 2 * (cy - y0), c = y0 - py;
    let t0 = -1, t1 = -1;
    if (Math.abs(a) < 1e-9) {
      if (Math.abs(b) > 1e-12) t0 = -c / b;
    } else {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        t0 = (-b + sq) / (2 * a);
        t1 = (-b - sq) / (2 * a);
      }
    }
    for (let k = 0; k < 2; k++) {
      const t = k === 0 ? t0 : t1;
      if (t < 0 || t > 1) continue;
      const u = 1 - t;
      xs[n] = u * u * x0 + 2 * u * t * cx + t * t * x1;
      sg[n] = 2 * a * t + b >= 0 ? 1 : -1;
      n++;
    }
  }
  return n;
}

/** The winding number and crossing count at one point (a rightward ray cast against the raw curves). */
export function windingAt(px, py, quads) {
  const cap = (quads.length / 6) * 2;
  const xs = new Float64Array(cap), sg = new Int8Array(cap);
  const n = crossingsAt(py, quads, xs, sg);
  let W = 0, K = 0;
  for (let i = 0; i < n; i++) {
    if (xs[i] > px) {
      W += sg[i];
      K++;
    }
  }
  return { W, K };
}

/**
 * The box filter, estimated as: for each pixel, the fraction of an N×N grid of sub-sample points that fall
 * inside the shape. `samples: 1` puts that single sample at the pixel centre — the classic binary fill.
 *
 * One ray per sub-sample ROW serves every sample column on it: walk the columns right-to-left past the
 * crossings sorted rightmost-first, keeping the running signed winding W and crossing count K. Each sample
 * then sees exactly the crossings with x > its own — the same winding a per-point ray cast computes, without
 * re-solving the quads per column. A pixel that no crossing falls inside is uniform across all N of its
 * samples, so it is filled in one step; the per-sample loop only runs where an edge actually is.
 */
export function pointCoverage(quads, { size, evenodd = false, samples = 24 } = {}) {
  const N = samples;
  const out = new Float64Array(size * size);
  const cap = (quads.length / 6) * 2;
  const xs = new Float64Array(cap), sg = new Int8Array(cap); // crossings, unsorted
  const order = new Uint32Array(cap);
  const sx = new Float64Array(cap), ss = new Int8Array(cap); // …and sorted rightmost-first
  const off = new Float64Array(N); // sub-sample offsets within a pixel
  for (let i = 0; i < N; i++) off[i] = (i + 0.5) / N;

  for (let y = 0; y < size; y++) {
    const row = y * size;
    for (let j = 0; j < N; j++) {
      const n = crossingsAt(y + off[j], quads, xs, sg);
      const idx = order.subarray(0, n);
      for (let i = 0; i < n; i++) idx[i] = i;
      idx.sort((p, q) => xs[q] - xs[p]);
      for (let i = 0; i < n; i++) {
        sx[i] = xs[idx[i]];
        ss[i] = sg[idx[i]];
      }
      let ptr = 0, W = 0, K = 0;
      for (let x = size - 1; x >= 0; x--) {
        const inside = evenodd ? (K & 1) === 1 : W !== 0;
        // No crossing lands inside this pixel ⇒ all N samples share the winding we already have.
        if (ptr >= n || sx[ptr] <= x + off[0]) {
          if (inside) out[row + x] += N;
          continue;
        }
        let hit = 0;
        for (let i = N - 1; i >= 0; i--) {
          const px = x + off[i];
          while (ptr < n && sx[ptr] > px) {
            W += ss[ptr];
            K++;
            ptr++;
          }
          if (evenodd ? (K & 1) === 1 : W !== 0) hit++;
        }
        out[row + x] += hit;
      }
    }
  }
  // Divide (not multiply by a reciprocal): 1/(N*N) is not exact in binary, and the extra rounding is enough
  // to flip the odd 8-bit pixel when a coverage lands on a .5 boundary.
  for (let i = 0; i < out.length; i++) out[i] /= N * N;
  return out;
}

// ── compare ─────────────────────────────────────────────────────────────────────────────────────────────
/** Mean and worst-pixel |Δcoverage| between two buffers. */
export function stats(a, b) {
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    sum += e;
    if (e > max) max = e;
  }
  return { mean: sum / a.length, max };
}

/**
 * |Δ| restricted to the anti-aliasing band — the pixels where `truth` is strictly between empty and full.
 *
 * This is the number to report. A whole-image mean is not comparable across render sizes: the error lives on
 * edges, edge pixels grow with the perimeter (∝ size) while the divisor grows with the area (∝ size²), so the
 * mean falls ~1/size for reasons that have nothing to do with a renderer's quality — doubling the size makes
 * every engine look twice as good. Averaged over the band instead, the same renderer scores the same at
 * 128px and at 2048px, so the figure is a property of the algorithm rather than of the frame it was drawn in.
 */
export function edgeStats(cov, truth, mask = truth) {
  let n = 0, sum = 0, max = 0, signed = 0;
  for (let i = 0; i < cov.length; i++) {
    if (mask[i] > 0 && mask[i] < 1) {
      const d = cov[i] - truth[i], e = Math.abs(d);
      n++;
      sum += e;
      signed += d;
      if (e > max) max = e;
    }
  }
  // `signed` separates a one-directional bias (a compositing or colour-space mismatch, which pushes every
  // edge the same way) from ordinary anti-aliasing disagreement, which is close to zero-mean.
  return { band: n, mean: n ? sum / n : 0, max, signed: n ? signed / n : 0 };
}

/** Per-pixel |a − b|. */
export function absDiff(a, b) {
  const d = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) d[i] = Math.abs(a[i] - b[i]);
  return d;
}
